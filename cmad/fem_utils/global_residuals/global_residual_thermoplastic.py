import numpy as np
from jax import jit, vmap, jacfwd
import jax.numpy as jnp

from cmad.fem_utils.models.global_deriv_types import GlobalDerivType

from abc import ABC
from cmad.fem_utils.utils.fem_utils import assemble_global_fields, assemble_prescribed

from scipy.sparse import coo_matrix, csc_matrix
from jax.lax import while_loop

class Global_residual_thermoplastic(ABC):
    def __init__(
            self, global_resid_fun, local_resid_fun, elem_surf_traction, elem_surf_heat_flux,
            volume_conn, elem_points, eq_num, params, num_nodes_elem, num_nodes_surf, dof_node, 
            num_quad_pts, num_free_dof, disp_node, disp_val, init_xi ,num_pres_dof, 
            num_elem, pres_surf_traction_points, pres_surf_traction, surf_traction_vector, 
            pres_surf_flux_points, pres_surf_flux, init_temp, dt):

        mapped_axes = [0, 0, None, 0, 0, 0]
        self._global_resid = jit(global_resid_fun)
        self._global_jac = [jit(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU)),
                            jit(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU_prev)),
                            jit(jacfwd(global_resid_fun, argnums=GlobalDerivType.DParams)),
                            jit(jacfwd(global_resid_fun, argnums=GlobalDerivType.DXI)),
                            jit(jacfwd(global_resid_fun, argnums=GlobalDerivType.DXI_prev))]

        self._global_resid_batch = jit(vmap(global_resid_fun, in_axes=mapped_axes))
        self._global_jac_batch = [jit(vmap(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU),
                                           in_axes=mapped_axes)),
                                  jit(vmap(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU_prev),
                                           in_axes=mapped_axes)),
                                  jit(vmap(jacfwd(global_resid_fun, argnums=GlobalDerivType.DParams),
                                           in_axes=mapped_axes)),
                                  jit(vmap(jacfwd(global_resid_fun, argnums=GlobalDerivType.DXI),
                                           in_axes=mapped_axes)),
                                  jit(vmap(jacfwd(global_resid_fun, argnums=GlobalDerivType.DXI_prev),
                                           in_axes=mapped_axes))]

        self._local_resid = jit(local_resid_fun)
        self._local_jac = [jit(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU)),
                           jit(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU_prev)),
                           jit(jacfwd(local_resid_fun, argnums=GlobalDerivType.DParams)),
                           jit(jacfwd(local_resid_fun, argnums=GlobalDerivType.DXI)),
                           jit(jacfwd(local_resid_fun, argnums=GlobalDerivType.DXI_prev))]

        self._local_resid_batch = jit(vmap(local_resid_fun, in_axes=mapped_axes))
        self._local_jac_batch = [jit(vmap(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU),
                                                 in_axes=mapped_axes)),
                                 jit(vmap(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU_prev),
                                                 in_axes=mapped_axes)),
                                 jit(vmap(jacfwd(local_resid_fun, argnums=GlobalDerivType.DParams),
                                                 in_axes=mapped_axes)),
                                 jit(vmap(jacfwd(local_resid_fun, argnums=GlobalDerivType.DXI),
                                                 in_axes=mapped_axes)),
                                 jit(vmap(jacfwd(local_resid_fun, argnums=GlobalDerivType.DXI_prev),
                                                 in_axes=mapped_axes))]
        
        self._batch_local_newton = jit(vmap(self._local_resid_newton, in_axes=mapped_axes))
        self._batch_tang = jit(vmap(self._compute_tang, in_axes=mapped_axes))

        self._surf_traction_batch = jit(vmap(elem_surf_traction, in_axes=[0, None]))

        self._surf_flux_batch = jit(vmap(elem_surf_heat_flux, in_axes=[0, 0, None]))
        self._surf_flux_jacobian = jit(vmap(jacfwd(elem_surf_heat_flux), in_axes=[0, 0, None]))

        self._deriv_mode = GlobalDerivType.DNONE

        self._num_free_dof = num_free_dof
        self._volume_conn = volume_conn
        self._elem_points = elem_points
        self._params = params
        self._eq_num = eq_num
        self._num_quad_pts = num_quad_pts
        self._init_temp = init_temp
        self._dt = dt
        self._init_xi = init_xi
        self._num_elem = num_elem

        # dirichlet displacement and temperature boundary conditions
        self._disp_node = disp_node
        self._disp_val = disp_val
        self._num_pres_dof = num_pres_dof

        # traction boundary conditions
        self._pres_surf_traction_points = pres_surf_traction_points
        self._surf_traction_vector = surf_traction_vector
        self._pres_surf_traction = pres_surf_traction

        # surf heat flux boundary conditions
        self._pres_surf_flux_points = pres_surf_flux_points
        self._pres_surf_flux = pres_surf_flux

        # indices for element vector assembly
        global_indices = eq_num[volume_conn, :].transpose(0, 2, 1).reshape(volume_conn.shape[0], -1)
        global_free_indices = np.where(global_indices > 0, global_indices - 1, -1)
        flat_global_free_indices = global_free_indices.ravel()
        mask_vector = flat_global_free_indices >= 0
        global_free_indices_vector = flat_global_free_indices[mask_vector]

        self._mask_vector = mask_vector
        self._global_free_indices_vector = global_free_indices_vector

        # indices for element matrix assembly
        elem_dofs = num_nodes_elem * dof_node
        ii, jj = np.meshgrid(np.arange(elem_dofs), np.arange(elem_dofs),
                                indexing='ij')
        row_f = global_free_indices[:, ii]
        col_f = global_free_indices[:, jj]
        mask_f = (row_f >= 0) & (col_f >= 0)
        row_f = row_f[mask_f]
        col_f = col_f[mask_f]

        self._ii = ii
        self._jj = jj
        self._row_f = row_f
        self._col_f = col_f
        self._mask_f = mask_f

        # index arrays for traction vector assembly
        if not pres_surf_traction is None:
            surf_global_indices_all = eq_num[:, :-2][pres_surf_traction, :]. \
                transpose(0, 2, 1).reshape(pres_surf_traction.shape[0], -1)
            flat_surf_global_free_indices = np.where(surf_global_indices_all > 0,
                                                     surf_global_indices_all - 1, -1).ravel()
            self._surf_valid_free_mask = flat_surf_global_free_indices >= 0
            self._surf_global_indices = flat_surf_global_free_indices[self._surf_valid_free_mask]

        # index arrays for heat flux bc vector and jacobian assembly
        if not pres_surf_flux is None:
            # indices for vector assembly
            surf_flux_global_indices_all = eq_num[:, -2][pres_surf_flux]
            surf_flux_global_free_indices = np.where(surf_flux_global_indices_all > 0,
                                                    surf_flux_global_indices_all - 1, -1)
            flat_surf_flux_global_free_indices = surf_flux_global_free_indices.ravel()
            self._surf_flux_valid_free_mask = flat_surf_flux_global_free_indices >= 0
            self._surf_flux_global_indices = flat_surf_flux_global_free_indices[self._surf_flux_valid_free_mask]

            # indices for jacobian assembly
            ii_flux, jj_flux = np.meshgrid(np.arange(num_nodes_surf),
                                           np.arange(num_nodes_surf),
                                           indexing='ij')
            row_f_flux = surf_flux_global_free_indices[:, ii_flux]
            col_f_flux = surf_flux_global_free_indices[:, jj_flux]
            mask_f_flux = (row_f_flux >= 0) & (col_f_flux >= 0)
            row_f_flux = row_f_flux[mask_f_flux]
            col_f_flux = col_f_flux[mask_f_flux]

            self._ii_flux = ii_flux
            self._jj_flux = jj_flux
            self._row_f_flux = row_f_flux
            self._col_f_flux = col_f_flux
            self._mask_f_flux = mask_f_flux

        # data storage
        self._point_data = []
        self._cell_data = []

    def reset_xi(self):
        self._xi_elem = self._xi_elem_prev.copy()
    
    def compute_local_state_variables(self):
        variables = self._variables()
        self._xi_elem = np.asarray(self._batch_local_newton(*variables))
    
    def evaluate_tang(self):
        variables = self._variables()
        tang = self._batch_tang(*variables)
        self._tang = np.asarray(tang)
        if not self._pres_surf_flux is None:
            surf_vars = self._surf_theta, self._pres_surf_flux_points, self._params
            self._Jac_Q = np.asarray(self._surf_flux_jacobian(*surf_vars))

    def evaluate_global(self):

        variables = self._variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == GlobalDerivType.DNONE:
            self._R = np.asarray(self._global_resid_batch(*variables))
            self._Jac_R = None
            if not self._pres_surf_flux is None:
                surf_vars = self._surf_theta, self._pres_surf_flux_points, self._params.values
                self._Q = np.asarray(self._surf_flux_batch(*surf_vars))
        else:
            self._Jac_R = np.asarray(self._global_jac_batch[deriv_mode](*variables))

    def evaluate_local(self):

        variables = self._variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == GlobalDerivType.DNONE:
            self._C = np.asarray(self._local_resid_batch(*variables))
            self._Jac_C = None
        else:
            self._Jac_C = np.asarray(self._local_jac_batch[deriv_mode](*variables))

    def initialize_variables(self):
        num_free_dof = self._num_free_dof
        num_pres_dof = self._num_pres_dof

        # free dofs
        self._UF = np.zeros(num_free_dof)
        self._UF_prev = np.zeros(num_free_dof)

        # prescribed dofs
        self._UP = np.zeros(num_pres_dof)
        self._UP_prev = np.zeros(num_pres_dof)

        # local state variables
        self._xi_elem_prev = np.tile(self._init_xi, (self._num_elem, 1))
        self._xi_elem = self._xi_elem_prev.copy()

        # initialize temperatures
        for i, temp in enumerate(self._init_temp):
            eqn_number = self._eq_num[i][-2]
            if eqn_number > 0:
                self._UF_prev[eqn_number - 1] = temp
            else :
                self._UP_prev[-eqn_number - 1] = temp
        self._UF = self._UF_prev.copy()
        self._elem_init_temp = self._init_temp[self._volume_conn]

    def set_prescribed_dofs(self, step):
        self._UP = assemble_prescribed(self._num_pres_dof, self._disp_node,
                                       self._disp_val[:, step], self._eq_num)

    def set_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        UUR_prev = assemble_global_fields(self._eq_num, self._UF_prev, self._UP_prev)
        self._u_elem = UUR[self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

        self._u_elem_prev = UUR_prev[self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

        self._surf_theta = UUR[:, -2][self._pres_surf_flux]

    def scatter_rhs(self):
        RF_global = np.zeros(self._num_free_dof)
        np.add.at(RF_global, self._global_free_indices_vector,
                  self._R.ravel()[self._mask_vector])
        if not self._pres_surf_flux is None:
            np.add.at(RF_global, self._surf_flux_global_indices,
                      self._Q.ravel()[self._surf_flux_valid_free_mask])
        return RF_global - self._FF

    def scatter_lhs(self):
        KFF = coo_matrix((self._tang[:, self._ii, self._jj][self._mask_f],
                          (self._row_f, self._col_f)),
                          shape=(self._num_free_dof, self._num_free_dof))

        if not self._pres_surf_flux is None:
            KFF = KFF + coo_matrix((self._Jac_Q[:, self._ii_flux, self._jj_flux][self._mask_f_flux],
                                    (self._row_f_flux, self._col_f_flux)),
                                   shape=(self._num_free_dof, self._num_free_dof))

        return csc_matrix(KFF)

    def compute_surf_tractions(self, step):
        self._FF = np.zeros(self._num_free_dof)
        if not self._pres_surf_traction is None:
            FEL = self._surf_traction_batch(self._pres_surf_traction_points,
                                            self._surf_traction_vector[step])
            np.add.at(self._FF, self._surf_global_indices,
                      FEL.ravel()[self._surf_valid_free_mask])

    def add_to_UF(self, delta):
        self._UF += delta
    
    def set_UF(self, UF):
        self._UF = UF.copy()
    
    def get_UF(self):
        return self._UF.copy()

    def advance_model(self):
        self._UF_prev = self._UF.copy()
        self._UP_prev = self._UP.copy()
        self._xi_elem_prev = self._xi_elem.copy()

    def save_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        disp_field = UUR[:, :-2]
        pressure_field = UUR[:, -1]
        temperature_field = UUR[:, -2]

        self._point_data.append({'displacement_field': disp_field,
                                 'temperature_field': temperature_field,
                                 'pressure_field': pressure_field })
        
        plastic_strain = self.average_plastic_strain()
        self._cell_data.append({'eq_plastic_strain': plastic_strain})

    def get_data(self):
        return self._point_data, self._cell_data

    def seed_none(self):
        self._deriv_mode = GlobalDerivType.DNONE

    def seed_U(self):
        self._deriv_mode = GlobalDerivType.DU
    
    def average_plastic_strain(self):
        return np.mean(self._xi_elem[:, -self._num_quad_pts:], axis=1).reshape(-1, 1)

    def _compute_tang(self, u, u_prev, params, xi, xi_prev, elem_points):
        variables = u, u_prev, params, xi, xi_prev, elem_points
        dc_du = self._local_jac[GlobalDerivType.DU](*variables)
        dc_dxi = self._local_jac[GlobalDerivType.DXI](*variables)
        dc_dxi_inv = jnp.linalg.inv(dc_dxi)
        dxi_du = dc_dxi_inv @ -dc_du

        dR_du = self._global_jac[GlobalDerivType.DU](*variables)
        dR_dxi = self._global_jac[GlobalDerivType.DXI](*variables)
        return dR_du + dR_dxi @ dxi_du
    
    def _variables(self):
        return self._u_elem, self._u_elem_prev, self._params.values, self._xi_elem, \
            self._xi_elem_prev, self._elem_points
    
    def C(self):
        return self._C
    
    def R(self):
        return self._R

    # newton solve for local state variables
    def _local_resid_newton(
            self, u, u_prev, params, xi_0, xi_prev, elem_points):
        max_iters = 20
        tol = 1e-10
        init_state = (xi_0, xi_prev, params, u, u_prev, elem_points, 0, max_iters, tol)
        final_state = while_loop(self._newton_cond_fun, self._newton_body_fun, init_state)
        return final_state[0]

    def _newton_cond_fun(self, state):
        xi, xi_prev, params, u, u_prev, elem_points, iter_count, max_iters, tol = state
        C = self._local_resid(u, u_prev, params, xi, xi_prev, elem_points)
        return jnp.logical_and(jnp.linalg.norm(C) > tol, iter_count < max_iters)

    def _newton_body_fun(self, state):
        xi, xi_prev, params, u, u_prev, elem_points, iter_count, max_iters, tol = state
        C = self._local_resid(u, u_prev, params, xi, xi_prev, elem_points)
        DC = self._local_jac[GlobalDerivType.DXI](u, u_prev, params, xi, xi_prev, elem_points)
        xi_new = xi - jnp.linalg.solve(DC, C)
        return xi_new, xi_prev, params, u, u_prev, elem_points, iter_count + 1, max_iters, tol














