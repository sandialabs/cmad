import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import pyvista as pv
import time

from cmad.fem_utils.models.global_deriv_types import GlobalDerivType

from abc import ABC
from cmad.fem_utils.utils.fem_utils import (assemble_global_fields,
                                      assemble_prescribed)
from cmad.models.deformation_types import DefType

from scipy.sparse import coo_matrix, csc_matrix
from jax.lax import while_loop

class Global_residual_plasticity(ABC):
    def __init__(
            self, global_resid_fun, local_resid_fun, elem_surf_traction, volume_conn,
            nodal_coords, eq_num, params, num_nodes_elem, dof_node, num_quad_pts, 
            num_free_dof, num_pres_dof, num_elem, disp_node, disp_val, init_xi,
            pres_surf_traction_points, pres_surf_traction, surf_traction_vector, def_type):

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

        self._local_hessian = [jit(jacfwd(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU),
                                          argnums=GlobalDerivType.DU)),
                               jit(jacfwd(jacfwd(local_resid_fun, argnums=GlobalDerivType.DXI),
                                          argnums=GlobalDerivType.DXI)),
                               jit(jacfwd(jacfwd(local_resid_fun, argnums=GlobalDerivType.DU),
                                          argnums=GlobalDerivType.DXI))]

        self._global_hessian = [jit(jacfwd(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU),
                                           argnums=GlobalDerivType.DU)),
                                jit(jacfwd(jacfwd(global_resid_fun, argnums=GlobalDerivType.DXI),
                                           argnums=GlobalDerivType.DXI)),
                                jit(jacfwd(jacfwd(global_resid_fun, argnums=GlobalDerivType.DU),
                                           argnums=GlobalDerivType.DXI))]

        self._batch_local_newton = jit(vmap(self._local_resid_newton, in_axes=mapped_axes))
        self._batch_tang = jit(vmap(self._compute_tang, in_axes=mapped_axes))

        # Halley's method
        mapped_halley_axes = [0, 0, None, 0, 0, 0, 0, 0, 0, 0]
        self._batch_halley_correction = jit(vmap(self._compute_halley,
                                                 in_axes=mapped_halley_axes))

        mapped_halley_fd_axes = [0, 0, None, 0, 0, 0, 0]
        self._batch_halley_correction_fd = jit(vmap(self._compute_halley_fd,
                                                    in_axes=mapped_halley_fd_axes))

        self._surf_traction_batch = jit(vmap(elem_surf_traction, in_axes=[0, None]))

        self._deriv_mode = GlobalDerivType.DNONE
        self._def_type = def_type

        self._num_free_dof = num_free_dof
        self._volume_conn = volume_conn
        self._nodal_coords = nodal_coords
        self._elem_points = nodal_coords[volume_conn, :]
        self._params = params
        self._eq_num = eq_num
        self._num_quad_pts = num_quad_pts
        self._init_xi = init_xi
        self._num_elem = num_elem

        if (def_type == DefType.PLANE_STRESS):
            self._elem_points = nodal_coords[:, :-1][volume_conn, :]

        # displacement and pressure boundary conditions
        self._disp_node = disp_node
        self._disp_val = disp_val
        self._num_pres_dof = num_pres_dof

        # traction boundary conditions
        self._pres_surf_traction_points = pres_surf_traction_points
        self._surf_traction_vector = surf_traction_vector
        self._pres_surf_traction = pres_surf_traction

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

        # indices for traction vector assembly
        if not pres_surf_traction is None:
            surf_global_indices_all = eq_num[:, :-1][pres_surf_traction, :]. \
                transpose(0, 2, 1).reshape(pres_surf_traction.shape[0], -1)
            flat_surf_global_free_indices = np.where(surf_global_indices_all > 0,
                                                    surf_global_indices_all - 1, -1).ravel()
            self._surf_valid_free_mask = flat_surf_global_free_indices >= 0
            self._surf_global_indices = flat_surf_global_free_indices[self._surf_valid_free_mask]

        # data storage
        self._point_data = []
        self._cell_data = []
        
    def initialize_variables(self):
        self._UF = np.zeros(self._num_free_dof)
        self._UF_prev = np.zeros(self._num_free_dof)
        self._UP_prev = np.zeros(self._num_pres_dof)
        self._xi_elem_prev = np.tile(self._init_xi, (self._num_elem, 1))
        self._xi_elem = self._xi_elem_prev.copy()

    def reset_xi(self):
        self._xi_elem = self._xi_elem_prev.copy()

    def compute_local_state_variables(self):
        variables = self._variables()
        self._xi_elem = np.asarray(self._batch_local_newton(*variables))

    def evaluate_tang(self):
        variables = self._variables()
        tang, dxi_du, dc_dxi_inv, dR_dxi = self._batch_tang(*variables)
        self._tang = np.asarray(tang)
        self._dxi_du = np.asarray(dxi_du)
        self._dc_dxi_inv = np.asarray(dc_dxi_inv)
        self._dR_dxi = np.asarray(dR_dxi)

    def evaluate_global(self):

        variables = self._variables()
        deriv_mode = self._deriv_mode

        if deriv_mode == GlobalDerivType.DNONE:
            self._R = np.asarray(self._global_resid_batch(*variables))
            self._Jac_R = None
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

    def R(self):
        return self._R

    def C(self):
        return self._C

    def Jac_global(self):
        return self._Jac_R

    def Jac_local(self):
        return self._Jac_C

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

    def save_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        disp_field = UUR[:, :-1]
        pressure_field = UUR[:, -1]

        if self._def_type == DefType.PLANE_STRESS:
            disp_field = np.append(disp_field, np.zeros((len(self._eq_num), 1)), axis=1)

        self._point_data.append({'displacement_field': disp_field,
                                 'pressure_field': pressure_field})

        plastic_strain = self.average_plastic_strain()
        self._cell_data.append({'eq_plastic_strain': plastic_strain})

    def get_num_plastic_elements(self):
        plastic_strain = self.average_plastic_strain()
        num_plastic = np.sum(np.abs(plastic_strain) > 1e-13)
        print('Number of plastic elements: ', num_plastic)

    def compute_surf_tractions(self, step):
        self._FF = np.zeros(self._num_free_dof)
        if not self._pres_surf_traction is None:
            FEL = self._surf_traction_batch(self._pres_surf_traction_points,
                                               self._surf_traction_vector[step])
            np.add.at(self._FF, self._surf_global_indices, FEL.ravel()[self._surf_valid_free_mask])

    def get_data(self):
        return self._point_data, self._cell_data

    def add_to_UF(self, delta):
        self._UF += delta

    def get_UF(self):
        return self._UF.copy()

    def set_UF(self, UF):
        self._UF = UF.copy()

    def scatter_rhs(self):
        RF_global = np.zeros(self._num_free_dof)
        np.add.at(RF_global, self._global_free_indices_vector,
                  self._R.ravel()[self._mask_vector])
        return RF_global - self._FF

    def scatter_lhs(self):
        KFF = csc_matrix(coo_matrix((self._tang[:, self._ii, self._jj][self._mask_f],
                                    (self._row_f, self._col_f)),
                                    shape=(self._num_free_dof, self._num_free_dof)))
        return KFF

    def seed_none(self):
        self._deriv_mode = GlobalDerivType.DNONE

    def seed_U(self):
        self._deriv_mode = GlobalDerivType.DU

    def seed_xi(self):
        self._deriv_mode = GlobalDerivType.DXI

    def seed_xi_prev(self):
        self._deriv_mode = GlobalDerivType.DXI_prev

    def seed_params(self):
        self._deriv_mode = GlobalDerivType.DParams

    def advance_model(self):
        self._xi_elem_prev = self._xi_elem.copy()
        self._UF_prev = self._UF.copy()
        self._UP_prev = self._UP.copy()

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
        return dR_du + dR_dxi @ dxi_du, dxi_du, dc_dxi_inv, dR_dxi

    def initialize_plot(self):
        num_elem = self._volume_conn.shape[0]

        if self._def_type == DefType.PLANE_STRESS:
            cell_types = np.full(len(self._volume_conn), pv.CellType.TRIANGLE, dtype=np.uint8)
            cells = np.hstack((3 * np.ones((num_elem, 1)), self._volume_conn))
        else:
            cell_types = np.full(len(self._volume_conn), pv.CellType.TETRA, dtype=np.uint8)
            cells = np.hstack((4 * np.ones((num_elem, 1)), self._volume_conn))

        cells = cells.flatten().astype(int)
        self._grid = pv.UnstructuredGrid(cells, cell_types, self._nodal_coords.copy())

        self._grid.cell_data["plastic_strain"] = np.zeros(num_elem)

        self._plotter = pv.Plotter()
        self._mesh_actor = self._plotter.add_mesh(
            self._grid,
            scalars="plastic_strain",
            cmap="coolwarm",
            show_edges=True
        )
        self._plotter.show(interactive_update=True)

    def update_plot(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        disp_field = UUR[:, :-1]
        if self._def_type == DefType.PLANE_STRESS:
            disp_field = np.append(disp_field, np.zeros((len(self._eq_num), 1)), axis=1)
        pressure_field = UUR[:, -1]

        scale = 5.0
        deformed_coords = self._nodal_coords + scale * disp_field
        self._grid.points = deformed_coords

        equiv_plastic_strain = self.average_plastic_strain().flatten()
        self._grid.cell_data["plastic_strain"] = equiv_plastic_strain

        # updating the colorbar scaling
        max = np.max(equiv_plastic_strain)
        min = np.min(equiv_plastic_strain)
        self._mesh_actor.mapper.scalar_range = (min, max)

        # update plot
        self._plotter.update()
        self._plotter.render()

        # delay
        time.sleep(0.05)

    # newton solve for local state variables
    def _local_resid_newton(self, u, u_prev, params, xi_0, xi_prev, elem_points):
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

    def _variables(self):
        return self._u_elem, self._u_elem_prev, self._params.values, \
            self._xi_elem, self._xi_elem_prev, self._elem_points

    # functions for Halley's method
    #########################################

    def set_newton_increment(self, delta_F):
        delta_UR = assemble_global_fields(self._eq_num, delta_F, np.zeros(self._num_pres_dof))
        self._delta_elem = delta_UR[self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

    def _compute_halley(
            self, u, u_prev, params, xi, xi_prev, elem_points,
            dc_dxi_inv, dxi_du, dR_dxi, delta):

        variables = u, u_prev, params, xi, xi_prev, elem_points
        d2C_du2 = self._local_hessian[GlobalDerivType.DU_DU](*variables)
        d2C_dxi2 = self._local_hessian[GlobalDerivType.DXI_DXI](*variables)
        d2C_du_dxi = self._local_hessian[GlobalDerivType.DU_DXI](*variables)

        d2R_du2 = self._global_hessian[GlobalDerivType.DU_DU](*variables)
        d2R_dxi2 = self._global_hessian[GlobalDerivType.DXI_DXI](*variables)
        d2R_du_dxi = self._global_hessian[GlobalDerivType.DU_DXI](*variables)

        v_n = dxi_du @ delta
        a_j_v_n = jnp.outer(delta, v_n)
        a_j_a_k = jnp.outer(delta, delta)
        v_n_v_m = jnp.outer(v_n, v_n)

        # compute d2xi_du2-vector contraction
        cont_axes = ([1, 2], [0, 1])
        rhs = 2 * jnp.tensordot(d2C_du_dxi, a_j_v_n, axes=cont_axes) \
            + jnp.tensordot(d2C_dxi2, v_n_v_m, axes=cont_axes) \
            + jnp.tensordot(d2C_du2, a_j_a_k, axes=cont_axes)
        d2xi_du2_a_a = -dc_dxi_inv @ rhs

        # compute hessian-vector contraction
        d2R_dU2_a_a = 2 * jnp.tensordot(d2R_du_dxi, a_j_v_n, axes=cont_axes) \
                    + jnp.tensordot(d2R_dxi2, v_n_v_m, axes=cont_axes) \
                    + jnp.tensordot(d2R_du2, a_j_a_k, axes=cont_axes) \
                    + dR_dxi @ d2xi_du2_a_a

        return d2R_dU2_a_a

    def _compute_halley_fd(self, u, u_prev, params, xi, xi_prev, elem_points, delta):
        h = 1.e-3
        R_ref = self._global_resid(u, u_prev, params, xi, xi_prev, elem_points)

        u_plus = u + h * delta
        xi_plus = self._local_resid_newton(u_plus, u_prev, params, xi_prev, xi_prev, elem_points)
        R_plus = self._global_resid(u_plus, u_prev, params, xi_plus, xi_prev, elem_points)

        u_minus = u - h * delta
        xi_minus = self._local_resid_newton(u_minus, u_prev, params, xi_prev, xi_prev, elem_points)
        R_minus = self._global_resid(u_minus, u_prev, params, xi_minus, xi_prev, elem_points)

        return (R_plus - 2 * R_ref + R_minus) / h ** 2

    def evaluate_halley_correction(self):
        variables = self._halley_variables()
        halley_batch = np.asarray(self._batch_halley_correction(*variables))

        halley_global = np.zeros(self._num_free_dof)
        np.add.at(halley_global, self._global_free_indices_vector,
                  halley_batch.ravel()[self._mask_vector])

        return halley_global

    def evaluate_halley_correction_multi(self, num_batches):
        variables = self._halley_variables_multi(num_batches)
        
        halley_batch_all = np.empty_like(self._R)
        pos = 0
        for variable in variables:
            batch = np.asarray(self._batch_halley_correction(*variable))
            batch_size = batch.shape[0]
            halley_batch_all[pos:pos + batch_size] = batch
            pos += batch_size

        halley_global = np.zeros(self._num_free_dof)
        np.add.at(halley_global, self._global_free_indices_vector,
                  halley_batch_all.ravel()[self._mask_vector])

        return halley_global
    
    def evaluate_halley_correction_fd(self):
        variables = self._halley_variables_fd()
        halley_batch = np.asarray(self._batch_halley_correction_fd(*variables))
        halley_global = np.zeros(self._num_free_dof)
        np.add.at(halley_global, self._global_free_indices_vector,
                  halley_batch.ravel()[self._mask_vector])

        return halley_global

    def _halley_variables(self):
        return self._u_elem, self._u_elem_prev, self._params.values, \
            self._xi_elem, self._xi_elem_prev, self._elem_points, \
            self._dc_dxi_inv, self._dxi_du, self._dR_dxi, self._delta_elem

    def _halley_variables_multi(self, num_batches):
        vars = (
            self._u_elem,
            self._u_elem_prev,
            self._params.values,   
            self._xi_elem,
            self._xi_elem_prev,
            self._elem_points,
            self._dc_dxi_inv,
            self._dxi_du,
            self._dR_dxi,
            self._delta_elem
        )

        vars_split = []
        for i, var in enumerate(vars):
            if i == 2:  # index of self._params.values
                vars_split.append([var] * num_batches)
            else:
                vars_split.append(np.array_split(var, num_batches))

        result = list(zip(*vars_split))
        return result

    def _halley_variables_fd(self):
        return self._u_elem, self._u_elem_prev, self._params.values, \
            self._xi_elem, self._xi_elem_prev, self._elem_points, \
            self._delta_elem




















