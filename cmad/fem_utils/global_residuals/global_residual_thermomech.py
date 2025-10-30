import numpy as np
from jax import jit, vmap, jacfwd

from cmad.fem_utils.models.global_deriv_types import GlobalDerivType

from abc import ABC
from cmad.fem_utils.utils.fem_utils import assemble_global_fields, assemble_prescribed

from scipy.sparse import coo_matrix, csr_matrix

class Global_residual_thermomech(ABC):
    def __init__(
            self, resid_fun, elem_surf_traction, elem_surf_heat_flux, volume_conn, elem_points,
            eq_num, params, num_nodes_elem, num_nodes_surf, dof_node, num_free_dof, disp_node,
            disp_val, num_pres_dof, pres_surf_traction_points, pres_surf_traction,
            surf_traction_vector, pres_surf_flux_points, pres_surf_flux, init_temp, dt):

        mapped_indicies = [0, 0, 0, 0, None, 0, 0]
        self._global_resid = jit(vmap(resid_fun, in_axes=mapped_indicies))
        self._global_jac = jit(vmap(jacfwd(resid_fun), in_axes=mapped_indicies))

        self._surf_traction_batch = jit(vmap(elem_surf_traction, in_axes=[0, None]))

        self._surf_flux_batch = jit(vmap(elem_surf_heat_flux, in_axes=[0, 0, None]))
        self._surf_flux_jacobian = jit(vmap(jacfwd(elem_surf_heat_flux), in_axes=[0, 0, None]))

        self._deriv_mode = GlobalDerivType.DNONE

        self._num_free_dof = num_free_dof
        self._volume_conn = volume_conn
        self._elem_points = elem_points
        self._params = params
        self._eq_num = eq_num
        self._init_temp = init_temp
        self._dt = dt

        # free dofs
        self._UF = np.zeros(num_free_dof)
        self._UF_prev = np.zeros(num_free_dof)

        # intermediate variables
        self._aF_prev = np.zeros(num_free_dof)
        self._vF_prev = np.zeros(num_free_dof)

        # prescribed dofs
        self._UP = np.zeros(num_pres_dof)
        self._UP_prev = np.zeros(num_pres_dof)
        self._aP_prev = np.zeros(num_pres_dof)
        self._vP_prev = np.zeros(num_pres_dof)

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
            surf_global_indices_all = eq_num[:, :-1][pres_surf_traction, :]. \
                transpose(0, 2, 1).reshape(pres_surf_traction.shape[0], -1)
            flat_surf_global_free_indices = np.where(surf_global_indices_all > 0,
                                                     surf_global_indices_all - 1, -1).ravel()
            self._surf_valid_free_mask = flat_surf_global_free_indices >= 0
            self._surf_global_indices = flat_surf_global_free_indices[self._surf_valid_free_mask]

        # index arrays for heat flux bc vector and jacobian assembly
        if not pres_surf_flux is None:
            # indices for vector assembly
            surf_flux_global_indices_all = eq_num[:, -1][pres_surf_flux]
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

    def initialize_variables(self):
        for i, temp in enumerate(self._init_temp):
            eqn_number = self._eq_num[i][-1]
            if eqn_number > 0:
                self._UF_prev[eqn_number - 1] = temp
        self._UF = self._UF_prev.copy()
        self._UP_prev = assemble_prescribed(self._num_pres_dof, self._disp_node,
                                            self._disp_val[:, 0], self._eq_num)
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

        aUR_prev = assemble_global_fields(self._eq_num, self._aF_prev, self._aP_prev)
        self._a_elem_prev = aUR_prev[:, :-1][self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

        vUR_prev = assemble_global_fields(self._eq_num, self._vF_prev, self._vP_prev)
        self._v_elem_prev = vUR_prev[:, :-1][self._volume_conn, :].transpose(0, 2, 1) \
            .reshape(self._volume_conn.shape[0], -1)

        self._surf_theta = UUR[:, -1][self._pres_surf_flux]

    def _variables(self):
        return self._u_elem, self._u_elem_prev, self._v_elem_prev, \
            self._a_elem_prev, self._params, self._elem_points, self._elem_init_temp

    def evaluate(self):
        variables = self._variables()
        if self._deriv_mode == GlobalDerivType.DNONE:
            self._R = np.asarray(self._global_resid(*variables))
            if not self._pres_surf_flux is None:
                surf_vars = self._surf_theta, self._pres_surf_flux_points, self._params
                self._Q = np.asarray(self._surf_flux_batch(*surf_vars))

        elif self._deriv_mode == GlobalDerivType.DU:
            self._Jac_R = np.asarray(self._global_jac(*variables))
            if not self._pres_surf_flux is None:
                surf_vars = self._surf_theta, self._pres_surf_flux_points, self._params
                self._Jac_Q = np.asarray(self._surf_flux_jacobian(*surf_vars))

    def scatter_rhs(self):
        RF_global = np.zeros(self._num_free_dof)
        np.add.at(RF_global, self._global_free_indices_vector,
                  self._R.ravel()[self._mask_vector])
        if not self._pres_surf_flux is None:
            np.add.at(RF_global, self._surf_flux_global_indices,
                      self._Q.ravel()[self._surf_flux_valid_free_mask])
        return RF_global - self._FF

    def scatter_lhs(self):
        KFF = coo_matrix((self._Jac_R[:, self._ii, self._jj][self._mask_f],
                          (self._row_f, self._col_f)),
                          shape=(self._num_free_dof, self._num_free_dof))

        if not self._pres_surf_flux is None:
            KFF = KFF + coo_matrix((self._Jac_Q[:, self._ii_flux, self._jj_flux][self._mask_f_flux],
                                    (self._row_f_flux, self._col_f_flux)),
                                   shape=(self._num_free_dof, self._num_free_dof))

        return csr_matrix(KFF)

    def compute_surf_tractions(self, step):
        self._FF = np.zeros(self._num_free_dof)
        if not self._pres_surf_traction is None:
            FEL = self._surf_traction_batch(self._pres_surf_traction_points,
                                            self._surf_traction_vector[step])
            np.add.at(self._FF, self._surf_global_indices,
                      FEL.ravel()[self._surf_valid_free_mask])

    def add_to_UF(self, delta):
        self._UF += delta

    def advance_model(self):
        dt = self._dt

        aF_new = 4 / dt ** 2 * (self._UF - self._UF_prev - self._vF_prev * dt) - self._aF_prev
        vF_new = self._vF_prev + 1 / 2 * (self._aF_prev + aF_new) * dt

        self._UF_prev = self._UF.copy()
        self._aF_prev = aF_new.copy()
        self._vF_prev = vF_new.copy()

        aP_new = 4 / dt ** 2 * (self._UP - self._UP_prev - self._vP_prev * dt) - self._aP_prev
        vP_new = self._vP_prev + 1 / 2 * (self._aP_prev + aP_new) * dt

        self._UP_prev = self._UP.copy()
        self._aP_prev = aP_new.copy()
        self._vP_prev = vP_new.copy()

    def save_global_fields(self):
        UUR = assemble_global_fields(self._eq_num, self._UF, self._UP)
        disp_field = UUR[:, :-1]
        temperature_field = UUR[:, -1]

        self._point_data.append({'displacement_field': disp_field,
                                 'temperature_field': temperature_field})
    def get_data(self):
        return self._point_data

    def seed_none(self):
        self._deriv_mode = GlobalDerivType.DNONE

    def seed_U(self):
        self._deriv_mode = GlobalDerivType.DU














