import numpy as np
import jax.numpy as jnp

from cmad.fem_utils.global_residuals.global_residual_thermo import Global_residual_thermo
from cmad.fem_utils.utils.fem_utils import (initialize_equation_thermo,
                                      compute_shape_jacobian,
                                      interpolate_vector_3D,
                                      interpolate_scalar,
                                      calc_element_traction_vector_3D)

from functools import partial
import jax


class Thermo(Global_residual_thermo):
    def __init__(self, problem):
        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn, ndim = problem.get_mesh_properties()

        disp_node, disp_val, pres_surf_traction, surf_traction_vector \
            = problem.get_boundary_conditions()

        pres_surf_flux = problem.get_convection_boundary_conditions()

        init_temp = problem.get_initial_temp()

        quad_rule_3D, shape_func_3D = problem.get_volume_basis_functions()
        quad_rule_2D, shape_func_2D = problem.get_surface_basis_functions()

        eq_num, num_free_dof, num_pres_dof = initialize_equation_thermo(num_nodes, disp_node)

        num_steps, dt = problem.num_steps()

        elem_points = nodal_coords[volume_conn, :]
        pres_surf_flux_points = nodal_coords[pres_surf_flux]

        print('Number of elements: ', num_elem)
        print('Number of free DOFS: ', num_free_dof)

        # k, h, c_0, rho_0
        params = np.array([16.2, 30., 490., 7930.])

        global_residual = partial(self._global_residual_thermo,
                                  num_nodes_elem=num_nodes_elem,
                                  gauss_weights_3D=quad_rule_3D.wgauss,
                                  shape_3D=shape_func_3D.values,
                                  dshape_3D=shape_func_3D.gradients,
                                  dt=dt)

        elem_surf_heat_flux = partial(self._calc_element_heat_flux_3D,
                                      num_nodes_surf=num_nodes_surf,
                                      gauss_weights_2D=quad_rule_2D.wgauss,
                                      shape_2D=shape_func_2D.values,
                                      dshape_2D=shape_func_2D.gradients)

        super().__init__(global_residual, elem_surf_heat_flux,
                         volume_conn, elem_points, eq_num, params, num_nodes_elem,
                         num_nodes_surf, num_free_dof, disp_node, disp_val,
                         num_pres_dof, pres_surf_flux_points, pres_surf_flux,
                         init_temp, dt)

    def _global_residual_thermo(
            self, theta, theta_prev, params, elem_points,
            num_nodes_elem, gauss_weights_3D, shape_3D,
            dshape_3D, dt):

        k, h, c_0, rho_0 = params
        residual = jnp.zeros(num_nodes_elem)

        for gaussPt3D in range(len(gauss_weights_3D)):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            theta_q = interpolate_scalar(theta, shape_3D_q)
            theta_prev_q = interpolate_scalar(theta_prev, shape_3D_q)

            # theta-dot term
            theta_dot_q = 1 / dt * (theta_q - theta_prev_q)
            residual += w_q * shape_3D_q * rho_0 * c_0 * theta_dot_q * dv_q

            # heat flux term
            grad_theta_q = gradphiXYZ_q @ theta
            q0 = -k * grad_theta_q
            residual -= w_q * gradphiXYZ_q.T @ q0 * dv_q

        return residual

    @staticmethod
    def _calc_element_heat_flux_3D(
            surf_theta, surf_points, params, num_nodes_surf,
            gauss_weights_2D, shape_2D, dshape_2D):

        QEL = jnp.zeros(num_nodes_surf)
        h = params[1]
        theta_inf = 300.

        for gaussPt2D in range(len(gauss_weights_2D)):
            w_q = gauss_weights_2D[gaussPt2D]
            shape_tri_q = shape_2D[gaussPt2D, :]
            dshape_tri_q = dshape_2D[gaussPt2D, :, :]

            J_q = dshape_tri_q @ surf_points

            da_q = jnp.linalg.norm(jnp.cross(J_q[0, :], J_q[1, :]))

            theta_q = interpolate_scalar(surf_theta, shape_tri_q)

            q_bar = h * (theta_q - theta_inf)

            QEL += w_q * shape_tri_q * q_bar * da_q

        return QEL