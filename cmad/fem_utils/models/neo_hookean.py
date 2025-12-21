import numpy as np
import jax.numpy as jnp

from cmad.fem_utils.global_residuals.global_residual import Global_residual
from cmad.fem_utils.utils.fem_utils import (initialize_equation,
                                      compute_shape_jacobian,
                                      interpolate_vector_3D,
                                      interpolate_scalar,
                                      calc_element_traction_vector_3D)

from functools import partial

class Neo_hookean(Global_residual):
    def __init__(self, problem):
        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn, ndim = problem.get_mesh_properties()

        disp_node, disp_val, pres_surf_traction, surf_traction_vector \
            = problem.get_boundary_conditions()

        quad_rule_3D, shape_func_3D = problem.get_volume_basis_functions()
        quad_rule_2D, shape_func_2D = problem.get_surface_basis_functions()

        eq_num, num_free_dof, num_pres_dof = initialize_equation(num_nodes, dof_node, disp_node)

        elem_points = nodal_coords[volume_conn, :]
        pres_surf_traction_points = nodal_coords[pres_surf_traction]

        print('Number of elements: ', num_elem)
        print('Number of free DOFS: ', num_free_dof)

        # E, nu
        params = np.array([200e3, 0.3])

        is_mixed = problem.is_mixed()

        if (is_mixed):
            residual = partial(self._compute_residual_u_p,
                               num_nodes_elem=num_nodes_elem,
                               ndim=ndim,
                               gauss_weights_3D=quad_rule_3D.wgauss,
                               shape_3D=shape_func_3D.values,
                               dshape_3D=shape_func_3D.gradients)
        else:
            residual = partial(self._compute_residual_u,
                               num_nodes_elem=num_nodes_elem,
                               ndim=ndim,
                               gauss_weights_3D=quad_rule_3D.wgauss,
                               shape_3D=shape_func_3D.values,
                               dshape_3D=shape_func_3D.gradients)

        elem_surf_traction = partial(calc_element_traction_vector_3D,
                                     num_nodes_surf=num_nodes_surf,
                                     ndim=ndim,
                                     gauss_weights_2D=quad_rule_2D.wgauss,
                                     shape_2D=shape_func_2D.values,
                                     dshape_2D=shape_func_2D.gradients)

        super().__init__(residual, elem_surf_traction, volume_conn, elem_points, eq_num,
                         params, num_nodes_elem, dof_node, num_free_dof, disp_node, disp_val,
                         num_pres_dof, is_mixed, pres_surf_traction_points, pres_surf_traction,
                         surf_traction_vector)

    def _compute_residual_u_p(
            self, u, params, elem_points, num_nodes_elem, ndim,
            gauss_weights_3D, shape_3D, dshape_3D):

        # extract element displacement and pressure
        elem_disp = u[0:num_nodes_elem * ndim]
        p = u[num_nodes_elem * ndim:]

        E = params[0]
        nu = params[1]
        G_param = E / (2 * (1 + nu))
        alpha = 1.0

        # incompressibility residual
        H = 0
        G = jnp.zeros(num_nodes_elem)
        incomp_residual = jnp.zeros(num_nodes_elem)

        # stress divergence residual
        S_D_vec = jnp.zeros((num_nodes_elem, ndim))

        for gaussPt3D in range(len(gauss_weights_3D)):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)
            p_q = interpolate_scalar(p, shape_3D_q)

            # compute stress divergence residual
            stress = self._compute_neo_hookean_stress_p(grad_u_q, p_q, params)

            S_D_vec += w_q * gradphiXYZ_q.T @ stress.T * dv_q

            # compute incompressibility residual
            F = jnp.eye(3) + grad_u_q
            J = jnp.linalg.det(F)
            incomp_residual += w_q * shape_3D_q * (J - 1) * dv_q

            # DB contibution (projection onto constant polynomial space)
            H += w_q * 1.0 * dv_q
            G += w_q * shape_3D_q * dv_q

            # (N.T)(alpha / G)(N)(p)
            incomp_residual -= (alpha / G_param) * w_q \
                * shape_3D_q * jnp.dot(shape_3D_q, p) * dv_q

        # alpha / G * (G.T)(H^-1)(G)(p)
        incomp_residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)

        return jnp.concatenate((S_D_vec.reshape(-1, order='F'), incomp_residual))

    def _compute_residual_u(
            self, u, params, elem_points, num_nodes_elem, ndim,
            gauss_weights_3D, shape_3D, dshape_3D):

        SD_vec = jnp.zeros((num_nodes_elem, ndim))

        for gaussPt3D in range(len(gauss_weights_3D)):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(u, shape_3D_q, gradphiXYZ_q, num_nodes_elem)

            stress = self._compute_neo_hookean_stress(grad_u_q, params)

            SD_vec +=  w_q * gradphiXYZ_q.T @ stress.T * dv_q

        return SD_vec.reshape(-1, order='F')

    @staticmethod
    def _compute_neo_hookean_stress_p(grad_u, p, params):

        # computes the first Piola-Kirchoff stress tensor (set J = 1)
        E = params[0]
        nu = params[1]

        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        F = jnp.eye(3) + grad_u
        F_inv_T = jnp.linalg.inv(F).T
        T = mu * F @ F.T - 1 / 3 * jnp.trace(mu * F @ F.T) * jnp.eye(3) + p * jnp.eye(3)
        P = T @ F_inv_T

        return P

    @staticmethod
    def _compute_neo_hookean_stress(grad_u, params):

        # computes the first Piola-Kirchoff stress tensor
        E = params[0]
        nu = params[1]

        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        F = jnp.eye(3) + grad_u
        F_inv_T = jnp.linalg.inv(F).T
        J = jnp.linalg.det(F)
        P = mu * (F - F_inv_T) + lam * J * (J - 1) * F_inv_T

        return P
