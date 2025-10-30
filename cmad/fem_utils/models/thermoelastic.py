import numpy as np
import jax.numpy as jnp

from cmad.fem_utils.global_residuals.global_residual_thermomech import Global_residual_thermomech
from cmad.fem_utils.utils.fem_utils import (initialize_equation,
                                      compute_shape_jacobian,
                                      interpolate_vector_3D,
                                      interpolate_scalar,
                                      calc_element_traction_vector_3D)

from functools import partial
import jax


class Thermoelastic(Global_residual_thermomech):
    def __init__(self, problem):
        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn, ndim = problem.get_mesh_properties()

        disp_node, disp_val, pres_surf_traction, surf_traction_vector \
            = problem.get_boundary_conditions()

        pres_surf_flux = problem.get_convection_boundary_conditions()

        init_temp = problem.get_initial_temp()

        quad_rule_3D, shape_func_3D = problem.get_volume_basis_functions()
        quad_rule_2D, shape_func_2D = problem.get_surface_basis_functions()

        eq_num, num_free_dof, num_pres_dof = initialize_equation(num_nodes, dof_node, disp_node)

        num_steps, dt = problem.num_steps()

        elem_points = nodal_coords[volume_conn, :]
        pres_surf_traction_points = nodal_coords[pres_surf_traction]
        pres_surf_flux_points = nodal_coords[pres_surf_flux]

        print('Number of elements: ', num_elem)
        print('Number of free DOFS: ', num_free_dof)

        # E_0, nu, k, h, c_0, rho_0, alpha_0
        params = np.array([200e9, 0.265, 16.2, 30., 490., 7930., 16.e-6])

        global_residual = partial(self._global_residual_full,
                                  num_nodes_elem=num_nodes_elem,
                                  ndim=ndim,
                                  gauss_weights_3D=quad_rule_3D.wgauss,
                                  shape_3D=shape_func_3D.values,
                                  dshape_3D=shape_func_3D.gradients,
                                  dt=dt)

        elem_surf_traction = partial(calc_element_traction_vector_3D,
                                     num_nodes_surf=num_nodes_surf,
                                     ndim=ndim,
                                     gauss_weights_2D=quad_rule_2D.wgauss,
                                     shape_2D=shape_func_2D.values,
                                     dshape_2D=shape_func_2D.gradients)

        elem_surf_heat_flux = partial(self._calc_element_heat_flux_3D,
                                      num_nodes_surf=num_nodes_surf,
                                      gauss_weights_2D=quad_rule_2D.wgauss,
                                      shape_2D=shape_func_2D.values,
                                      dshape_2D=shape_func_2D.gradients)

        super().__init__(global_residual, elem_surf_traction, elem_surf_heat_flux,
                         volume_conn, elem_points, eq_num, params, num_nodes_elem,
                         num_nodes_surf, dof_node, num_free_dof, disp_node, disp_val,
                         num_pres_dof, pres_surf_traction_points, pres_surf_traction,
                         surf_traction_vector, pres_surf_flux_points, pres_surf_flux,
                         init_temp, dt)


    def _global_residual_full(
            self, u, u_prev, v_prev, a_prev, params, elem_points, elem_init_temp,
            num_nodes_elem, ndim, gauss_weights_3D, shape_3D,
            dshape_3D, dt):

        momentum_residual = self._compute_momentum_residual(u, u_prev, v_prev, a_prev, params,
                                                            elem_points, elem_init_temp, num_nodes_elem,
                                                            ndim, gauss_weights_3D, shape_3D,
                                                            dshape_3D, dt)

        thermal_residual = self._compute_thermal_residual(u, u_prev, v_prev, a_prev, params,
                                                          elem_points, num_nodes_elem, ndim,
                                                          gauss_weights_3D, shape_3D,
                                                          dshape_3D, dt)

        return jnp.concatenate((momentum_residual, thermal_residual))

    def _compute_momentum_residual(
            self, u, u_prev, v_prev, a_prev, params, elem_points, elem_init_temp,
            num_nodes_elem, ndim, gauss_weights_3D, shape_3D, dshape_3D, dt):

        SD_vec = jnp.zeros((num_nodes_elem, ndim))
        inertial_vec = jnp.zeros((num_nodes_elem, ndim))

        rho_0 = params[5]

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        elem_theta_prev = u_prev[num_nodes_elem * ndim:]

        # compute a_{n+1}
        a = (elem_disp - elem_disp_prev - v_prev * dt) * 4 / dt ** 2 - a_prev

        for gaussPt3D in range(len(gauss_weights_3D)):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)

            a_q, grad_a_q = interpolate_vector_3D(a, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)

            theta_q = interpolate_scalar(elem_theta, shape_3D_q)

            # compute initial stress
            theta_init_q = interpolate_scalar(elem_init_temp, shape_3D_q)
            init_grad_u_q = jnp.zeros((ndim, ndim))
            init_stress_q = self._compute_stress(init_grad_u_q, theta_init_q, params)

            # Inertial term
            inertial_vec += w_q * rho_0 * \
                jnp.column_stack([shape_3D_q, shape_3D_q, shape_3D_q]) * \
                    a_q * dv_q

            # stress-divergence term
            stress = self._compute_stress(grad_u_q, theta_q, params)
            SD_vec +=  w_q * gradphiXYZ_q.T @ (stress - init_stress_q).T * dv_q

        residual_full = SD_vec + inertial_vec
        return residual_full.reshape(-1, order='F')

    def _compute_thermal_residual(
            self, u, u_prev, v_prev, a_prev, params, elem_points,
            num_nodes_elem, ndim, gauss_weights_3D, shape_3D,
            dshape_3D, dt):

        residual = jnp.zeros(num_nodes_elem)

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        elem_theta_prev = u_prev[num_nodes_elem * ndim:]

        k = params[2]

        # compute a_{n+1} and v_{n+1}
        a = (elem_disp - elem_disp_prev - v_prev * dt) * 4 / dt ** 2 - a_prev
        v = v_prev + 1 / 2 * (a_prev + a) * dt

        for gaussPt3D in range(len(gauss_weights_3D)):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)

            v_q, grad_v_q = interpolate_vector_3D(v, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)

            theta_q = interpolate_scalar(elem_theta, shape_3D_q)
            theta_prev_q = interpolate_scalar(elem_theta_prev, shape_3D_q)

            # theta-dot term
            theta_dot_q = 1 / dt * (theta_q - theta_prev_q)
            residual += w_q * shape_3D_q * \
                self.compute_heat_capacity(grad_u_q, theta_q, params) * \
                    theta_dot_q * dv_q

            # heat flux term
            grad_theta_q = gradphiXYZ_q @ elem_theta
            q0 = -k * grad_theta_q
            residual -= w_q * gradphiXYZ_q.T @ q0 * dv_q

            # stress power term
            M_q = self._compute_stress_temp_modulus(grad_u_q, theta_q, params)
            residual -= w_q * shape_3D_q * theta_q * jnp.sum(M_q * grad_v_q) * dv_q

        return residual

    @staticmethod
    def _calc_element_heat_flux_3D(
            surf_theta, surf_points, params, num_nodes_surf,
            gauss_weights_2D, shape_2D, dshape_2D):

        QEL = jnp.zeros(num_nodes_surf)
        E_0, nu, k, h, c_0, rho_0, alpha = params
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

    @staticmethod
    def _f(theta):
        a = -5.5
        b = 1.0
        theta_0_m = 475.

        f = b * (theta / theta_0_m) ** a + b * (a - 1) + \
              (1 - a * b) * theta / theta_0_m

        return 1.

    def _compute_stress_temp_modulus(self, grad_u, theta, params):
        df = jax.jacfwd(self._f)

        E_0, nu, k, h, c_0, rho_0, alpha = params
        theta_0 = 300.

        K_0 = E_0 / (3 * (1 - 2 * nu))
        mu_0 = E_0 / (2 * (1 + nu))
        lam_0 = E_0 * nu / ((1 + nu) * (1 - 2 * nu))

        eps = 1 / 2 * (grad_u + grad_u.T)

        sigma_0 = 2 * mu_0 * eps + lam_0 * jnp.trace(eps) * jnp.eye(3)
        M = df(theta) * sigma_0 - (df(theta) * (theta - theta_0) + self._f(theta)) * \
              K_0 * alpha * 1 / (1 + jnp.trace(eps)) * jnp.eye(3)

        return M

    def compute_heat_capacity(self, grad_u, theta, params):
        df = jax.jacfwd(self._f)
        d2f = jax.jacfwd(jax.jacfwd(self._f))
        E_0, nu, k, h, c_0, rho_0, alpha = params
        theta_0 = 300.

        K_0 = E_0 / (3 * (1 - 2 * nu))
        mu_0 = E_0 / (2 * (1 + nu))
        lam_0 = E_0 * nu / ((1 + nu) * (1 - 2 * nu))

        eps = 1 / 2 * (grad_u + grad_u.T)

        sigma_0 = 2 * mu_0 * eps + lam_0 * jnp.trace(eps) * jnp.eye(3)

        c = -theta * d2f(theta) * 1 / 2 * jnp.sum(eps * sigma_0) + \
            (theta * d2f(theta) * (theta - theta_0) + 2 * theta * df(theta)) * \
                K_0 * alpha * jnp.log(1 + jnp.trace(eps)) + c_0 * rho_0

        return c

    def _compute_stress(self, grad_u, theta, params):
        # computes the stress tensor
        E_0, nu, k, h, c_0, rho_0, alpha = params
        theta_0 = 300.

        K_0 = E_0 / (3 * (1 - 2 * nu))
        mu_0 = E_0 / (2 * (1 + nu))
        lam_0 = E_0 * nu / ((1 + nu) * (1 - 2 * nu))

        eps = 1 / 2 * (grad_u + grad_u.T)
        sigma_0 = 2 * mu_0 * eps + lam_0 * jnp.trace(eps) * jnp.eye(3)
        sigma_theta = K_0 * alpha * (theta - theta_0) / (1 + jnp.trace(eps)) * jnp.eye(3)
        sigma = self._f(theta) * (sigma_0 - sigma_theta)

        return sigma
