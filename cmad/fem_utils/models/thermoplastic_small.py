import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map

from cmad.fem_utils.global_residuals.global_residual_thermoplastic import Global_residual_thermoplastic
from cmad.fem_utils.utils.fem_utils import (initialize_equation,
                                      compute_shape_jacobian,
                                      interpolate_vector_3D,
                                      interpolate_scalar,
                                      calc_element_traction_vector_3D)
from cmad.parameters.parameters import Parameters
from cmad.models.deformation_types import DefType
from cmad.models.small_elastic_plastic import SmallElasticPlastic
from cmad.models.var_types import get_sym_tensor_from_vector

def create_J2_parameters():
    # E = 200.e3
    # nu = 0.249
    # Y = 349.
    # K = 1.e5
    # S = 1.23e3
    # D = 0.55

    E = 69.e3
    nu = 0.31
    Y = 128.
    K = 1.e5
    S = 230.42
    D = 11.37

    # thermal parameters
    k = 16.2
    h = 30. 
    c_0 = 490. 
    rho_0 = 7930. 
    alpha_0 = 16.e-6

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    linear_params = {"K": K}
    hardening_params = {"voce": voce_params}
    thermal_params = {"k": k, "h": h, "c_0": c_0, "rho_0": rho_0, "alpha_0": alpha_0}

    Y_log_scale = np.array([48.])
    K_log_scale = np.array([100.])
    S_log_scale = np.array([106.])
    D_log_scale = np.array([25.])

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "thermal": thermal_params,
        "plastic": {
            "effective stress": J2_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "hardening": hardening_params}}}
    
    J2_active_flags = J2_values.copy()
    J2_active_flags = tree_map(lambda a: False, J2_active_flags)
    J2_active_flags["plastic"]["flow stress"] = tree_map(
        lambda x: True, J2_active_flags["plastic"]["flow stress"])
    
    J2_transforms = J2_values.copy()
    J2_transforms = tree_map(lambda a: None, J2_transforms)
    J2_flow_stress_transforms = J2_transforms["plastic"]["flow stress"]
    J2_flow_stress_transforms["initial yield"]["Y"] = Y_log_scale
    # J2_flow_stress_transforms["hardening"]["linear"]["K"] = K_log_scale
    J2_flow_stress_transforms["hardening"]["voce"]["S"] = S_log_scale
    J2_flow_stress_transforms["hardening"]["voce"]["D"] = D_log_scale

    J2_parameters = \
        Parameters(J2_values, J2_active_flags, J2_transforms)
    return J2_parameters

class Thermoplastic_small(Global_residual_thermoplastic):
    def __init__(self, problem):
        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn, ndim = problem.get_mesh_properties()

        disp_node, disp_val, pres_surf_traction, surf_traction_vector \
            = problem.get_boundary_conditions()

        pres_surf_flux = problem.get_convection_boundary_conditions()

        init_temp = problem.get_initial_temp()

        quad_rule_3D, shape_func_3D = problem.get_volume_basis_functions()
        quad_rule_2D, shape_func_2D = problem.get_surface_basis_functions()
        num_quad_pts = len(quad_rule_3D.wgauss)

        eq_num, num_free_dof, num_pres_dof = initialize_equation(num_nodes, dof_node, disp_node)

        num_steps, dt = problem.num_steps()

        elem_points = nodal_coords[volume_conn, :]
        pres_surf_traction_points = nodal_coords[pres_surf_traction]
        pres_surf_flux_points = nodal_coords[pres_surf_flux]

        print('Number of elements: ', num_elem)
        print('Number of free DOFS: ', num_free_dof)

        params = create_J2_parameters()

        def_type = DefType.FULL_3D

        # material point model
        mat_point_model = SmallElasticPlastic(params, def_type=def_type)
        self._local_residual_material_point = mat_point_model.get_local_residual()
        self._cauchy_fun = mat_point_model.get_cauchy()
        num_local_resid_dofs = mat_point_model.num_dofs

        init_xi = np.zeros(num_local_resid_dofs * num_quad_pts)

        elem_local_resid = partial(self._elem_local_resid,
                                   num_nodes_elem=num_nodes_elem,
                                   ndim=ndim,
                                   gauss_weights_3D=quad_rule_3D.wgauss,
                                   shape_3D=shape_func_3D.values,
                                   dshape_3D=shape_func_3D.gradients,
                                   num_local_resid_dofs=num_local_resid_dofs)

        elem_global_resid = partial(self._elem_global_resid_full,
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

        super().__init__(elem_global_resid, elem_local_resid, elem_surf_traction, elem_surf_heat_flux,
                         volume_conn, elem_points, eq_num, params, num_nodes_elem, num_nodes_surf, dof_node, 
                         num_quad_pts, num_free_dof, disp_node, disp_val, init_xi, num_pres_dof, 
                         num_elem, pres_surf_traction_points, pres_surf_traction, surf_traction_vector, 
                         pres_surf_flux_points, pres_surf_flux, init_temp, dt)

    def _elem_local_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D, num_local_resid_dofs):

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]

        num_quad_pts = len(gauss_weights_3D)

        elem_local_residual = jnp.zeros((num_quad_pts, num_local_resid_dofs))

        ep_dofs = 6
        elem_ep = xi[:num_quad_pts * ep_dofs]
        elem_ep_prev = xi_prev[:num_quad_pts * ep_dofs]
        elem_alpha = xi[-num_quad_pts:]
        elem_alpha_prev = xi_prev[-num_quad_pts:]

        for gaussPt3D in range(num_quad_pts):

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            u_prev_q, grad_u_prev_q = interpolate_vector_3D(elem_disp_prev, shape_3D_q, gradphiXYZ_q, num_nodes_elem)

            F_q = [grad_u_q + jnp.eye(ndim)]
            F_prev_q = [grad_u_prev_q + jnp.eye(ndim)]

            ep_q = elem_ep[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            ep_prev_q = elem_ep_prev[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            alpha_q = jnp.array([elem_alpha[gaussPt3D - num_quad_pts]])
            alpha_prev_q = jnp.array([elem_alpha_prev[gaussPt3D - num_quad_pts]])

            xi_recast = [ep_q, alpha_q]
            xi_prev_recast = [ep_prev_q, alpha_prev_q]

            elem_residual_q = self._local_residual_material_point(xi_recast, xi_prev_recast, params, F_q, F_prev_q)

            elem_local_residual = elem_local_residual.at[gaussPt3D, :].set(elem_residual_q)

        return elem_local_residual.reshape(-1)

    def _elem_global_resid_full(
            self, u, u_prev, params, xi, xi_prev, elem_points,
            num_nodes_elem, ndim, gauss_weights_3D, shape_3D, dshape_3D, dt):

        momentum_residual = self._elem_momentum_resid(u, u_prev, params, xi, xi_prev, elem_points, 
                                                      num_nodes_elem, ndim, gauss_weights_3D, shape_3D, dshape_3D)

        thermal_residual = self._elem_thermal_residual(u, u_prev, params, xi, xi_prev, elem_points,
                                                       num_nodes_elem, ndim, gauss_weights_3D, shape_3D, dshape_3D, dt)
        
        DB_residual = self._elem_DB_residual(u, params, elem_points, num_nodes_elem,
                                             ndim, gauss_weights_3D, shape_3D, dshape_3D)

        return jnp.concatenate((momentum_residual, thermal_residual, DB_residual))
    
    def _elem_momentum_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem, 
            ndim, gauss_weights_3D, shape_3D, dshape_3D):

        # reference temperature
        theta_0 = 300.

        # extract element displacement and pressure
        elem_disp = u[0:num_nodes_elem * ndim]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]
        p = u[-num_nodes_elem:]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        K_param = E / (3 * (1 - 2 * nu))

        rho_0 = params['thermal']['rho_0']
        alpha_0 = params['thermal']['alpha_0']

        # stress divergence residual
        S_D_vec = jnp.zeros((num_nodes_elem, ndim))

        num_quad_pts = len(gauss_weights_3D)

        ep_dofs = 6
        elem_ep = xi[:num_quad_pts * ep_dofs]
        elem_ep_prev = xi_prev[:num_quad_pts * ep_dofs]
        elem_alpha = xi[-num_quad_pts:]
        elem_alpha_prev = xi_prev[-num_quad_pts:]

        for gaussPt3D in range(num_quad_pts):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            u_prev_q, grad_u_prev_q = interpolate_vector_3D(elem_disp_prev, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            theta_q = interpolate_scalar(elem_theta, shape_3D_q)
            p_q = interpolate_scalar(p, shape_3D_q)

            F_q = [grad_u_q + jnp.eye(ndim)]
            F_prev_q = [grad_u_prev_q + jnp.eye(ndim)]

            ep_q = elem_ep[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            ep_prev_q = elem_ep_prev[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            alpha_q = jnp.array([elem_alpha[gaussPt3D - num_quad_pts]])
            alpha_prev_q = jnp.array([elem_alpha_prev[gaussPt3D - num_quad_pts]])

            xi_recast = [ep_q, alpha_q]
            xi_prev_recast = [ep_prev_q, alpha_prev_q]

            elastic_stress = self._cauchy_fun(xi_recast, xi_prev_recast, params, F_q, F_prev_q)
            dev_elastic_stress = elastic_stress - 1 / ndim * jnp.trace(elastic_stress) * jnp.eye(ndim) 
            stress = dev_elastic_stress + p_q * jnp.eye(ndim)

            #stress divergence term
            S_D_vec += w_q * gradphiXYZ_q.T @ stress.T * dv_q

        return (S_D_vec).reshape(-1, order='F')
    
    def _elem_DB_residual(
            self, u, params, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D):
        
        elem_disp = u[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]
        p = u[-num_nodes_elem:]

        alpha_0 = params['thermal']['alpha_0']
        theta_0 = 300.

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        G_param = E / (2 * (1 + nu))
        K_param = E / (3 * (1 - 2 * nu))
        alpha = 100.

        # incompressibility residual
        H = 0
        G = jnp.zeros(num_nodes_elem)
        DB_residual = jnp.zeros(num_nodes_elem)

        num_quad_pts = len(gauss_weights_3D)

        for gaussPt3D in range(num_quad_pts):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            p_q = interpolate_scalar(p, shape_3D_q)
            theta_q = interpolate_scalar(elem_theta, shape_3D_q)

            DB_residual += w_q *  shape_3D_q * (jnp.trace(grad_u_q) \
                - alpha_0 * (theta_q - theta_0) / (1 + jnp.trace(grad_u_q))) * dv_q

            # DB contibution (projection onto constant polynomial space)
            H += w_q * 1.0 * dv_q
            G += w_q * shape_3D_q * dv_q

            # (N.T)(alpha / G)(N)(p)
            DB_residual -= (alpha / G_param + 1 / K_param) * w_q \
                * shape_3D_q * p_q * dv_q

        # alpha / G * (G.T)(H^-1)(G)(p)
        DB_residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)

        return DB_residual

    def _elem_thermal_residual(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem, 
            ndim, gauss_weights_3D, shape_3D, dshape_3D, dt):

        thermal_resid = jnp.zeros(num_nodes_elem)

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        elem_theta_prev = u_prev[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        K_param = E / (3 * (1 - 2 * nu))

        rho_0 = params['thermal']['rho_0']
        alpha_0 = params['thermal']['alpha_0']
        k = params['thermal']['k']
        c_0 = params['thermal']['c_0']

        # Taylor Quinney
        beta = 0.9

        num_quad_pts = len(gauss_weights_3D)
        ep_dofs = 6
        elem_ep = xi[:num_quad_pts * ep_dofs]
        elem_ep_prev = xi_prev[:num_quad_pts * ep_dofs]
        elem_alpha = xi[-num_quad_pts:]
        elem_alpha_prev = xi_prev[-num_quad_pts:]

        for gaussPt3D in range(num_quad_pts):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q,
                                                  gradphiXYZ_q, num_nodes_elem)
            u_prev_q, grad_u_prev_q = interpolate_vector_3D (elem_disp_prev, shape_3D_q, 
                                                             gradphiXYZ_q, num_nodes_elem)

            theta_q = interpolate_scalar(elem_theta, shape_3D_q)
            theta_prev_q = interpolate_scalar(elem_theta_prev, shape_3D_q)

            ep_q = elem_ep[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            ep_prev_q = elem_ep_prev[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            alpha_q = jnp.array([elem_alpha[gaussPt3D - num_quad_pts]])
            alpha_prev_q = jnp.array([elem_alpha_prev[gaussPt3D - num_quad_pts]])

            F_q = [grad_u_q + jnp.eye(ndim)]
            F_prev_q = [grad_u_prev_q + jnp.eye(ndim)]
            xi_recast = [ep_q, alpha_q]
            xi_prev_recast = [ep_prev_q, alpha_prev_q]

            # theta-dot term
            theta_dot_q = 1 / dt * (theta_q - theta_prev_q)
            thermal_resid += w_q * shape_3D_q * c_0 * rho_0 * theta_dot_q * dv_q

            # heat flux term
            grad_theta_q = gradphiXYZ_q @ elem_theta
            q0 = -k * grad_theta_q
            thermal_resid -= w_q * gradphiXYZ_q.T @ q0 * dv_q

            # plastic dissipation term
            ep_dot_q = get_sym_tensor_from_vector((ep_q - ep_prev_q) / dt, ndim)
            elastic_stress = self._cauchy_fun(xi_recast, xi_prev_recast, params, F_q, F_prev_q)
            thermal_resid -= w_q * shape_3D_q * beta * jnp.sum(elastic_stress * ep_dot_q) * 1.e6 * dv_q

            # stress power term
            e_q = 1 / 2 * (grad_u_q + grad_u_q.T)
            e_prev_q = 1 / 2 * (grad_u_prev_q + grad_u_prev_q.T)
            e_dot_q = (e_q - e_prev_q) / dt
            ee_dot_q = e_dot_q - ep_dot_q
            M_q = -K_param * alpha_0 * 1 / (1 + jnp.trace(grad_u_q)) * jnp.eye(ndim)
            thermal_resid -= w_q * shape_3D_q * theta_q * jnp.sum(M_q * ee_dot_q) * 1.e6 * dv_q

        return thermal_resid * 1.e-4

    @staticmethod
    def _calc_element_heat_flux_3D(
            surf_theta, surf_points, params, num_nodes_surf,
            gauss_weights_2D, shape_2D, dshape_2D):

        QEL = jnp.zeros(num_nodes_surf)
        h = params['thermal']['h']
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
