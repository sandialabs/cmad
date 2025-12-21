import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map

from cmad.fem_utils.global_residuals.global_residual_thermoplastic import Global_residual_thermoplastic
from cmad.fem_utils.utils.fem_utils import (initialize_equation,
                                      compute_shape_jacobian,
                                      interpolate_vector_3D,
                                      interpolate_scalar,
                                      calc_element_traction_vector_3D)
from cmad.models.var_types import (get_sym_tensor_from_vector,
                                   get_vector_from_sym_tensor)
from cmad.models.elastic_stress import two_mu_scale_factor
from cmad.parameters.parameters import Parameters
from cmad.models.deformation_types import DefType
from jax.lax import cond
from cmad.fem_utils.interpolants import interpolants
from cmad.fem_utils.quadrature import quadrature_rule


def create_J2_parameters():

    E = 206.9e3 # MPa
    nu = 0.29
    Y_0 = 450.0 # MPa
    K_0 = 129.24 # MPa
    S_0 = 265.0 # MPa
    D = 16.92

    k = 45.0 # W/(m K)
    h = 30. # W/(m^2 K)
    c_0 = 1328.9 # J/(kg K)
    rho_0 = 2700.0 # kg/m^3
    alpha_0 = 1e-05 # 1/K
    theta_0 = 300. # K

    w = 0.002 # 1/K

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y_0": Y_0}
    voce_params = {"S_0": S_0, "D": D}
    linear_params = {"K_0": K_0}
    hardening_params = {"voce": voce_params, "linear": linear_params}
    thermal_params = {"k": k, "h": h, "c_0": c_0, "rho_0": rho_0, "alpha_0": alpha_0, "theta_0": theta_0}
    softening_params = {"w": w}

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
        "thermal": thermal_params,
        "plastic": {
            "effective stress": J2_effective_stress_params,
            "flow stress": {
                "initial yield": initial_yield_params,
                "softening": softening_params,
                "hardening": hardening_params}}}

    J2_parameters = \
        Parameters(J2_values)

    return J2_parameters

def evaluate_yield_function(s, alpha, theta, params):
    plastic_params = params["plastic"]
    thermal_params = params["thermal"]
    Y_0 = plastic_params["flow stress"]["initial yield"]["Y_0"]
    hardening_params = plastic_params["flow stress"]["hardening"]
    softening_params = plastic_params["flow stress"]["softening"]
    voce_params = hardening_params["voce"]
    linear_params = hardening_params["linear"]
    S_0 = voce_params["S_0"]
    D = voce_params["D"]
    K_0 = linear_params["K_0"]
    w = softening_params["w"]
    theta_0 = thermal_params["theta_0"]

    # temperature dependent parameters
    Y = Y_0 * (1 - w * (theta - theta_0))
    K = K_0 * (1 - w * (theta - theta_0))
    S = S_0 * (1 - w * (theta - theta_0))

    s_norm = jnp.sqrt(jnp.sum(s * s))
    flow_stress = jnp.sqrt(2 / 3) * (Y + K * alpha + S * (1 - jnp.exp(-D * alpha)))
    return (s_norm - flow_stress) / two_mu_scale_factor(params)

def evaluate_current_yield_stress(alpha, theta, params):
    plastic_params = params["plastic"]
    thermal_params = params["thermal"]
    Y_0 = plastic_params["flow stress"]["initial yield"]["Y_0"]
    hardening_params = plastic_params["flow stress"]["hardening"]
    softening_params = plastic_params["flow stress"]["softening"]
    voce_params = hardening_params["voce"]
    linear_params = hardening_params["linear"]
    S_0 = voce_params["S_0"]
    D = voce_params["D"]
    K_0 = linear_params["K_0"]
    w = softening_params["w"]
    theta_0 = thermal_params["theta_0"]

    # temperature dependent parameters
    Y = Y_0 * (1 - w * (theta - theta_0))
    K = K_0 * (1 - w * (theta - theta_0))
    S = S_0 * (1 - w * (theta - theta_0))

    return Y + K * alpha + S * (1 - jnp.exp(-D * alpha))

def compute_yield_normal(s):
    s_norm = jnp.sqrt(jnp.sum(s * s))
    return s / s_norm

class Thermoplastic_finite(Global_residual_thermoplastic):
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

        num_local_resid_dofs = 8
        init_xi = np.zeros(num_local_resid_dofs * num_quad_pts)
        init_xi[-2 * num_quad_pts:-num_quad_pts] = np.ones(num_quad_pts)


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

    def _elem_local_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D, num_local_resid_dofs):

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]

        num_quad_pts = len(gauss_weights_3D)

        elem_local_residual = jnp.zeros((num_quad_pts, num_local_resid_dofs))

        zeta_dofs = 6
        elem_Gamma_bar = xi[:num_quad_pts * zeta_dofs].reshape(num_quad_pts, zeta_dofs)
        elem_Gamma_bar_prev = xi_prev[:num_quad_pts * zeta_dofs].reshape(num_quad_pts, zeta_dofs)
        elem_I_bar = xi[-2 * num_quad_pts:-num_quad_pts]
        elem_I_bar_prev = xi_prev[-2 * num_quad_pts:-num_quad_pts]
        elem_alpha = xi[-num_quad_pts:]
        elem_alpha_prev = xi_prev[-num_quad_pts:]

        for gaussPt3D in range(num_quad_pts):

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)

            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            u_prev_q, grad_u_prev_q = interpolate_vector_3D(elem_disp_prev, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
            theta_q = interpolate_scalar(elem_theta, shape_3D_q)

            F_q = grad_u_q + jnp.eye(ndim)
            F_prev_q = grad_u_prev_q + jnp.eye(ndim)

            Gamma_bar_q = elem_Gamma_bar[gaussPt3D]
            Gamma_bar_prev_q = elem_Gamma_bar_prev[gaussPt3D]
            I_bar_q = elem_I_bar[gaussPt3D]
            I_bar_prev_q = elem_I_bar_prev[gaussPt3D]
            alpha_q = elem_alpha[gaussPt3D]
            alpha_prev_q = elem_alpha_prev[gaussPt3D]

            xi_recast = [Gamma_bar_q, I_bar_q, alpha_q]
            xi_prev_recast = [Gamma_bar_prev_q, I_bar_prev_q, alpha_prev_q]

            elem_residual_q = self._local_resid_material_pt(theta_q, F_q, F_prev_q, params, xi_recast, xi_prev_recast)

            elem_local_residual = elem_local_residual.at[gaussPt3D, :].set(elem_residual_q)

        return elem_local_residual.reshape(-1)

    def _elem_momentum_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D):

        # extract element displacement and pressure
        elem_disp = u[0:num_nodes_elem * ndim]
        p = u[-num_nodes_elem:]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        mu = E / (2 * (1 + nu))

        # stress divergence residual
        S_D_vec = jnp.zeros((num_nodes_elem, ndim))

        num_quad_pts = len(gauss_weights_3D)

        zeta_dofs = 6
        elem_Gamma_bar = xi[:num_quad_pts * zeta_dofs].reshape(num_quad_pts, zeta_dofs)

        for gaussPt3D in range(num_quad_pts):
            w_q = gauss_weights_3D[gaussPt3D]

            dshape_3D_q = dshape_3D[gaussPt3D, :, :]
            shape_3D_q = shape_3D[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
    
            p_q = interpolate_scalar(p, shape_3D_q)
            grad_p_q = gradphiXYZ_q @ p

            F_q = grad_u_q + jnp.eye(ndim)
            F_inv_q = jnp.linalg.inv(F_q)
            J_q = jnp.linalg.det(F_q)

            elem_Gamma_bar_q = get_sym_tensor_from_vector(elem_Gamma_bar[gaussPt3D], 3)

            P = mu * elem_Gamma_bar_q @ F_inv_q.T + J_q * p_q * F_inv_q.T
            S_D_vec += w_q * gradphiXYZ_q.T @ P.T * dv_q

        return S_D_vec.reshape(-1, order='F') 
    
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

            F_q = grad_u_q + jnp.eye(ndim)
            F_inv_q = jnp.linalg.inv(F_q)
            J_q = jnp.linalg.det(F_q)

            F_prev_q = grad_u_prev_q + jnp.eye(ndim)
            J_prev_q = jnp.linalg.det(F_prev_q)

            # theta-dot term
            theta_dot_q = 1 / dt * (theta_q - theta_prev_q)
            thermal_resid += w_q * shape_3D_q * c_0 * rho_0 * theta_dot_q * dv_q

            # heat flux term
            grad_theta_q = gradphiXYZ_q @ elem_theta
            q0 = -J_q * k * F_inv_q @ F_inv_q.T @ grad_theta_q
            thermal_resid -= w_q * gradphiXYZ_q.T @ q0 * dv_q

            # mechanical dissipation
            alpha_q = elem_alpha[gaussPt3D]
            alpha_prev_q = elem_alpha_prev[gaussPt3D]
            y_q = evaluate_current_yield_stress(alpha_q, theta_q, params)
            D_mech = beta * (alpha_q - alpha_prev_q) / dt * y_q * 1.e6
            thermal_resid -= w_q * shape_3D_q * D_mech * dv_q

            # structural heating
            eta_q = - K_param / 2 * alpha_0 * ((J_q ** 2 - 1) / J_q)
            eta_prev_q = - K_param / 2 * alpha_0 * ((J_prev_q ** 2 - 1) / J_prev_q)
            H_q = - theta_q * (eta_q - eta_prev_q) / dt * 1.e6
            thermal_resid += w_q * shape_3D_q * H_q * dv_q
        
        return thermal_resid * 1.e-4
    
    def _elem_DB_residual(
            self, u, params, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D):
        
        elem_disp = u[0:num_nodes_elem * ndim]
        elem_theta = u[num_nodes_elem * ndim:num_nodes_elem * (ndim + 1)]
        p = u[-num_nodes_elem:]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        G_param = E / (2 * (1 + nu))
        K_param = E / (3 * (1 - 2 * nu))
        alpha = 1000.

        alpha_0 = params['thermal']['alpha_0']
        theta_0 = params['thermal']['theta_0']

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

            F_q = grad_u_q + jnp.eye(ndim)
            J_q = jnp.linalg.det(F_q)

            DB_residual += w_q * shape_3D_q * (1 / 2 * (J_q ** 2 - 1) / J_q \
                - 1 / 2 * alpha_0 * (theta_q - theta_0) * (J_q ** 2 + 1) / J_q ** 2) * dv_q

            # DB contibution (projection onto constant polynomial space)
            H += w_q * 1.0 * dv_q
            G += w_q * shape_3D_q * dv_q

            # (N.T)(alpha / G)(N)(p)
            DB_residual -= (alpha / G_param + 1 / K_param) * w_q \
                * shape_3D_q * p_q * dv_q

        # alpha / G * (G.T)(H^-1)(G)(p)
        DB_residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)

        return DB_residual
    
    def _local_resid_material_pt(self, theta, F, F_prev, params, xi, xi_prev):
        # material parameters
        E = params['elastic']['E']
        nu = params['elastic']['nu']
        mu = E / (2 * (1 + nu))

        # extract local state variables
        Gamma_bar = get_sym_tensor_from_vector(xi[0], 3)
        I_bar = xi[1]
        alpha = xi[2]

        Gamma_bar_prev = get_sym_tensor_from_vector(xi_prev[0], 3)
        I_bar_prev = xi_prev[1]
        alpha_prev = xi_prev[2]

        # isochoric Deformation Gradients
        F_bar = 1 / jnp.cbrt(jnp.linalg.det(F)) * F
        F_bar_prev = 1 / jnp.cbrt(jnp.linalg.det(F_prev)) * F_prev
        F_bar_prev_inv = jnp.linalg.inv(F_bar_prev)

        b_bar_prev = Gamma_bar_prev + I_bar_prev * jnp.eye(3)

        # define trial variables
        b_bar_trial = F_bar @ F_bar_prev_inv @ b_bar_prev @ F_bar_prev_inv.T @ F_bar.T
        Gamma_bar_trial = b_bar_trial - 1 / 3 * jnp.trace(b_bar_trial) * jnp.eye(3)
        dev_tau_trial = mu * Gamma_bar_trial
        phi_trial = evaluate_yield_function(dev_tau_trial, alpha_prev, theta, params)

        return self._cond_residual(phi_trial, Gamma_bar, I_bar, alpha, b_bar_trial, 
                                   dev_tau_trial, alpha_prev, theta, params, 1.e-14)
    
    @staticmethod
    def _elastic_path(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params):
        I_bar_trial = 1 / 3 * jnp.trace(b_bar_trial)
        Gamma_bar_trial = b_bar_trial - I_bar_trial * jnp.eye(3)
        C_Gamma_bar_elastic = get_vector_from_sym_tensor(Gamma_bar - Gamma_bar_trial, 3) 
        C_I_bar_elastic = I_bar - I_bar_trial
        C_alpha_elastic = alpha - alpha_prev
        return jnp.r_[C_Gamma_bar_elastic, C_I_bar_elastic, C_alpha_elastic]
    
    @staticmethod
    def _plastic_path(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params):
        E = params['elastic']['E']
        nu = params['elastic']['nu']
        mu = E / (2 * (1 + nu))

        I_bar_trial = 1 / 3 * jnp.trace(b_bar_trial)
        Gamma_bar_trial = b_bar_trial - I_bar_trial * jnp.eye(3)

        n = compute_yield_normal(dev_tau_trial)
        C_Gamma_bar_plastic = get_vector_from_sym_tensor(Gamma_bar - Gamma_bar_trial + 2 * jnp.sqrt(3 / 2) \
                                                        * (alpha - alpha_prev) * I_bar_trial * n, 3)
        
        dev_tau = mu * (Gamma_bar_trial - 2 * jnp.sqrt(3 / 2) * (alpha - alpha_prev) * I_bar_trial * n)
        C_alpha_plastic = evaluate_yield_function(dev_tau, alpha, theta, params)

        C_I_bar_plastic = jnp.linalg.det(Gamma_bar + I_bar * jnp.eye(3)) - 1

        return jnp.r_[C_Gamma_bar_plastic, C_I_bar_plastic, C_alpha_plastic]

    def _cond_residual(
        self, phi, Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params, tol):

        def inner_cond_residual(
                Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params): 
            return cond(jnp.abs(phi) < tol, self._plastic_path, self._elastic_path, 
                        Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params)

        def outer_cond_residual(
                Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params): 
            return cond(phi > tol, self._plastic_path, inner_cond_residual, 
                        Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params)
        
        return outer_cond_residual(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, theta, params)

                           
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


















