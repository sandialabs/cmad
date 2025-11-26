import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map

from cmad.fem_utils.global_residuals.global_residual_plasticity import Global_residual_plasticity
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

    elastic_params = {"E": E, "nu": nu}
    J2_effective_stress_params = {"J2": 0.}
    initial_yield_params = {"Y": Y}
    voce_params = {"S": S, "D": D}
    linear_params = {"K": K}
    hardening_params = {"voce": voce_params}

    Y_log_scale = np.array([48.])
    K_log_scale = np.array([100.])
    S_log_scale = np.array([106.])
    D_log_scale = np.array([25.])

    J2_values = {
        "rotation matrix": np.eye(3),
        "elastic": elastic_params,
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

def evaluate_yield_function(s, alpha, params):
    plastic_params = params["plastic"]
    Y = plastic_params["flow stress"]["initial yield"]["Y"]
    hardening_params = plastic_params["flow stress"]["hardening"]
    voce_params = hardening_params["voce"]
    S = voce_params["S"]
    D = voce_params["D"]
    s_norm = jnp.sqrt(jnp.sum(s * s))
    flow_stress = jnp.sqrt(2 / 3) * (Y + S * (1 - jnp.exp(-D * alpha)))
    return (s_norm - flow_stress) / two_mu_scale_factor(params)

def compute_yield_normal(s):
    s_norm = jnp.sqrt(jnp.sum(s * s))
    return s / s_norm

class Elastic_plastic_finite(Global_residual_plasticity):
    def __init__(self, problem):
        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn, ndim = problem.get_mesh_properties()

        disp_node, disp_val, pres_surf_traction, surf_traction_vector \
            = problem.get_boundary_conditions()

        quad_rule_3D, shape_func_3D = problem.get_volume_basis_functions()
        quad_rule_2D, shape_func_2D = problem.get_surface_basis_functions()
        num_quad_pts = len(quad_rule_3D.wgauss)

        eq_num, num_free_dof, num_pres_dof = initialize_equation(num_nodes, dof_node, disp_node)

        pres_surf_traction_points = nodal_coords[pres_surf_traction]

        print('Number of elements: ', num_elem)
        print('Number of free DOFS: ', num_free_dof)

        params = create_J2_parameters()

        def_type = DefType.FULL_3D

        num_local_resid_dofs = 8
        init_xi = np.zeros(num_local_resid_dofs * num_quad_pts)
        init_xi[-2 * num_quad_pts:-num_quad_pts] = np.ones(num_quad_pts)

        # 4-point quadrature rule for pressure term (hard-coded for now)
        quad_rule_tetra \
            = quadrature_rule.create_quadrature_rule_on_tetrahedron(2)
        gauss_pts_tetra = quad_rule_tetra.xigauss
        shape_func_tetra = interpolants.shape_tetrahedron(gauss_pts_tetra)


        elem_local_resid = partial(self._elem_local_resid,
                                   num_nodes_elem=num_nodes_elem,
                                   ndim=ndim,
                                   gauss_weights_3D=quad_rule_3D.wgauss,
                                   shape_3D=shape_func_3D.values,
                                   dshape_3D=shape_func_3D.gradients,
                                   num_local_resid_dofs=num_local_resid_dofs)

        elem_global_resid = partial(self._elem_global_resid,
                                    num_nodes_elem=num_nodes_elem,
                                    ndim=ndim,
                                    gauss_weights_3D=quad_rule_3D.wgauss,
                                    shape_3D=shape_func_3D.values,
                                    dshape_3D=shape_func_3D.gradients,
                                    num_local_resid_dofs=num_local_resid_dofs,
                                    gauss_weights_3D_2=quad_rule_tetra.wgauss,
                                    shape_3D_2=shape_func_tetra.values,
                                    dshape_3D_2=shape_func_tetra.gradients)

        elem_surf_traction = partial(calc_element_traction_vector_3D,
                                     num_nodes_surf=num_nodes_surf,
                                     ndim=ndim,
                                     gauss_weights_2D=quad_rule_2D.wgauss,
                                     shape_2D=shape_func_2D.values,
                                     dshape_2D=shape_func_2D.gradients)

        super().__init__(elem_global_resid, elem_local_resid, elem_surf_traction, volume_conn,
                         nodal_coords, eq_num, params, num_nodes_elem, dof_node, num_quad_pts, 
                         num_free_dof, num_pres_dof, num_elem, disp_node, disp_val, init_xi,
                         pres_surf_traction_points, pres_surf_traction, surf_traction_vector, 
                         def_type)

    def _elem_local_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D, num_local_resid_dofs):

        elem_disp = u[0:num_nodes_elem * ndim]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]

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

            elem_residual_q = self._local_resid_material_pt(F_q, F_prev_q, params, xi_recast, xi_prev_recast)

            elem_local_residual = elem_local_residual.at[gaussPt3D, :].set(elem_residual_q)

        return elem_local_residual.reshape(-1)

    def _elem_global_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D, num_local_resid_dofs,
            gauss_weights_3D_2, shape_3D_2, dshape_3D_2):

        # extract element displacement and pressure
        elem_disp = u[0:num_nodes_elem * ndim]
        p = u[num_nodes_elem * ndim:]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        mu = E / (2 * (1 + nu))
        K_param = E / (3 * (1 - 2 * nu))
        G_param = E / (2 * (1 + nu))
        alpha = 1000.0
        
        H = 0
        G = jnp.zeros(num_nodes_elem)
        incomp_residual = jnp.zeros(num_nodes_elem)

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

            incomp_residual += w_q * shape_3D_q * (1 / 2 * (J_q ** 2 - 1) / J_q) * dv_q

            # DB contibution (projection onto constant polynomial space)
            H += w_q * 1.0 * dv_q
            G += w_q * shape_3D_q * dv_q

        # alpha / G * (G.T)(H^-1)(G)(p)
        incomp_residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)

        # 4-point quadrature
        num_quad_pts_2 = len(gauss_weights_3D_2)
        for gaussPt3D in range(num_quad_pts_2):
            w_q = gauss_weights_3D_2[gaussPt3D]

            dshape_3D_q = dshape_3D_2[gaussPt3D, :, :]
            shape_3D_q = shape_3D_2[gaussPt3D, :]

            dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_3D_q)
            u_q, grad_u_q = interpolate_vector_3D(elem_disp, shape_3D_q, gradphiXYZ_q, num_nodes_elem)
    
            p_q = interpolate_scalar(p, shape_3D_q)
            
            # (N.T)(alpha / G)(N)(p)
            incomp_residual -= (alpha / G_param + 1 / K_param) * w_q \
                * shape_3D_q * p_q * dv_q

        return jnp.concatenate((S_D_vec.reshape(-1, order='F'), incomp_residual)) 
    
    
    def _local_resid_material_pt(self, F, F_prev, params, xi, xi_prev):
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
        phi_trial = evaluate_yield_function(dev_tau_trial, alpha_prev, params)

        return self._cond_residual(phi_trial, Gamma_bar, I_bar, alpha, b_bar_trial, 
                                   dev_tau_trial, alpha_prev, params, 1.e-14)
    
    @staticmethod
    def _elastic_path(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params):
        I_bar_trial = 1 / 3 * jnp.trace(b_bar_trial)
        Gamma_bar_trial = b_bar_trial - I_bar_trial * jnp.eye(3)
        C_Gamma_bar_elastic = get_vector_from_sym_tensor(Gamma_bar - Gamma_bar_trial, 3) 
        C_I_bar_elastic = I_bar - I_bar_trial
        C_alpha_elastic = alpha - alpha_prev
        return jnp.r_[C_Gamma_bar_elastic, C_I_bar_elastic, C_alpha_elastic]
    
    @staticmethod
    def _plastic_path(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params):
        E = params['elastic']['E']
        nu = params['elastic']['nu']
        mu = E / (2 * (1 + nu))

        I_bar_trial = 1 / 3 * jnp.trace(b_bar_trial)
        Gamma_bar_trial = b_bar_trial - I_bar_trial * jnp.eye(3)

        n = compute_yield_normal(dev_tau_trial)
        C_Gamma_bar_plastic = get_vector_from_sym_tensor(Gamma_bar - Gamma_bar_trial + 2 * jnp.sqrt(3 / 2) \
                                                        * (alpha - alpha_prev) * I_bar_trial * n, 3)
        
        dev_tau = mu * (Gamma_bar_trial - 2 * jnp.sqrt(3 / 2) * (alpha - alpha_prev) * I_bar_trial * n)
        C_alpha_plastic = evaluate_yield_function(dev_tau, alpha, params)

        C_I_bar_plastic = jnp.linalg.det(Gamma_bar + I_bar * jnp.eye(3)) - 1

        return jnp.r_[C_Gamma_bar_plastic, C_I_bar_plastic, C_alpha_plastic]

    def _cond_residual(
        self, phi, Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params, tol):

        def inner_cond_residual(
                Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params): 
            return cond(jnp.abs(phi) < tol, self._plastic_path, self._elastic_path, 
                        Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params)

        def outer_cond_residual(
                Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params): 
            return cond(phi > tol, self._plastic_path, inner_cond_residual, 
                        Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params)
        
        return outer_cond_residual(Gamma_bar, I_bar, alpha, b_bar_trial, dev_tau_trial, alpha_prev, params)

                           



















