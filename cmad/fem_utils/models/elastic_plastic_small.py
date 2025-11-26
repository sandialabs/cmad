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
from cmad.parameters.parameters import Parameters
from cmad.models.deformation_types import DefType
from cmad.models.small_elastic_plastic import SmallElasticPlastic

def create_J2_parameters():
    E = 200.e3
    nu = 0.249
    Y = 349.
    K = 1.e5
    S = 1.23e3
    D = 0.55

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

class Elastic_plastic_small(Global_residual_plasticity):
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

        elem_global_resid = partial(self._elem_global_resid,
                                    num_nodes_elem=num_nodes_elem,
                                    ndim=ndim,
                                    gauss_weights_3D=quad_rule_3D.wgauss,
                                    shape_3D=shape_func_3D.values,
                                    dshape_3D=shape_func_3D.gradients,
                                    num_local_resid_dofs=num_local_resid_dofs)

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

    def _elem_global_resid(
            self, u, u_prev, params, xi, xi_prev, elem_points, num_nodes_elem,
            ndim, gauss_weights_3D, shape_3D, dshape_3D, num_local_resid_dofs):

        # extract element displacement and pressure
        elem_disp = u[0:num_nodes_elem * ndim]
        elem_disp_prev = u_prev[0:num_nodes_elem * ndim]
        p = u[num_nodes_elem * ndim:]

        E = params['elastic']['E']
        nu = params['elastic']['nu']
        G_param = E / (2 * (1 + nu))
        K_param = E / (3 * (1 - 2 * nu))
        alpha = 0.01

        # incompressibility residual
        H = 0
        G = jnp.zeros(num_nodes_elem)
        incomp_residual = jnp.zeros(num_nodes_elem)

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
            p_q = interpolate_scalar(p, shape_3D_q)

            F_q = [grad_u_q + jnp.eye(ndim)]
            F_prev_q = [grad_u_prev_q + jnp.eye(ndim)]

            ep_q = elem_ep[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            ep_prev_q = elem_ep_prev[gaussPt3D * ep_dofs: (gaussPt3D + 1) * ep_dofs]
            alpha_q = jnp.array([elem_alpha[gaussPt3D - num_quad_pts]])
            alpha_prev_q = jnp.array([elem_alpha_prev[gaussPt3D - num_quad_pts]])

            xi_recast = [ep_q, alpha_q]
            xi_prev_recast = [ep_prev_q, alpha_prev_q]

            stress = self._cauchy_fun(xi_recast, xi_prev_recast, params, F_q, F_prev_q)
            stress = stress - 1 / ndim * jnp.trace(stress) * jnp.eye(ndim) + p_q * jnp.eye(ndim)

            S_D_vec += w_q * gradphiXYZ_q.T @ stress.T * dv_q

            incomp_residual += w_q *  shape_3D_q * (jnp.trace(grad_u_q)) * dv_q

            # DB contibution (projection onto constant polynomial space)
            H += w_q * 1.0 * dv_q
            G += w_q * shape_3D_q * dv_q

            # (N.T)(alpha / G)(N)(p)
            incomp_residual -= (alpha / G_param + 1 / K_param) * w_q \
                * shape_3D_q * jnp.dot(shape_3D_q, p) * dv_q

        # alpha / G * (G.T)(H^-1)(G)(p)
        incomp_residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)

        return jnp.concatenate((S_D_vec.reshape(-1, order='F'), incomp_residual))












