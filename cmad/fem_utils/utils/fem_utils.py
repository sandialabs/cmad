import numpy as np
import jax.numpy as jnp

def assemble_global_fields(eq_num, UF, UP):

    # combine free and prescribed displacements into one array
    UUR = np.zeros_like(eq_num, dtype=UF.dtype)

    pos_mask = eq_num > 0
    neg_mask = eq_num < 0

    # Fill in values from UF for positive indices
    UUR[pos_mask] = UF[eq_num[pos_mask] - 1]

    # Fill in values from UP for negative indices
    UUR[neg_mask] = UP[-eq_num[neg_mask] - 1]

    return UUR

def initialize_equation(num_nodes, dof_node, disp_node):

    eq_num = np.zeros((num_nodes, dof_node), dtype=int)
    for i, node in enumerate(disp_node):
        node_number = node[0]
        dof_number = node[1]
        eq_num[node_number][dof_number - 1] = -(i + 1)

    num_free_dof = 0
    for i in range(len(eq_num)):
        for j in range(len(eq_num[i]) - 1):
            if (eq_num[i, j] == 0):
                num_free_dof += 1
                eq_num[i, j] = num_free_dof

    for i in range(len(eq_num)):
        if (eq_num[i, -1] == 0):
            num_free_dof += 1
            eq_num[i, -1] = num_free_dof

    num_pres_dof = num_nodes * dof_node - num_free_dof

    return eq_num, num_free_dof, num_pres_dof

def initialize_equation_thermo(num_nodes, disp_node):
    eq_num = np.zeros(num_nodes, dtype = int)
    for i, node in enumerate(disp_node):
        eq_num[node] = -(i + 1)

    num_free_dof = 0
    for i in range(num_nodes):
        if eq_num[i] == 0:
            num_free_dof += 1
            eq_num[i] = num_free_dof

    num_pres_dof = num_nodes - num_free_dof
    return eq_num, num_free_dof, num_pres_dof

def compute_shape_jacobian(elem_points, dshape):

    J = (dshape @ elem_points).T

    dv = jnp.linalg.det(J)

    # derivatives of shape functions with respect to spacial coordinates
    gradphi = jnp.linalg.inv(J).T @ dshape

    return dv, gradphi

def interpolate_vector_3D(u_elem, shape_3D, gradphiXYZ, num_nodes_elem):

    ux = u_elem[0:num_nodes_elem]
    uy = u_elem[num_nodes_elem:num_nodes_elem * 2]
    uz = u_elem[num_nodes_elem * 2:num_nodes_elem * 3]

    u = jnp.array([jnp.dot(ux, shape_3D),
                   jnp.dot(uy, shape_3D),
                   jnp.dot(uz, shape_3D)])

    grad_u = jnp.vstack([gradphiXYZ @ ux,
                         gradphiXYZ @ uy,
                         gradphiXYZ @ uz])

    return u, grad_u

def interpolate_vector_2D(u_elem, shape_2D, gradphiXY, num_nodes_elem):

    ux = u_elem[0:num_nodes_elem]
    uy = u_elem[num_nodes_elem:num_nodes_elem * 2]

    u = jnp.array([jnp.dot(ux, shape_2D),
                   jnp.dot(uy, shape_2D)])

    grad_u = jnp.vstack([gradphiXY @ ux,
                         gradphiXY @ uy])

    return u, grad_u

def interpolate_scalar(p, shape):
    p = jnp.dot(p, shape)
    return p

def assemble_prescribed(
        num_pres_dof, disp_node, disp_val, eq_num):

    UP = np.zeros(num_pres_dof)
    for i, row in enumerate(disp_node):
        node_number = row[0]
        dof = row[1]
        pres_value = disp_val[i]
        eqn_number = -eq_num[node_number][dof - 1]
        UP[eqn_number - 1] = pres_value

    return UP

def calc_element_traction_vector_3D(
        surf_points, surf_traction_vector, num_nodes_surf, ndim,
        gauss_weights_2D, shape_2D, dshape_2D):
    FEL = jnp.zeros(num_nodes_surf * ndim)

    for gaussPt2D in range(len(gauss_weights_2D)):
        shape_tri_q = shape_2D[gaussPt2D, :]
        dshape_tri_q = dshape_2D[gaussPt2D, :, :]

        J_q = dshape_tri_q @ surf_points

        da_q = jnp.linalg.norm(jnp.cross(J_q[0, :], J_q[1, :]))

        FEL += gauss_weights_2D[gaussPt2D] \
            * (jnp.column_stack([shape_tri_q, shape_tri_q, shape_tri_q]) \
               * surf_traction_vector).T.reshape(-1) * da_q

    return FEL

def calc_element_traction_vector_2D(
        surf_points, surf_traction_vector, num_nodes_surf, ndim,
        gauss_weights_1D, shape_1D, dshape_1D):
    FEL = jnp.zeros(num_nodes_surf * ndim)

    for gaussPt1D in range(len(gauss_weights_1D)):
        shape_1D_q = shape_1D[gaussPt1D, :]
        dshape_1D_q = dshape_1D[gaussPt1D, :]

        J12 = jnp.dot(dshape_1D_q, surf_points[:, 0])
        J22 = jnp.dot(dshape_1D_q, surf_points[:, 1])

        da_q = jnp.sqrt(J12 ** 2 + J22 ** 2)

        FEL += gauss_weights_1D[gaussPt1D] \
            * (jnp.column_stack([shape_1D_q, shape_1D_q]) \
               * surf_traction_vector).T.reshape(-1) * da_q

    return FEL