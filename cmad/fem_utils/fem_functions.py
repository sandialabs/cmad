import numpy as np
import jax.numpy as jnp
import jax

def initialize_equation(num_nodes, dof_node, disp_node):

    eq_num = np.zeros((num_nodes, dof_node), dtype=int)
    for i, node in enumerate(disp_node):
        node_number = node[0]
        direction = node[1]
        eq_num[node_number][direction - 1] = -(i + 1)

    num_free_dof = 0
    for i in range(len(eq_num)):
        for j in range(len(eq_num[i])):
            if (eq_num[i, j] == 0):
                num_free_dof += 1
                eq_num[i, j] = num_free_dof
    num_pres_dof = num_nodes * dof_node - num_free_dof

    return eq_num, num_free_dof, num_pres_dof

def assemble_prescribed_displacement(
        num_pres_dof, disp_node, disp_val, eq_num):

    UP = np.zeros(num_pres_dof)
    for i, row in enumerate(disp_node):
        node_number = row[0]
        dof = row[1]
        displacement = disp_val[i]
        eqn_number = -eq_num[node_number][dof - 1]
        UP[eqn_number - 1] = displacement

    return UP

def calc_element_traction_vector(
        surf_num, pres_surf, nodal_coords, num_nodes_surf, dof_node,
        surf_traction_vector, quad_rule_2D, shape_func_triangle):

    gauss_weights_2D = quad_rule_2D.wgauss
    shape_triangle = shape_func_triangle.values
    dshape_triangle = shape_func_triangle.gradients

    surf_points = nodal_coords[pres_surf[surf_num], :]

    FEL = np.zeros(num_nodes_surf * dof_node)

    for gaussPt2D in range(len(gauss_weights_2D)):
        shape_tri_q = shape_triangle[gaussPt2D, :]
        dshape_tri_q = dshape_triangle[gaussPt2D, :, :]

        J_q = dshape_tri_q @ surf_points

        da_q = np.linalg.norm(np.cross(J_q[0, :], J_q[1, :]))

        FEL += gauss_weights_2D[gaussPt2D] \
            * (np.column_stack([shape_tri_q, shape_tri_q, shape_tri_q]) \
               * surf_traction_vector).T.reshape(-1) * da_q

    return FEL

def assemble_global_traction_vector(
        FEL, pres_surf, surf_num, eq_num, FF, FP):

    surf_conn = pres_surf[surf_num]
    surf_eq_num = eq_num[surf_conn, : ]
    global_indices = surf_eq_num.T.reshape(-1)

    local_pres_indices = np.where(global_indices < 0)[0]
    local_free_indices = np.where(global_indices > 0)[0]

    global_free_indices = global_indices[local_free_indices] - 1
    global_pres_indices = - global_indices[local_pres_indices] - 1

    FF[global_free_indices] += FEL[local_free_indices]
    FP[global_pres_indices] += FEL[local_pres_indices]

def assemble_global_displacement_field(eq_num, UF, UP):

    UUR = np.zeros(eq_num.shape)
    for i,val in enumerate(eq_num):
        for j, row in enumerate(val):
            if row > 0:
                UUR[i, j] = UF[row - 1]
            else:
                UUR[i, j] = UP[- row - 1]
    return UUR

def compute_shape_jacobian(elem_points, dshape_tetra):

    J = (dshape_tetra @ elem_points).T

    dv = jnp.linalg.det(J)

    # derivatives of shape functions with respect to spacial coordinates
    gradphiXYZ = jnp.linalg.inv(J).T @ dshape_tetra

    return dv, gradphiXYZ

def interpolate(u, shape_tetra, gradphiXYZ, num_nodes_elem):

    ux = u[0:num_nodes_elem]
    uy = u[num_nodes_elem:num_nodes_elem * 2]
    uz = u[num_nodes_elem * 2:num_nodes_elem * 3]

    u = jnp.array([jnp.dot(ux, shape_tetra),
                     jnp.dot(uy, shape_tetra),
                     jnp.dot(uz, shape_tetra)])

    grad_u = jnp.vstack([gradphiXYZ @ ux,
                           gradphiXYZ @ uy,
                           gradphiXYZ @ uz])

    return u, grad_u

def compute_elastic_stress(grad_u, params):

    E = params[0]
    nu = params[1]

    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    strain = 1 / 2 * (grad_u + grad_u.T)

    stress = lam * jnp.trace(strain) * jnp.eye(3) + 2 * mu * strain

    return stress

def compute_neo_hookean_stress(grad_u, params):

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

def compute_stress_divergence_vector(u, params, elem_points, num_nodes_elem,
        dof_node, gauss_weights_3D, shape_tetra, dshape_tetra):

    SD_vec = jnp.zeros((num_nodes_elem, dof_node))

    for gaussPt3D in range(len(gauss_weights_3D)):
        w_q = gauss_weights_3D[gaussPt3D]

        dshape_tetra_q = dshape_tetra[gaussPt3D, :, :]
        shape_tetra_q = shape_tetra[gaussPt3D, :]

        dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_tetra_q)
        u_q, grad_u_q = interpolate(u, shape_tetra_q, gradphiXYZ_q, num_nodes_elem)

        stress = compute_neo_hookean_stress(grad_u_q, params)

        SD_vec +=  w_q * gradphiXYZ_q.T @ stress.T * dv_q

    return SD_vec.reshape(-1, order='F')

def assemble_residual(KEL, REL, volume_conn, eq_num, elem_num, KPP, KPF, KFF, KFP, RF, RP):
    elem_conn = volume_conn[elem_num]
    elem_eq_num = eq_num[elem_conn, :]
    global_indices = elem_eq_num.T.reshape(-1)

    local_pres_indices = np.where(global_indices < 0)[0]
    local_free_indices = np.where(global_indices > 0)[0]

    global_free_indices = global_indices[local_free_indices] - 1
    global_pres_indices = - global_indices[local_pres_indices] - 1

    KFF[np.ix_(global_free_indices, global_free_indices)] \
        += KEL[np.ix_(local_free_indices, local_free_indices)]
    KFP[np.ix_(global_free_indices, global_pres_indices)] \
        += KEL[np.ix_(local_free_indices, local_pres_indices)]
    KPF[np.ix_(global_pres_indices, global_free_indices)] \
        += KEL[np.ix_(local_pres_indices, local_free_indices)]
    KPP[np.ix_(global_pres_indices, global_pres_indices)] \
        += KEL[np.ix_(local_pres_indices, local_pres_indices)]
    
    RF[global_free_indices] -= REL[local_free_indices]
    RP[global_pres_indices] -= REL[local_pres_indices]

def solve_fem_newton(num_pres_dof, num_free_dof, num_elem, num_nodes_elem, dof_node,
        num_nodes_surf, surf_traction_vector, params, disp_node,
        disp_val, eq_num, volume_conn, nodal_coords, pres_surf,
        quad_rule_3D, shape_func_tetra, quad_rule_2D, shape_func_triangle, tol, max_iters):
    
    FP = np.zeros(num_pres_dof)
    FF = np.zeros(num_free_dof)

    for surf_num in range(len(pres_surf)):

        # get local traction vector
        FEL = calc_element_traction_vector(surf_num, pres_surf, nodal_coords, num_nodes_surf,
                                           dof_node, surf_traction_vector, quad_rule_2D,
                                           shape_func_triangle)

        # assemble traction vector
        assemble_global_traction_vector(FEL, pres_surf,
                                        surf_num, eq_num, FF, FP)
           
    grad_SD_res = jax.jit(jax.jacfwd(compute_stress_divergence_vector),
                            static_argnames=['num_nodes_elem', 'dof_node'])
    SD_residual_jit = jax.jit(compute_stress_divergence_vector,static_argnames=['num_nodes_elem', 'dof_node'])

    gauss_weights_3D = quad_rule_3D.wgauss
    shape_tetra = shape_func_tetra.values
    dshape_tetra = shape_func_tetra.gradients

    ## Prescribed displacements
    UP = assemble_prescribed_displacement(num_pres_dof, disp_node,
                                          disp_val, eq_num)
    # free displacements
    UF = np.zeros(num_free_dof)

    for i in range(max_iters):
        UUR = assemble_global_displacement_field(eq_num, UF, UP)
        RF = np.zeros(num_free_dof)
        RP = np.zeros(num_pres_dof)
        KPP = np.zeros((num_pres_dof, num_pres_dof))
        KPF = np.zeros((num_pres_dof, num_free_dof))
        KFP = np.zeros((num_free_dof, num_pres_dof))
        KFF = np.zeros((num_free_dof, num_free_dof))
        # assemble global stiffness and force
        for elem_num in range(0, num_elem):
            u_elem = UUR[volume_conn[elem_num], :].reshape(-1, order='F')
            elem_points = nodal_coords[volume_conn[elem_num], :]
            # get element tangent stiffness matrix
            KEL = grad_SD_res(u_elem, params, elem_points, num_nodes_elem, dof_node,
                                gauss_weights_3D, shape_tetra, dshape_tetra)
            
            REL = SD_residual_jit(u_elem, params, elem_points, num_nodes_elem, dof_node,
                                gauss_weights_3D, shape_tetra, dshape_tetra)

            assemble_residual(np.array(KEL), np.array(REL), volume_conn,
                              eq_num, elem_num, KPP, KPF, KFF, KFP, RF, RP)
            
        print("||R||: ", np.linalg.norm(RF + FF))

        if (np.linalg.norm(RF + FF) < tol):
            return UUR

        UF += np.linalg.solve(KFF, RF + FF)