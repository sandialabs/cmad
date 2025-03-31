import jax
import jax.numpy as jnp
import numpy as np

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

def interpolate_u(u, shape_tetra, gradphiXYZ, num_nodes_elem):

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

def interpolate_p(p, shape_tetra):
    p = jnp.dot(p, shape_tetra)
    return p

def compute_piola_stress(grad_u, p, params):

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

def compute_stress_divergence_vector(u, p, params, elem_points, num_nodes_elem,
        dof_node, gauss_weights_3D, shape_tetra, dshape_tetra):

    S_D_vec = jnp.zeros((num_nodes_elem, dof_node))

    for gaussPt3D in range(len(gauss_weights_3D)):
        w_q = gauss_weights_3D[gaussPt3D]

        dshape_tetra_q = dshape_tetra[gaussPt3D, :, :]
        shape_tetra_q = shape_tetra[gaussPt3D, :]

        dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_tetra_q)
        u_q, grad_u_q = interpolate_u(u, shape_tetra_q, gradphiXYZ_q, num_nodes_elem)
        p_q = interpolate_p(p, shape_tetra_q)

        stress = compute_piola_stress(grad_u_q, p_q, params)

        S_D_vec +=  w_q * gradphiXYZ_q.T @ stress.T * dv_q

    return S_D_vec.reshape(-1, order='F')

def compute_incompressibility_residual(u, p, params, elem_points, num_nodes_elem,
        gauss_weights_3D, shape_tetra, dshape_tetra):

    E = params[0]
    nu = params[1]
    G_param = E/(2 * (1 + nu))
    alpha = 1.0

    H = 0
    G = jnp.zeros(num_nodes_elem)
    residual = jnp.zeros(num_nodes_elem)

    for gaussPt3D in range(len(gauss_weights_3D)):
        w_q = gauss_weights_3D[gaussPt3D]

        dshape_tetra_q = dshape_tetra[gaussPt3D, :, :]
        shape_tetra_q = shape_tetra[gaussPt3D, :]

        dv_q, gradphiXYZ_q = compute_shape_jacobian(elem_points, dshape_tetra_q)
        u_q, grad_u_q = interpolate_u(u, shape_tetra_q, gradphiXYZ_q, num_nodes_elem)

        F = jnp.eye(3) + grad_u_q 

        residual +=  w_q *  shape_tetra_q * (jnp.linalg.det(F) - 1) * dv_q
        
        # DB contibution (projection onto constant polynomial space)
        H += w_q * 1.0 * dv_q
        G += w_q * shape_tetra_q * dv_q

        # (N.T)(alpha / G)(N)(p)
        residual -= alpha / G_param * w_q * shape_tetra_q * jnp.dot(shape_tetra_q, p) * dv_q

    # alpha / G * (G.T)(H^-1)(G)(p)
    residual += alpha / G_param * G * (1 / H) * jnp.dot(G, p)    

    return residual 

def assemble_module(KEL, C1_EL, C2_EL, VEL, SD_EL_res, J_EL_res, volume_conn,
        eq_num_u, eq_num_p, elem_num, KFF, C1FF, C2FF, VFF, SD_F, J_F):
    
    elem_conn = volume_conn[elem_num]

    elem_eq_num_u = eq_num_u[elem_conn, :]
    global_indices_u = elem_eq_num_u.T.reshape(-1)
    local_free_indices_u = np.where(global_indices_u > 0)[0]
    global_free_indices_u = global_indices_u[local_free_indices_u] - 1

    global_indices_p = eq_num_p[elem_conn]
    local_free_indices_p = np.where(global_indices_p > 0)[0]
    global_free_indices_p = global_indices_p[local_free_indices_p] - 1

    KFF[np.ix_(global_free_indices_u, global_free_indices_u)] \
        += KEL[np.ix_(local_free_indices_u, local_free_indices_u)]
    
    C1FF[np.ix_(global_free_indices_u, global_free_indices_p)] \
        += C1_EL[np.ix_(local_free_indices_u, local_free_indices_p)] 

    C2FF[np.ix_(global_free_indices_p, global_free_indices_u)] \
        += C2_EL[np.ix_(local_free_indices_p, local_free_indices_u)] 

    VFF[np.ix_(global_free_indices_p, global_free_indices_p)] \
        += VEL[np.ix_(local_free_indices_p, local_free_indices_p)]

    SD_F[global_free_indices_u] += SD_EL_res[local_free_indices_u]

    J_F[global_free_indices_p] += J_EL_res[local_free_indices_p]

def solve_fem_newton(num_pres_dof, num_free_dof, num_elem, num_nodes_elem, dof_node,
        num_nodes_surf, surf_traction_vector, params, disp_node,
        disp_val, eq_num_u, eq_num_p, volume_conn, nodal_coords, pres_surf,
        quad_rule_3D, shape_func_tetra, quad_rule_2D, shape_func_triangle, tol, max_iters):
    
    num_nodes = len(nodal_coords)
    
    FP = np.zeros(num_pres_dof)
    FF = np.zeros(num_free_dof)

    for surf_num in range(len(pres_surf)):

        # get local traction vector
        FEL = calc_element_traction_vector(surf_num, pres_surf, nodal_coords, num_nodes_surf,
                                           dof_node, surf_traction_vector, quad_rule_2D,
                                           shape_func_triangle)

        # assemble traction vector
        assemble_global_traction_vector(FEL, pres_surf,
                                        surf_num, eq_num_u, FF, FP)

    #derivative of stress-divergence residual w.r.t u
    grad_SD_res_u = jax.jit(jax.jacfwd(compute_stress_divergence_vector),
                            static_argnames=['num_nodes_elem', 'dof_node'])
    #derivative of stress-divergence residual w.r.t p
    grad_SD_res_p = jax.jit(jax.jacfwd(compute_stress_divergence_vector, argnums=1),
                            static_argnames=['num_nodes_elem', 'dof_node'])
    #evaluate stress-divergence residual
    SD_residual_jit = jax.jit(compute_stress_divergence_vector,
                              static_argnames=['num_nodes_elem', 'dof_node'])
    
    #derivative of incompressibility residual w.r.t u
    grad_J_res_u = jax.jit(jax.jacfwd(compute_incompressibility_residual),
                            static_argnames=['num_nodes_elem'])
    #derivative of incompressibility residual w.r.t p
    grad_J_res_p = jax.jit(jax.jacfwd(compute_incompressibility_residual, argnums=1), 
                            static_argnames=['num_nodes_elem'])
    #evaluate incompressibility residual
    J_residual_jit = jax.jit(compute_incompressibility_residual,
                             static_argnames=['num_nodes_elem'])
    
    gauss_weights_3D = quad_rule_3D.wgauss
    shape_tetra = shape_func_tetra.values
    dshape_tetra = shape_func_tetra.gradients

    ## Prescribed displacements
    UP = assemble_prescribed_displacement(num_pres_dof, disp_node,
                                          disp_val, eq_num_u)
    # free displacements
    UF = np.zeros(num_free_dof)

    # free pressure nodes (fix pressure at one node)
    pF = np.zeros(num_nodes - 1)

    M = np.zeros((num_free_dof + num_nodes - 1, num_free_dof + num_nodes - 1))
    f = np.zeros(num_free_dof + num_nodes - 1)

    for i in range(max_iters):
        # global displacement and pressure fields
        UUR = assemble_global_displacement_field(eq_num_u, UF, UP)
        pUR = np.append(pF.copy(), 0.0)

        # initialize
        KFF = np.zeros((num_free_dof, num_free_dof))
        C1FF = np.zeros((num_free_dof, num_nodes - 1))
        C2FF = np.zeros((num_nodes - 1, num_free_dof))
        VFF = np.zeros((num_nodes - 1, num_nodes - 1))
        SD_F = np.zeros(num_free_dof)
        J_F = np.zeros(num_nodes - 1)

        # assemble global stiffness and force
        for elem_num in range(0, num_elem):
            u_elem = UUR[volume_conn[elem_num], :].reshape(-1, order='F')
            p_elem = pUR[volume_conn[elem_num]]
            elem_points = nodal_coords[volume_conn[elem_num], :]
           
            # get element tangent matrices
            KEL = grad_SD_res_u(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                dof_node, gauss_weights_3D, shape_tetra, dshape_tetra)
            
            C1_EL = grad_SD_res_p(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                dof_node, gauss_weights_3D, shape_tetra, dshape_tetra)
            
            C2_EL = grad_J_res_u(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                 gauss_weights_3D, shape_tetra, dshape_tetra)
            
            VEL = grad_J_res_p(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                 gauss_weights_3D, shape_tetra, dshape_tetra)
            
            SD_EL_res = SD_residual_jit(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                dof_node, gauss_weights_3D, shape_tetra, dshape_tetra)
            
            J_EL_res = J_residual_jit(u_elem, p_elem, params, elem_points, num_nodes_elem,
                                 gauss_weights_3D, shape_tetra, dshape_tetra)

            assemble_module(np.array(KEL), np.array(C1_EL), np.array(C2_EL), np.array(VEL),
                            np.array(SD_EL_res), np.array(J_EL_res), volume_conn, eq_num_u,
                            eq_num_p, elem_num, KFF, C1FF, C2FF, VFF, SD_F, J_F)
            
        f[0:num_free_dof] = SD_F - FF
        f[num_free_dof:] = J_F
            
        print("||R||: ", np.linalg.norm(f))

        if (np.linalg.norm(f) < tol):
            return UUR, pUR
        
        M[0:num_free_dof, 0:num_free_dof] = KFF
        M[0:num_free_dof, num_free_dof:] = C1FF
        M[num_free_dof:, 0:num_free_dof] = C2FF
        M[num_free_dof:, num_free_dof:] = VFF
        
        # print('Cond(M): ', np.linalg.cond(M))
        
        delta = np.linalg.solve(M, -f)

        UF += delta[0:num_free_dof]
        pF += delta[num_free_dof:]
