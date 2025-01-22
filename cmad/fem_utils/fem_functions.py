import numpy as np
import time
import jax

from cmad.fem_utils.fem_functions_AD import elem_residual

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
    
def calc_element_stiffness_matrix(
        elem_points, num_nodes_elem, dof_node,
        params, gauss_weights_3D, dshape_tetra):

    E = params[0]
    nu = params[1]

    k = E / (3 * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    c1 = k + 4/3 * mu
    c2 = k - 2/3 * mu
    material_stiffness = np.array([[c1, c2, c2, 0, 0, 0],
                                   [c2, c1, c2, 0, 0, 0],
                                   [c2, c2, c1, 0, 0, 0],
                                   [0, 0, 0, mu, 0, 0],
                                   [0, 0, 0, 0, mu, 0],
                                   [0, 0, 0, 0, 0, mu]])

    KEL = np.zeros((num_nodes_elem * dof_node, num_nodes_elem * dof_node))

    for gaussPt3D in range(len(gauss_weights_3D)):
        w_q = gauss_weights_3D[gaussPt3D]

        dshape_tetra_q = dshape_tetra[gaussPt3D, :, :]

        J_q = (dshape_tetra_q @ elem_points).T

        dv_q = np.linalg.det(J_q)

        gradphiXYZ_q = np.linalg.inv(J_q).T @ dshape_tetra_q

        D_phi_q = np.zeros((6, num_nodes_elem * dof_node))
        D_phi_q[0, 0:num_nodes_elem] = gradphiXYZ_q[0, :]
        D_phi_q[1, num_nodes_elem:num_nodes_elem * 2] = gradphiXYZ_q[1, :]
        D_phi_q[2, num_nodes_elem * 2:num_nodes_elem * 3] = gradphiXYZ_q[2, :]
        D_phi_q[3, 0:num_nodes_elem * 2] \
            = np.concatenate((gradphiXYZ_q[1, :], gradphiXYZ_q[0, :]))
        D_phi_q[4, num_nodes_elem:num_nodes_elem * 3] \
            = np.concatenate((gradphiXYZ_q[2, :], gradphiXYZ_q[1, :]))
        D_phi_q[5, 0:num_nodes_elem] = gradphiXYZ_q[2, :]
        D_phi_q[5, num_nodes_elem * 2:num_nodes_elem * 3] = gradphiXYZ_q[0, :]
                                                                               
        KEL += w_q * D_phi_q.T @ material_stiffness @ D_phi_q * dv_q
        
    return KEL

def calc_element_traction_vector(
        surf_num, pres_surf, nodal_coords, num_nodes_surf, dof_node,
        surf_traction_vector, quad_rule_2D, shape_func_triangle):

    gauss_weights_2D = quad_rule_2D.wgauss
    shape_triangle = shape_func_triangle.values
    dshape_triangle = shape_func_triangle.gradients

    surf_points = nodal_coords[pres_surf[surf_num], :]

    PEL = np.zeros(num_nodes_surf * dof_node)

    for gaussPt2D in range(len(gauss_weights_2D)):
        shape_tri_q = shape_triangle[gaussPt2D, :]
        dshape_tri_q = dshape_triangle[gaussPt2D, :, :]

        J_q = dshape_tri_q @ surf_points

        da_q = np.linalg.norm(np.cross(J_q[0, :], J_q[1, :]))

        PEL += gauss_weights_2D[gaussPt2D] \
            * (np.column_stack([shape_tri_q, shape_tri_q, shape_tri_q]) \
               * surf_traction_vector).T.reshape(-1) * da_q
        
    return PEL

def assemble_global_stiffness(
        KEL, volume_conn, eq_num, elem_num, KPP, KPF, KFF, KFP):

    elem_conn = volume_conn[elem_num]
    elem_eq_num = eq_num[elem_conn, : ]
    global_indices = elem_eq_num.T.reshape(-1)

    local_pres_indices = np.where(global_indices < 0)[0]
    local_free_indices = np.where(global_indices > 0)[0]

    global_free_indices = global_indices[local_free_indices] - 1
    global_pres_indices = - global_indices[local_pres_indices] - 1

    KFF[np.ix_(global_free_indices,global_free_indices)] \
        += KEL[np.ix_(local_free_indices,local_free_indices)]
    KFP[np.ix_(global_free_indices,global_pres_indices)] \
        += KEL[np.ix_(local_free_indices,local_pres_indices)]
    KPF[np.ix_(global_pres_indices,global_free_indices)] \
        += KEL[np.ix_(local_pres_indices,local_free_indices)]
    KPP[np.ix_(global_pres_indices,global_pres_indices)] \
        += KEL[np.ix_(local_pres_indices,local_pres_indices)]


def assemble_global_traction_vector(
        PEL, pres_surf, surf_num, eq_num, PF, PP):

    surf_conn = pres_surf[surf_num]
    surf_eq_num = eq_num[surf_conn, : ]
    global_indices = surf_eq_num.T.reshape(-1)

    local_pres_indices = np.where(global_indices < 0)[0]
    local_free_indices = np.where(global_indices > 0)[0]

    global_free_indices = global_indices[local_free_indices] - 1
    global_pres_indices = - global_indices[local_pres_indices] - 1

    PF[global_free_indices] += PEL[local_free_indices]
    PP[global_pres_indices] += PEL[local_pres_indices]


def assemble_module(
        num_pres_dof, num_free_dof, num_elem, num_nodes_elem, dof_node,
        num_nodes_surf, surf_traction_vector, params, disp_node,
        disp_val, eq_num, volume_conn, nodal_coords, pres_surf,
        quad_rule_3D, shape_func_tetra, quad_rule_2D, shape_func_triangle):

    # Initialize arrays that need to be returned (KPP, KPF, KFF, KFP, PP)
    KPP = np.zeros((num_pres_dof, num_pres_dof))
    KPF = np.zeros((num_pres_dof, num_free_dof))
    KFP = np.zeros((num_free_dof, num_pres_dof))
    KFF = np.zeros((num_free_dof, num_free_dof))
    PP = np.zeros(num_pres_dof)
    PF = np.zeros(num_free_dof)
                                                                               
    ## Prescribe boundary conditions
    UP = assemble_prescribed_displacement(num_pres_dof, disp_node,
                                          disp_val, eq_num)
    
    gauss_weights_3D = quad_rule_3D.wgauss
    shape_tetra = shape_func_tetra.values
    dshape_tetra = shape_func_tetra.gradients

    start = time.time()
    # assemble global stiffness and force
    for elem_num in range(0, num_elem):

        elem_points = nodal_coords[volume_conn[elem_num], :]
        # get element stiffness matrix
        KEL = calc_element_stiffness_matrix(elem_points, num_nodes_elem,dof_node,
                                            params, gauss_weights_3D, dshape_tetra)

        # assemble global stiffness
        assemble_global_stiffness(KEL, volume_conn, eq_num, 
                                  elem_num, KPP, KPF, KFF, KFP)
    end = time.time()
    print("Time to assemble stiffness matrix: ", end - start)

    start = time.time()
    for surf_num in range(len(pres_surf)):
        
        # get local traction vector
        PEL = calc_element_traction_vector(surf_num, pres_surf, nodal_coords, num_nodes_surf,
                                           dof_node,surf_traction_vector, quad_rule_2D,
                                           shape_func_triangle)

        # assemble traction vector
        assemble_global_traction_vector(PEL, pres_surf,
                                        surf_num, eq_num, PF, PP)
    end = time.time()
    print("Time to assemble load vector: ", end - start)

    return KPP, KPF, KFF, KFP, PF, PP, UP

def assemble_module_AD(
        num_pres_dof, num_free_dof, num_elem, num_nodes_elem, dof_node,
        num_nodes_surf, surf_traction_vector, params, disp_node,
        disp_val, eq_num, volume_conn, nodal_coords, pres_surf,
        quad_rule_3D, shape_func_tetra, quad_rule_2D, shape_func_triangle):

    # Initialize arrays that need to be returned (KPP, KPF, KFF, KFP, PP)
    KPP = np.zeros((num_pres_dof, num_pres_dof))
    KPF = np.zeros((num_pres_dof, num_free_dof))
    KFP = np.zeros((num_free_dof, num_pres_dof))
    KFF = np.zeros((num_free_dof, num_free_dof))
    PP = np.zeros(num_pres_dof)
    PF = np.zeros(num_free_dof)
                                                                               
    ## Prescribe boundary conditions
    UP = assemble_prescribed_displacement(num_pres_dof, disp_node,
                                          disp_val, eq_num)

    start = time.time()

    grad_residual = jax.jit(jax.jacfwd(elem_residual),
                            static_argnames=['num_nodes_elem', 'dof_node'])
    
    u_guess = np.ones(num_nodes_elem * dof_node)

    gauss_weights_3D = quad_rule_3D.wgauss
    shape_tetra = shape_func_tetra.values
    dshape_tetra = shape_func_tetra.gradients

    # assemble global stiffness and force
    for elem_num in range(0, num_elem):

        elem_points = nodal_coords[volume_conn[elem_num], :]
        # get element stiffness matrix
        KEL = grad_residual(u_guess, params, elem_points, num_nodes_elem, dof_node,
                            gauss_weights_3D, shape_tetra, dshape_tetra)

        # assemble global stiffness
        assemble_global_stiffness(np.array(KEL), volume_conn, eq_num, 
                                  elem_num, KPP, KPF, KFF, KFP)
    end = time.time()
    print("Time to assemble stiffness matrix: ", end - start)

    start = time.time()
    for surf_num in range(len(pres_surf)):
        
        # get local traction vector
        PEL = calc_element_traction_vector(surf_num, pres_surf, nodal_coords, num_nodes_surf,
                                           dof_node,surf_traction_vector, quad_rule_2D,
                                           shape_func_triangle)

        # assemble traction vector
        assemble_global_traction_vector(PEL, pres_surf,
                                        surf_num, eq_num, PF, PP)
    end = time.time()
    print("Time to assemble load vector: ", end - start)

    return KPP, KPF, KFF, KFP, PF, PP, UP

def assemble_global_displacement_field(eq_num, UF, UP):

    UUR = np.zeros(eq_num.shape)
    for i,val in enumerate(eq_num):
        for j, row in enumerate(val):
            if row > 0:
                UUR[i, j] = UF[row - 1]
            else:
                UUR[i, j] = UP[- row - 1]
    return UUR

def solve_module(KPP, KPF, KFF, KFP, PP, PF, UP, eq_num):

    UF = np.linalg.solve(KFF, PF - KFP @ UP)
    R = KPP @ UP + KPF @ UF - PP

    UUR = assemble_global_displacement_field(eq_num, UF, UP)

    return UUR, UF, R