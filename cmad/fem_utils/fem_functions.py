import numpy as np
import time

def initialize_equation(NUM_NODES, DOF_NODE, disp_node):

    eq_num = np.zeros((NUM_NODES, DOF_NODE), dtype = int)
    for i, node in enumerate(disp_node):
        node_number = node[0]
        direction = node[1]
        eq_num[node_number][direction - 1] = -(i + 1)
        
    NUM_FREE_DOF = 0
    for i in range(len(eq_num)):
        for j in range(len(eq_num[i])):
            if (eq_num[i, j] == 0):
                NUM_FREE_DOF += 1
                eq_num[i, j] = NUM_FREE_DOF
    NUM_PRES_DOF = NUM_NODES * DOF_NODE - NUM_FREE_DOF

    return eq_num, NUM_FREE_DOF, NUM_PRES_DOF

def assemble_prescribed_displacement(
        NUM_PRES_DOF, disp_node, disp_val, eq_num):
    
    UP = np.zeros(NUM_PRES_DOF)
    for i, row in enumerate(disp_node):
        node_number = row[0]
        dof = row[1]
        displacement = disp_val[i]
        eqn_number = -eq_num[node_number][dof - 1]
        UP[eqn_number - 1] = displacement

    return UP
    
def calc_element_stiffness_matrix(
        elem_num, volume_conn, nodal_coords, NUM_NODES_ELE,
        DOF_NODE, E, nu, quad_rule_3D, shape_func_tetra):

    gauss_weights_3D = quad_rule_3D.wgauss
    dshape_tetra = shape_func_tetra.gradients

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

    elem_points = nodal_coords[volume_conn[elem_num], :]

    KEL = np.zeros((NUM_NODES_ELE * DOF_NODE, NUM_NODES_ELE * DOF_NODE))

    for gaussPt3D in range(len(gauss_weights_3D)):
        Nzeta = dshape_tetra[gaussPt3D, :, :]

        J = (Nzeta @ elem_points).T

        det_J = np.linalg.det(J)

        gradphiXYZ = np.linalg.inv(J).T @ Nzeta

        D_phi = np.zeros((6, NUM_NODES_ELE * DOF_NODE))
        D_phi[0, 0:NUM_NODES_ELE] = gradphiXYZ[0, :]
        D_phi[1, NUM_NODES_ELE:NUM_NODES_ELE * 2] = gradphiXYZ[1, :]
        D_phi[2, NUM_NODES_ELE * 2:NUM_NODES_ELE * 3] = gradphiXYZ[2, :]
        D_phi[3, 0:NUM_NODES_ELE * 2] \
            = np.concatenate((gradphiXYZ[1, :], gradphiXYZ[0, :]))
        D_phi[4, NUM_NODES_ELE:NUM_NODES_ELE * 3] \
            = np.concatenate((gradphiXYZ[2, :], gradphiXYZ[1, :]))
        D_phi[5, 0:NUM_NODES_ELE] = gradphiXYZ[2, :]
        D_phi[5, NUM_NODES_ELE * 2:NUM_NODES_ELE * 3] = gradphiXYZ[0, :]
                                                                               
        KEL += gauss_weights_3D[gaussPt3D] \
            * D_phi.T @ material_stiffness @ D_phi * det_J
        
    return KEL

def calc_element_traction_vector(
        surf_num, pres_surf, nodal_coords, NUM_NODES_SURF, DOF_NODE,
        SURF_TRACTION_VECTOR, quad_rule_2D, shape_func_triangle):

    gauss_weights_2D = quad_rule_2D.wgauss
    shape_triangle = shape_func_triangle.values
    dshape_triangle = shape_func_triangle.gradients

    surf_points = nodal_coords[pres_surf[surf_num], :]

    PEL = np.zeros(NUM_NODES_SURF * DOF_NODE)

    for gaussPt2D in range(len(gauss_weights_2D)):
        N = shape_triangle[gaussPt2D, :]
        Nzeta = dshape_triangle[gaussPt2D, :, :]

        J = Nzeta @ surf_points

        Js = np.linalg.norm(np.cross(J[0, :], J[1, :]))

        PEL += gauss_weights_2D[gaussPt2D] \
            * (np.column_stack([N, N, N]) \
               * SURF_TRACTION_VECTOR).T.reshape(-1) * Js
        
    return PEL

def assemble_global_stiffness(
        KEL, volume_conn, eq_num, ELEM_NUM, KPP, KPF, KFF, KFP):

    elem_conn = volume_conn[ELEM_NUM]
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
        NUM_PRES_DOF, NUM_FREE_DOF, NUM_ELE, NUM_NODES_ELE, DOF_NODE,
        NUM_NODES_SURF, SURF_TRACTION_VECTOR, E, nu, disp_node,
        disp_val, eq_num, volume_conn, nodal_coords, pres_surf,
        quad_rule_3D, shape_func_tetra, quad_rule_2D, shape_func_triangle):

    # Initialize arrays that need to be returned (KPP, KPF, KFF, KFP, PP)
    KPP = np.zeros((NUM_PRES_DOF, NUM_PRES_DOF))
    KPF = np.zeros((NUM_PRES_DOF, NUM_FREE_DOF))
    KFP = np.zeros((NUM_FREE_DOF, NUM_PRES_DOF))
    KFF = np.zeros((NUM_FREE_DOF, NUM_FREE_DOF))
    PP = np.zeros(NUM_PRES_DOF)
    PF = np.zeros(NUM_FREE_DOF)
                                                                               
    ## Prescribe boundary conditions
    UP = assemble_prescribed_displacement(NUM_PRES_DOF, disp_node,
                                          disp_val, eq_num)

    start = time.time()
    # assemble global stiffness and force
    for elem_num in range(0, NUM_ELE):

        # get element stiffness matrix
        KEL = calc_element_stiffness_matrix(
            elem_num, volume_conn, nodal_coords, NUM_NODES_ELE,
            DOF_NODE, E, nu, quad_rule_3D, shape_func_tetra)

        # assemble global stiffness
        assemble_global_stiffness(KEL, volume_conn, eq_num, 
                                  elem_num, KPP, KPF, KFF, KFP)
    end = time.time()
    print("Time to assemble stiffness matrix: ", end - start)

    start = time.time()
    for surf_num in range(len(pres_surf)):
        
        # get local traction vector
        PEL = calc_element_traction_vector(
            surf_num, pres_surf, nodal_coords, NUM_NODES_SURF, DOF_NODE,
            SURF_TRACTION_VECTOR, quad_rule_2D, shape_func_triangle)

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