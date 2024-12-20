from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.fem_functions import (initialize_equation, 
                                          assemble_module, 
                                          solve_module)
import time

"""

DOF_NODE: number of degrees of freedom per node
NUM_NODES: total number of nodes in the mesh
NUM_NODE_ELE: number of nodes per element
NUM_ELE: number of elements
NUM_NODES_SURF: number of nodes per surface element
NUM_FREE_DOF: number of free degrees of freedom
NUM_PRES_DOF: number of prescribed degrees of freedom
SURF_TRACTION_VECTOR: surface traction vector
E: Youngs Modulus
nu: Poisson's ratio
disp_node: (NUM_PRES_DOF x 2) array that specifies 
    which node and dof is prescribed
disp_val: (NUM_PRED_DOF x 1) array of values 
    of the prescribed displacements
eq_num: (NUM_NODES x DOF_NODE) array that specifies where 
    each node and its DOFs belong in the global stifness matrix
volume_conn: connectivity matrix for 3D elements
pres_surf: connectivity for surfaces that have a prescribed traction
nodal_coords: spacial coordinates for each node in mesh
    
"""

order = 3
problem = fem_problem("patch_B", order)

DOF_NODE, NUM_NODES, NUM_NODES_ELE, NUM_ELE, NUM_NODES_SURF, \
    nodal_coords, volume_conn = problem.get_mesh_properties()

disp_node, disp_val, pres_surf, SURF_TRACTION_VECTOR \
    = problem.get_boundary_conditions()

quad_rule_3D, shape_func_tetra = problem.get_3D_basis_functions()
quad_rule_2D, shape_func_triangle = problem.get_2D_basis_functions()


print("Number of elements:", NUM_ELE)

E = 200
nu = 0.3

eq_num, NUM_FREE_DOF, NUM_PRES_DOF \
    = initialize_equation(NUM_NODES, DOF_NODE, disp_node)

KPP, KPF, KFF, KFP, PF, PP, UP \
    = assemble_module(NUM_PRES_DOF, NUM_FREE_DOF, NUM_ELE, NUM_NODES_ELE,
                      DOF_NODE, NUM_NODES_SURF, SURF_TRACTION_VECTOR, E, nu,
                      disp_node, disp_val, eq_num, volume_conn, nodal_coords, 
                      pres_surf, quad_rule_3D, shape_func_tetra, quad_rule_2D,
                      shape_func_triangle)

solve_start = time.time()

UUR, UF, R = solve_module(KPP, KPF, KFF, KFP, PP, PF, UP, eq_num)

print(UUR)

solve_end = time.time()

#problem.save_data("simple_shear", UUR)

print("Solve time: ", solve_end - solve_start)

