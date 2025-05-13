from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.fem_functions import (initialize_equation,
                                          solve_fem_newton)
import numpy as np

"""

dof_node: number of degrees of freedom per node
num_nodes: total number of nodes in the mesh
num_node_elem: number of nodes per element
num_elem: number of elements
num_nodes_surf: number of nodes per surface element
num_free_dof: number of free degrees of freedom
num_pres_dof: number of prescribed degrees of freedom
surf_traction_vector: surface traction vector
E: Youngs Modulus
nu: Poisson's ratio
disp_node: (num_pres_dof x 2) array that specifies
    which node and dof is prescribed
disp_val: (NUM_PRED_DOF x 1) array of values
    of the prescribed displacements
eq_num: (num_nodes x dof_node) array that specifies where
    each node and its DOFs belong in the global stifness matrix
volume_conn: connectivity matrix for 3D elements
pres_surf: connectivity for surfaces that have a prescribed traction
nodal_coords: spacial coordinates for each node in mesh

In the definitions below, "P" stands for prescribed displacements
and "F" stands for free displacments

KPP: (num_pres_dof x num_pres_dof) partion of stiffness matrix with rows
    and columns corresponding with prescribed DOFS
KPF: (num_pres_dof x num_free_dof) partion of stiffness matrix with rows
    corresponding with prescribed DOFS and columns corresponding with free DOFS
KFF: (num_free_dof x num_free_dof) partion of stiffness matrix with rows and
    columns corresponding with free DOFS
KFP: (num_free_dof x num_free_dof) partion of stiffness matrix with rows
    corresponding with free DOFS and columns corresponding with prescribed DOFS
FF: (num_free_dof, ) RHS vector corresponding with free DOFS
FP: (num_pres_dof, ) RHS vector corresponding with prescribed DOFS
UP: (num_pres_dof, ) vector of prescribed displacements
UF: (num_free_dof, ) vector of free displacements (the ones that we solve for)
UUR: (num_nodes x 3) matrix of displacements at all nodes in the mesh
"""

order = 3
problem = fem_problem("cook_membrane", order)

dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
    nodal_coords, volume_conn = problem.get_mesh_properties()

disp_node, disp_val, pres_surf, surf_traction_vector \
    = problem.get_boundary_conditions()

quad_rule_3D, shape_func_tetra = problem.get_3D_basis_functions()
quad_rule_2D, shape_func_triangle = problem.get_2D_basis_functions()


print("Number of elements: ", num_elem)

params = np.array([200, 0.3])

eq_num, num_free_dof, num_pres_dof \
    = initialize_equation(num_nodes, dof_node, disp_node)

print("Number of free degrees of freedom: ", num_free_dof)

tol = 5e-12
max_iters = 10

UUR = solve_fem_newton(num_pres_dof, num_free_dof, num_elem, num_nodes_elem,
                       dof_node, num_nodes_surf, surf_traction_vector, params,
                       disp_node, disp_val, eq_num, volume_conn, nodal_coords,
                       pres_surf, quad_rule_3D, shape_func_tetra, quad_rule_2D,
                       shape_func_triangle, tol, max_iters)

print(UUR)

#problem.save_data("cook_membrane", {"displacement_field": UUR})
