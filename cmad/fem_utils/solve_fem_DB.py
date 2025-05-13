from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.fem_functions_bochev import (initialize_equation,
                                                 solve_fem_newton)
import numpy as np

order = 3
problem = fem_problem("cook_membrane", order)

dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
    nodal_coords, volume_conn = problem.get_mesh_properties()

disp_node, disp_val, pres_surf, surf_traction_vector \
    = problem.get_boundary_conditions()

quad_rule_3D, shape_func_tetra = problem.get_3D_basis_functions()
quad_rule_2D, shape_func_triangle = problem.get_2D_basis_functions()

print("Number of elements: ", num_elem)

params = np.array([200, 0.5])

eq_num_u, num_free_dof, num_pres_dof \
    = initialize_equation(num_nodes, dof_node, disp_node)

print("Number of free degrees of freedom: ", num_free_dof)

eq_num_p = np.append(np.linspace(1, num_nodes - 1, num_nodes - 1, dtype = int), -1)

tol = 5e-12
max_iters = 10

UUR, pUR = solve_fem_newton(num_pres_dof, num_free_dof, num_elem, num_nodes_elem, dof_node,
                            num_nodes_surf, surf_traction_vector, params, disp_node, disp_val,
                            eq_num_u, eq_num_p, volume_conn, nodal_coords, pres_surf, quad_rule_3D,
                            shape_func_tetra, quad_rule_2D, shape_func_triangle, tol, max_iters)

problem.save_data("cook_membrane_incompressible", {'displacement_field': UUR, 'pressure_field': pUR})