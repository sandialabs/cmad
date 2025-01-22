import numpy as np
from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.fem_functions import (initialize_equation,
                                          assemble_module, solve_module)
import unittest

class TestFEM(unittest.TestCase):

    def test_patch_form_B(self):

        order = 3
        problem = fem_problem("patch_B", order)

        dof_node, num_nodes, num_nodes_elem, num_elem, num_nodes_surf, \
            nodal_coords, volume_conn = problem.get_mesh_properties()
        
        disp_node, disp_val, pres_surf, surf_traction_vector \
            = problem.get_boundary_conditions()
        
        quad_rule_3D, shape_func_tetra = problem.get_3D_basis_functions()
        quad_rule_2D, shape_func_triangle = problem.get_2D_basis_functions()

        params = np.array([200, 0.3])

        eq_num, num_free_dof, num_pres_dof \
            = initialize_equation(num_nodes, dof_node, disp_node)

        KPP, KPF, KFF, KFP, PF, PP, UP \
            = assemble_module(num_pres_dof, num_free_dof, num_elem, num_nodes_elem,
                            dof_node, num_nodes_surf,surf_traction_vector, params,
                            disp_node, disp_val, eq_num, volume_conn, nodal_coords, 
                            pres_surf, quad_rule_3D, shape_func_tetra, quad_rule_2D,
                            shape_func_triangle)

        UUR, UF, R = solve_module(KPP, KPF, KFF, KFP, PP, PF, UP, eq_num)

        assert np.allclose(UUR, problem.UUR_true)

if __name__ == "__main__":
    fem_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFEM)
    unittest.TextTestRunner(verbosity=2).run(fem_test_suite)