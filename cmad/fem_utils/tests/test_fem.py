import jax.numpy as np
from cmad.fem_utils.fem_problem import fem_problem
from cmad.fem_utils.fem_functions import (initialize_equation,
                                          assemble_module, solve_module)
import unittest

class TestFEM(unittest.TestCase):

    def test_patch_form_B(self):

        order = 3
        problem = fem_problem("patch_B", order)

        DOF_NODE, NUM_NODES, NUM_NODES_ELE, NUM_ELE, NUM_NODES_SURF, \
            nodal_coords, volume_conn = problem.get_mesh_properties()
        
        disp_node, disp_val, pres_surf, SURF_TRACTION_VECTOR \
            = problem.get_boundary_conditions()
        
        quad_rule_3D, shape_func_tetra = problem.get_3D_basis_functions()
        quad_rule_2D, shape_func_triangle = problem.get_2D_basis_functions()

        E = 200
        nu = 0.3

        eq_num, NUM_FREE_DOF, NUM_PRES_DOF\
            = initialize_equation(NUM_NODES, DOF_NODE, disp_node)

        KPP, KPF, KFF, KFP, PF, PP, UP \
            = assemble_module(NUM_PRES_DOF, NUM_FREE_DOF, NUM_ELE, NUM_NODES_ELE,
                            DOF_NODE, NUM_NODES_SURF,SURF_TRACTION_VECTOR, E, nu,
                            disp_node, disp_val, eq_num, volume_conn, nodal_coords, 
                            pres_surf, quad_rule_3D, shape_func_tetra, quad_rule_2D,
                            shape_func_triangle)

        UUR, UF, R = solve_module(KPP, KPF, KFF, KFP, PP, PF, UP, eq_num)

        assert np.allclose(UUR, problem.UUR_true)

if __name__ == "__main__":
    fem_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFEM)
    unittest.TextTestRunner(verbosity=2).run(fem_test_suite)