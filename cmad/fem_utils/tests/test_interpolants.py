from cmad.fem_utils import interpolants
from cmad.fem_utils import quadrature_rule
import unittest
import jax.numpy as np

class TestInterpolants(unittest.TestCase):

    def test_1D_shape_functions(self):
        for i in range(2, 8):
            test_points = quadrature_rule.create_quadrature_rule_1D(i).xigauss
            shape_func_1D = interpolants.shape1d(test_points)
            shape = shape_func_1D.values
            dshape = shape_func_1D.gradients

            assert np.allclose(np.sum(shape, axis = 1), np.ones(len(shape)))
            assert np.allclose(np.sum(dshape, axis = 1), np.zeros(len(shape)))

    def test_triangle_shape_functions(self):
        for i in range(2, 7):
            test_points = quadrature_rule.create_quadrature_rule_on_triangle(i).xigauss
            shape_func_triangle = interpolants.shape_triangle(test_points)
            shape = shape_func_triangle.values
            dshape = shape_func_triangle.gradients

            assert np.allclose(np.sum(shape, axis = 1), np.ones(len(shape)))

            for j in range(len(dshape)):
                gradient = dshape[j]
                assert np.allclose(np.sum(gradient, axis = 1), np.zeros(len(gradient)))

    def test_quad_shape_functions(self):
        for i in range(2, 6):
            test_points = quadrature_rule.create_quadrature_rule_on_quad(i).xigauss
            shape_func_quad = interpolants.shape_quad(test_points)
            shape = shape_func_quad.values
            dshape = shape_func_quad.gradients

            assert np.allclose(np.sum(shape, axis = 1), np.ones(len(shape)))

            for j in range(len(dshape)):
                gradient = dshape[j]
                assert np.allclose(np.sum(gradient, axis = 1), np.zeros(len(gradient)))

    def test_tetrahedron_shape_functions(self):
        for i in range(2, 7):
            test_points = quadrature_rule.create_quadrature_rule_on_tetrahedron(i).xigauss
            shape_func_tetra = interpolants.shape_tetrahedron(test_points)
            shape = shape_func_tetra.values
            dshape = shape_func_tetra.gradients

            assert np.allclose(np.sum(shape, axis = 1), np.ones(len(shape)))

            for j in range(len(dshape)):
                gradient = dshape[j]
                assert np.allclose(np.sum(gradient, axis = 1), np.zeros(len(gradient)))

if __name__ == "__main__":
    Interpolants_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestInterpolants)
    unittest.TextTestRunner(verbosity=2).run(Interpolants_test_suite)