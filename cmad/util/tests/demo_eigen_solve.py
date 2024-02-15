import numpy as np

import jax.numpy as jnp

from jax import jit
from functools import partial
from scipy.stats import special_ortho_group, uniform

from cmad.util.numpy_eigen_decomposition \
    import compute_eigen_decomposition as np_eigen_decomp
from cmad.util.jax_eigen_decomposition \
    import compute_eigen_decomposition as jax_eigen_decomp
from jax.numpy.linalg import eigh as built_in_jax_eigen_decomp


def generate_test_matrix(test_case, distribution, perturb=1.,
                         test_eigenvalues_scale=5.):

    random_numbers = distribution.rvs(3)
    if test_case == "diagonal":
        Q = np.eye(3)
    else:
        Q = special_ortho_group.rvs(3)

    if test_case == "diagonal" or test_case == "completely random":
        exact_eigenvalues = test_eigenvalues_scale * random_numbers
    elif test_case == "two nearly identical":
        close_eigenvalue = test_eigenvalues_scale * random_numbers[1]
        exact_eigenvalues = np.array([
            test_eigenvalues_scale * random_numbers[0],
            close_eigenvalue,
            close_eigenvalue + perturb * random_numbers[2]])
    elif test_case == "three nearly identical":
        close_eigenvalue = test_eigenvalues_scale * random_numbers[0]
        exact_eigenvalues = np.array([close_eigenvalue,
                                      close_eigenvalue + perturb * random_numbers[1],
                                      close_eigenvalue + perturb * random_numbers[2]])
    else:
        raise NotImplementedError("test case not supported")

    return Q * exact_eigenvalues @ Q.T


def compute_eigen_decomposition_error(A, eigenfun):
    eigenvalues, eigenvectors = eigenfun(A)
    Z = eigenvectors * eigenvalues @ eigenvectors.T

    max_component_diff = np.max(np.abs(np.asarray(A - Z)))
    relative_max_component_diff = max_component_diff / np.linalg.norm(A)

    return relative_max_component_diff


np.random.seed(22)
uniform_dist = uniform(-1, 2)  # U[-1, 1]
num_matrices = int(1e4)
error_tol = 1e-14
test_eigenvalues_scale = 5.
perturbations = np.r_[np.logspace(0, -15, 16), 0.]

test_cases = ["diagonal", "completely random", "two nearly identical",
              "three nearly identical"]

built_in_np_eigen_error = partial(compute_eigen_decomposition_error,
                                  eigenfun=np.linalg.eigh)
np_eigen_error = partial(compute_eigen_decomposition_error,
                         eigenfun=np_eigen_decomp)
built_in_jax_eigen_error = partial(compute_eigen_decomposition_error,
                                   eigenfun=jit(built_in_jax_eigen_decomp))
jax_eigen_error = partial(compute_eigen_decomposition_error,
                          eigenfun=jit(jax_eigen_decomp))
error_funs = [
    built_in_np_eigen_error,
    np_eigen_error,
    built_in_jax_eigen_error,
    jax_eigen_error]

error = np.zeros((num_matrices, len(perturbations), len(test_cases),
                  len(error_funs)))

for test_case_idx, test_case in enumerate(test_cases):
    if test_case == "diagonal" or test_case == "completely random":
        perturbs = np.ones(1)
    else:
        perturbs = perturbations
    for perturb_idx, perturb in enumerate(perturbs):
        for ii in range(num_matrices):
            A = generate_test_matrix(test_case, uniform_dist, perturb,
                                     test_eigenvalues_scale)
            for method_idx, error_fun in enumerate(error_funs):
                error[ii, perturb_idx, test_case_idx,
                      method_idx] = error_fun(A)
