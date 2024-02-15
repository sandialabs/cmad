import jax.numpy as jnp
from jax.lax import cond


def elastic_path(yield_fun, C_elastic, C_plastic, tol):
    return C_elastic


def plastic_path(yield_fun, C_elastic, C_plastic, tol):
    return C_plastic


def cond_residual(f, C_e, C_p, tol):
    def inner_cond_residual(f, C_e, C_p, tol): return cond(
        jnp.abs(f) < tol, plastic_path, elastic_path, f, C_e, C_p, tol)

    def outer_cond_residual(f, C_e, C_p, tol): return cond(
        f > tol, plastic_path, inner_cond_residual, f, C_e, C_p, tol)
    return outer_cond_residual(f, C_e, C_p, tol)
