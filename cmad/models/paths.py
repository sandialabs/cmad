import jax.numpy as jnp
from jax.lax import cond

from cmad.typing import JaxArray


def elastic_path(
        yield_fun: JaxArray, C_elastic: JaxArray,
        C_plastic: JaxArray, tol: float,
) -> JaxArray:
    return C_elastic


def plastic_path(
        yield_fun: JaxArray, C_elastic: JaxArray,
        C_plastic: JaxArray, tol: float,
) -> JaxArray:
    return C_plastic


def cond_residual(
        f: JaxArray, C_e: JaxArray, C_p: JaxArray, tol: float,
) -> JaxArray:
    def inner_cond_residual(f, C_e, C_p, tol): return cond(
        jnp.abs(f) < tol, plastic_path, elastic_path, f, C_e, C_p, tol)

    def outer_cond_residual(f, C_e, C_p, tol): return cond(
        f > tol, plastic_path, inner_cond_residual, f, C_e, C_p, tol)
    return outer_cond_residual(f, C_e, C_p, tol)
