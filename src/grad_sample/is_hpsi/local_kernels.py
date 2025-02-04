from typing import Optional
from functools import partial

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.stats import Stats
from netket.utils import mpi
from netket.utils.types import PyTree

from netket.vqs import MCState
from netket import stats as nkstats
from .operator import IS_Operator
from jax.tree_util import tree_map


from netket.utils.dispatch import dispatch
from collections.abc import Callable
from functools import partial
from netket.utils.types import PyTree, Array

from netket.vqs.mc import (
    kernels,
    check_hilbert,
    get_local_kernel_arguments,
    get_local_kernel,
)
from netket.operator import DiscreteJaxOperator

@dispatch
def get_local_kernel_arguments(vstate: MCState, op: IS_Operator):  # noqa: F811
    check_hilbert(vstate.hilbert, op.hilbert)
    apply_fun, variables = op.get_log_importance(vstate)

    sigma = vstate.samples_distribution(
        apply_fun, variables=variables, resample_fraction=op.resample_fraction
    )
    
    sigmap, mels = op.operator.get_conn_padded(sigma)
    # returns samples, jax operator associated to the is operatorm and the IS apply function
    return sigma, op.operator, apply_fun, variables

@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCState, Ô: IS_Operator, chunk_size: int
):  # noqa: F811
    return local_value_kernel_jax_chunked

def local_value_kernel_jax_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    log_is_fun: Callable,
    is_vars: PyTree,
    *,
    chunk_size: int | None = None,
):
    """
    local_value kernel for MCState and jaxcoompatible operators
    """
    if chunk_size >= O.max_conn_size:
        local_value_kernel = lambda s: local_value_kernel_jax(logpsi, pars, s, O, log_is_fun, is_vars)
        local_value_chunked = nkjax.apply_chunked(
            local_value_kernel,
            in_axes=0,
            chunk_size=max(1, chunk_size // O.max_conn_size),
        )
    else:
        local_value_chunked = lambda s: local_value_kernel_jax_conn_chunked(
            logpsi, pars, s, O, log_is_fun, is_vars, chunk_size
        )

    return local_value_chunked(σ)

def local_value_kernel_jax(
    logpsi: Callable, pars: PyTree, σ: Array, O: DiscreteJaxOperator, log_is_fun: Callable, is_vars: PyTree
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = O.get_conn_padded(σ)
    logpsi_σ = logpsi(pars, σ)
    log_is_sigma = log_is_fun(is_vars, σ) # Note: could improve computation time in some cases by making the isfunction a function of logpsi
    logpsi_σp = logpsi(pars, σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])

    w_is_sigma = jnp.abs(jnp.exp(logpsi_σ- log_is_sigma))**2
    Z_ratio = 1/nkstats.mean(w_is_sigma)
    w_is_sigma = w_is_sigma * Z_ratio #provide self normalized is weight
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1), w_is_sigma


def local_value_kernel_jax_conn_chunked(
    logpsi: Callable,
    pars: PyTree,
    σ: Array,
    O: DiscreteJaxOperator,
    log_is_fun: Callable,
    is_vars: PyTree,
    chunk_size: int,
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    apply_conn = lambda s: logpsi(pars, s)
    apply_conn = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    apply_conn_is = lambda s: log_is_fun(is_vars, s)
    apply_conn_is = nkjax.apply_chunked(apply_conn, in_axes=0, chunk_size=chunk_size)

    σp, mel = O.get_conn_padded(σ)

    logpsi_σ = apply_conn(σ)
    logpsi_σp = apply_conn(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])

    log_is_sigma = apply_conn_is(σ)
    w_is_sigma = jnp.abs(jnp.exp(logpsi_σ- log_is_sigma))**2
    Z_ratio = 1/nkstats.mean(w_is_sigma)
    w_is_sigma = w_is_sigma * Z_ratio #provide self normalized is weight
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1), w_is_sigma


# ## Chunked versions of those kernels are defined below.
# def local_value_kernel_chunked(
#     logpsi: Callable,
#     pars: PyTree,
#     σ: Array,
#     args: PyTree,
#     *,
#     chunk_size: int | None = None,
# ):
#     """
#     local_value kernel for MCState and generic operators
#     """
#     σp, mels = args

#     if jnp.ndim(σp) != 3:
#         σp = σp.reshape((σ.shape[0], -1, σ.shape[-1]))
#         mels = mels.reshape(σp.shape[:-1])

#     logpsi_chunked = nkjax.vmap_chunked(
#         partial(logpsi, pars), in_axes=0, chunk_size=chunk_size
#     )
#     N = σ.shape[-1]

#     logpsi_σ = logpsi_chunked(σ.reshape((-1, N))).reshape(σ.shape[:-1] + (1,))
#     logpsi_σp = logpsi_chunked(σp.reshape((-1, N))).reshape(σp.shape[:-1])

#     return jnp.sum(mels * jnp.exp(logpsi_σp - logpsi_σ), axis=-1)

# @batch_discrete_kernel
# def local_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
#     """
#     local_value kernel for MCState and generic operators
#     """
#     σp, mel = args
#     return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))
