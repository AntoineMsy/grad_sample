# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from functools import partial
import math

import jax
from jax import numpy as jnp

from netket.utils import mpi
from netket import stats as nkstats
import netket.jax as nkjax


from netket.optimizer.qgt.qgt_jacobian_dense import QGTJacobianDenseT
from netket.optimizer.qgt.qgt_jacobian_common import (
    to_shift_offset,
    rescale,
)


@partial(jax.jit, static_argnums=(0, 1, 5))
def importance_factor_sq(logψ, logφ, varsψ, varsφ, σ, chunk_size):
    def _fun(x):
        return jnp.exp(2 * (logψ(varsψ, x).real - logφ(varsφ, x).real))

    return nkjax.apply_chunked(_fun, in_axes=(0), chunk_size=chunk_size)(σ)


def QGTJacobianDefaultConstructorIS(
    apply_fun,
    parameters,
    model_state,
    samples,
    log_is_fun,
    is_vars,
    *,
    importance_operator=None,
    mode: Optional[str] = None,
    holomorphic: Optional[bool] = None,
    diag_shift=0.0,
    diag_scale=None,
    chunk_size=None,
    **kwargs,
) -> "QGTJacobianDenseT":
    r"""Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a dense matrix.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Numerical estimates of the QGT are usually ill-conditioned and require
    regularisation. The standard approach is to add a positive constant to the diagonal;
    alternatively, Becca and Sorella (2017) propose scaling this offset with the
    diagonal entry itself. NetKet allows using both in tandem:

    .. math::

        S_{ii} \\mapsto S_{ii} + \\epsilon_1 S_{ii} + \\epsilon_2;

    :math:`\\epsilon_{1,2}` are specified using `diag_scale` and `diag_shift`,
    respectively.

    Args:
        vstate: The variational state
        mode: "real", "complex" or "holomorphic": specifies the implementation
              used to compute the jacobian. "real" discards the imaginary part
              of the output of the model. "complex" splits the real and imaginary
              part of the parameters and output. It works also for non holomorphic
              models. holomorphic works for any function assuming it's holomorphic
              or real valued.
        holomorphic: a flag to indicate that the function is holomorphic.
        diag_scale: Fractional shift :math:`\\epsilon_1` added to diagonal entries (see above).
        diag_shift: Constant shift :math:`\\epsilon_2` added to diagonal entries (see above).
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).

    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    if importance_operator is None:
        raise ValueError("Must specify the importance_operator")
    
    if mode is None:
        mode = nkjax.jacobian_default_mode(
            apply_fun,
            parameters,
            model_state,
            samples,
            holomorphic=holomorphic,
        )

    if samples.ndim >= 3:
        # use jit so that we can do it on global shared array
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)
    sqrt_n_samp = jnp.sqrt(samples.shape[0] * mpi.n_nodes)  # maintain weak type

    """
    J_ψ_η = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        samples_ψ,
        vstate.model_state,
        mode=mode,
        pdf=None,
        chunk_size=chunk_size,
        dense=True,
        center=False,
        _sqrt_rescale=False,
    )
    """

    jac_dense = nkjax.jacobian(
        apply_fun,
        parameters,
        samples,
        model_state,
        mode=mode,
        chunk_size=chunk_size,
        dense=True,
        center=False,
        _sqrt_rescale=False,
    )
    op = importance_operator.operator
    log_psi_sigma = nkjax.apply_chunked(lambda x: apply_fun({"params":parameters}, x), chunk_size=chunk_size)(samples)

    log_Hpsi_sigma = nkjax.apply_chunked(lambda x: log_is_fun(is_vars, x), chunk_size=chunk_size)(samples)
    w_is_sigma = jnp.abs(jnp.exp(log_psi_sigma - log_Hpsi_sigma))**2
    Z_ratio = 1/jnp.mean(w_is_sigma)

    jacobians = jnp.sqrt(Z_ratio*w_is_sigma)[:,None]/sqrt_n_samp * (jac_dense - Z_ratio*jnp.mean(w_is_sigma[:,None]*jac_dense, axis=0))

    shift, offset = to_shift_offset(diag_shift, diag_scale)

    if offset is not None:
        ndims = 1 if mode != "complex" else 2
        jacobians, scale = rescale(jacobians, offset, ndims=ndims)
    else:
        scale = None

    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    return QGTJacobianDenseT(
        O=jacobians,
        scale=scale,
        mode=mode,
        _params_structure=pars_struct,
        diag_shift=shift,
        **kwargs,
    )

def QGTJacobianDenseImportanceSampling(
    vstate=None,
    *,
    importance_operator=None,
    mode: Optional[str] = None,
    holomorphic: Optional[bool] = None,
    diag_shift=0.0,
    diag_scale=None,
    chunk_size=None,
    **kwargs,
) -> "QGTJacobianDenseT":
    r"""Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a dense matrix.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Numerical estimates of the QGT are usually ill-conditioned and require
    regularisation. The standard approach is to add a positive constant to the diagonal;
    alternatively, Becca and Sorella (2017) propose scaling this offset with the
    diagonal entry itself. NetKet allows using both in tandem:

    .. math::

        S_{ii} \\mapsto S_{ii} + \\epsilon_1 S_{ii} + \\epsilon_2;

    :math:`\\epsilon_{1,2}` are specified using `diag_scale` and `diag_shift`,
    respectively.

    Args:
        vstate: The variational state
        mode: "real", "complex" or "holomorphic": specifies the implementation
              used to compute the jacobian. "real" discards the imaginary part
              of the output of the model. "complex" splits the real and imaginary
              part of the parameters and output. It works also for non holomorphic
              models. holomorphic works for any function assuming it's holomorphic
              or real valued.
        holomorphic: a flag to indicate that the function is holomorphic.
        diag_scale: Fractional shift :math:`\\epsilon_1` added to diagonal entries (see above).
        diag_shift: Constant shift :math:`\\epsilon_2` added to diagonal entries (see above).
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).

    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    if vstate is None:
        return partial(
            QGTJacobianDenseImportanceSampling,
            importance_operator=importance_operator,
            mode=mode,
            holomorphic=holomorphic,
            chunk_size=chunk_size,
            diag_shift=diag_shift,
            diag_scale=diag_scale,
            **kwargs,
        )

    if importance_operator is None:
        raise ValueError("Must specify the importance_operator")

    if chunk_size is None and hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size

    log_Hpsi, Hpsi_vars = importance_operator.get_log_importance(vstate)

    sigma = vstate.samples_distribution(
        log_Hpsi,
        variables=Hpsi_vars,
    )
    if mode is None:
        mode = nkjax.jacobian_default_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            sigma,
            holomorphic=holomorphic,
        )

    if sigma.ndim >= 3:
        # use jit so that we can do it on global shared array
        sigma = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(sigma, 0, 2)
    sqrt_n_samp = jnp.sqrt(sigma.shape[0] * mpi.n_nodes)  # maintain weak type

    """
    J_ψ_η = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        samples_ψ,
        vstate.model_state,
        mode=mode,
        pdf=None,
        chunk_size=chunk_size,
        dense=True,
        center=False,
        _sqrt_rescale=False,
    )
    """

    jac_dense = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        sigma,
        vstate.model_state,
        mode=mode,
        chunk_size=chunk_size,
        dense=True,
        center=False,
        _sqrt_rescale=False,
    )
    op = importance_operator.operator
    log_psi_sigma = nkjax.apply_chunked(lambda x: vstate.model.apply({"params":vstate.parameters}, x), chunk_size=chunk_size)(sigma)

    log_Hpsi_sigma = nkjax.apply_chunked(lambda x: log_Hpsi(Hpsi_vars, x), chunk_size=chunk_size)(sigma)
    w_is_sigma = jnp.abs(jnp.exp(log_psi_sigma - log_Hpsi_sigma))**2
    Z_ratio = 1/jnp.mean(w_is_sigma)

    jacobians = jnp.sqrt(Z_ratio*w_is_sigma)[:,None]/sqrt_n_samp * (jac_dense - Z_ratio*jnp.mean(w_is_sigma[:,None]*jac_dense, axis=0))

    shift, offset = to_shift_offset(diag_shift, diag_scale)

    if offset is not None:
        ndims = 1 if mode != "complex" else 2
        jacobians, scale = rescale(jacobians, offset, ndims=ndims)
    else:
        scale = None

    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters
    )

    return QGTJacobianDenseT(
        O=jacobians,
        scale=scale,
        mode=mode,
        _params_structure=pars_struct,
        diag_shift=shift,
        **kwargs,
    )


@partial(jax.jit, static_argnames=("logψ", "chunk_size"))
def compute_importance_factor(logψ, ψ_vars, η_ψ, operator, chunk_size):
    O = operator.operator
    ϵ = operator.epsilon

    logψ_η = nkjax.apply_chunked(lambda x: logψ(ψ_vars, x), chunk_size=chunk_size)(η_ψ)

    # Compute standard Expectation value
    ηp_ψ, ηp_mels = O.get_conn_padded(η_ψ)
    _ηp_ψ = ηp_ψ.reshape(-1, ηp_ψ.shape[-1])
    logψ_ηp = nkjax.apply_chunked(lambda x: logψ(ψ_vars, x), chunk_size=chunk_size)(
        _ηp_ψ
    )
    del _ηp_ψ
    logψ_ηp = logψ_ηp.reshape(ηp_ψ.shape[:-1])

    O_loc_η = jnp.sum(ηp_mels * jnp.exp(logψ_ηp - jnp.expand_dims(logψ_η, -1)), axis=-1)
    O_η = nkstats.mean(O_loc_η)

    if operator.square_fast:
        # use fast, approximate, bad variance formula
        O2_η = nkstats.mean(jnp.abs(O_loc_η) ** 2)
    else:
        # use exact expression
        ηp2_ψ, ηp2_mels = operator._operator_squared.get_conn_padded(η_ψ)
        logψ_ηp2 = logψ(ψ_vars, ηp2_ψ)
        O2_loc_η = jnp.sum(
            ηp2_mels * jnp.exp(logψ_ηp2 - jnp.expand_dims(logψ_η, -1)), axis=-1
        )
        O2_η = nkstats.mean(O2_loc_η)

    K = (1 - ϵ) ** 2 + 2 * (1 - ϵ) * ϵ * O_η + ϵ**2 * O2_η
    # End of compute standard expectation value
    return K.real


@partial(jax.jit, static_argnames=("logψ", "chunk_size"))
def compute_importance_factor_full(logψ, ψ_vars, η_ψ, operator, chunk_size):
    O = operator.operator
    ϵ = operator.epsilon

    # Compute standard Expectation value
    logψ_η = nkjax.apply_chunked(lambda x: logψ(ψ_vars, x), chunk_size=chunk_size)(η_ψ)

    # Compute standard Expectation value
    ηp_ψ, ηp_mels = O.get_conn_padded(η_ψ)
    _ηp_ψ = ηp_ψ.reshape(-1, ηp_ψ.shape[-1])
    logψ_ηp = nkjax.apply_chunked(lambda x: logψ(ψ_vars, x), chunk_size=chunk_size)(
        _ηp_ψ
    )
    del _ηp_ψ
    logψ_ηp = logψ_ηp.reshape(ηp_ψ.shape[:-1])
    ψ_η = jnp.exp(logψ_η)

    O_loc_η = jnp.sum(ηp_mels * jnp.exp(logψ_ηp - jnp.expand_dims(logψ_η, -1)), axis=-1)
    P_η = jnp.abs(ψ_η) ** 2 / jnp.linalg.norm(ψ_η) ** 2
    O_η = P_η @ O_loc_η

    if operator.square_fast:
        # use fast, approximate, bad variance formula
        O2_loc_η = jnp.abs(O_loc_η) ** 2
        O2_η = P_η @ O2_loc_η
    else:
        # use exact expression
        ηp2_ψ, ηp2_mels = operator._operator_squared.get_conn_padded(η_ψ)
        logψ_ηp2 = logψ(ψ_vars, ηp2_ψ)
        O2_loc_η = jnp.sum(
            ηp2_mels * jnp.exp(logψ_ηp2 - jnp.expand_dims(logψ_η, -1)), axis=-1
        )
        O2_η = P_η @ O2_loc_η

    K = (1 - ϵ) ** 2 + 2 * (1 - ϵ) * ϵ * O_η + ϵ**2 * O2_η

    return K.real
