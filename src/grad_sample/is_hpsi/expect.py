from typing import Optional
from functools import partial

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.stats import Stats
from netket.utils import mpi
from netket.utils.types import PyTree

from netket.vqs import MCState, expect, expect_and_grad, expect_and_forces
from netket.vqs.mc.common import force_to_grad
from netket import stats as nkstats

from netket_pro.utils import make_logpsi_U_afun, make_logpsi_sum_afun

from .operator import IS_Operator

from grad_sample.is_hpsi.is_utils import _prepare_H

from jax.tree_util import tree_map

from netket.stats import subtract_mean
from netket.jax._jacobian.logic import _multiply_by_pdf

@expect.dispatch
def expect_IS_Operator_is(
    vstate: MCState, op: IS_Operator, chunk_size: int | None
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    apply_fun, variables = _prepare_H(vstate._apply_fun, vstate.variables, op)

    Hpsi_samples = vstate.samples_distribution(
        apply_fun, variables=variables, resample_fraction=op.resample_fraction
    )
    print("calling IS expect function")
    return expect_grad_is(
        vstate._apply_fun,
        vstate.parameters,
        apply_fun,
        variables,
        vstate.model_state,
        op,
        Hpsi_samples,
        return_grad=False,
        chunk_size=chunk_size,
    )

@expect_and_grad.dispatch
def expect_and_grad_default_formula(
    vstate: MCState,
    O: IS_Operator,
    chunk_size: int | None,
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:

    if use_covariance is None:
        use_covariance = O.operator.is_hermitian

    if use_covariance:
        # Implementation of expect_and_grad for `use_covariance == True` (due to the Literal[True]
        # type in the signature).` This case is equivalent to the composition of the
        # `expect_and_forces` and `force_to_grad` functions.
        # return expect_and_grad_from_covariance(vstate, Ô, *args, mutable=mutable)
        O_exp, O_grad = expect_and_forces(vstate, O, chunk_size, *args, mutable=mutable)
        O_grad = force_to_grad(O_grad, vstate.parameters)
        return O_exp, O_grad
    else:
        raise NotImplementedError
        # return expect_and_grad_nonhermitian(
        #    vstate, Ô, chunk_size, *args, mutable=mutable
        # )


@expect_and_forces.dispatch
def expect_and_forces_is(
    vstate: MCState, op: IS_Operator, chunk_size: int | None, *, mutable
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    apply_fun, variables = _prepare_H(vstate._apply_fun, vstate.variables, op)

    Hpsi_samples = vstate.samples_distribution(
        apply_fun, variables=variables, resample_fraction=op.resample_fraction
    )

    return expect_grad_is(
        vstate._apply_fun,
        vstate.parameters,
        apply_fun,
        variables,
        vstate.model_state,
        op,
        Hpsi_samples,
        return_grad=True,
        chunk_size=chunk_size,
    )
    # return expect_grad_is(
    #     vstate._apply_fun,
    #     vstate.parameters,
    #     vstate._apply_fun,
    #     vstate.parameters,
    #     vstate.model_state,
    #     op,
    #     vstate.samples,
    #     return_grad=True,
    #     chunk_size=chunk_size,
    # )

    


@partial(jax.jit, static_argnames=("log_psi", "log_Hpsi", "return_grad", "chunk_size"))
def expect_grad_is(
    log_psi, parameters, log_Hpsi, Hpsi_vars, model_state, operator, sigma, return_grad, chunk_size
):
    O = operator.operator
    parameters = {"params": parameters}
    sigma = sigma.reshape(sigma.shape[0]*sigma.shape[1], -1)
    # Compute standard Expectation value
    log_psi_sigma = nkjax.apply_chunked(lambda x: log_psi(parameters, x), chunk_size=chunk_size)(sigma)

    log_Hpsi_sigma = nkjax.apply_chunked(lambda x: log_Hpsi(Hpsi_vars, x), chunk_size=chunk_size)(sigma)
    # log_Hpsi_sigma = nkjax.apply_chunked(lambda x: log_psi(parameters, x), chunk_size=chunk_size)(sigma)
    # compute the expectation value of O 
    eta, etap_mels = O.get_conn_padded(sigma)
    _eta = eta.reshape(-1, eta.shape[-1])
    log_psi_eta = nkjax.apply_chunked(lambda x: log_psi(parameters, x), chunk_size=chunk_size)(
        _eta
    )
    # del _eta_Hpsi
    log_psi_eta = log_psi_eta.reshape(eta.shape[:-1])
    w_is_sigma = jnp.abs(jnp.exp(log_psi_sigma - log_Hpsi_sigma))**2

    Z_ratio = 1/nkstats.mean(w_is_sigma)

    # IS estimate of mean of O
    # O_loc_sigma = jnp.exp(log_psi_sigma)*jnp.sum(etap_mels * jnp.exp(log_psi_eta), axis=-1)/jnp.abs(jnp.exp(log_Hpsi_sigma))**2
    # O_mean = nkstats.mean(O_loc_sigma)

    O_loc_sigma = w_is_sigma*jnp.sum(etap_mels * jnp.exp(log_psi_eta- jnp.expand_dims(log_psi_sigma, axis=-1)), axis=-1)
    O_mean = nkstats.mean(O_loc_sigma)

    # jax.debug.print("{x}", x=O_mean)

    # estimate normalization constant for centering, ie mean of O^2, or inverse of average of importance weights
    # eta2, eta2_mels = operator._operator_squared.get_conn_padded(sigma)
    # log_psi_eta2 = nkjax.apply_chunked(lambda x: log_psi(parameters, x), chunk_size=chunk_size)(
    #     eta2
    # )
    # O2_loc = jnp.sum(
    #     etap2_mels * jnp.exp(log_psi_eta2 - log_psi_sigma), axis=-1
    # )
    # O2_estim = nkstats.mean(O2_loc_eta)

    if not return_grad:

        # op_loc_sigma_R = jnp.sum(
        #     1/(eta_p_mels
        #     * jnp.exp(log_psi_eta - log_psi_sigma)).conj(),
        #     axis=-1,
        # )

        op_loc = O_loc_sigma * Z_ratio

        return nkstats.statistics(op_loc)

    else:
        def dagger_pytree(jac_pytree):
            return tree_map(lambda x: x.conj().T, jac_pytree)

        def vjp_pytree(jac_pytree, vector):
            return tree_map(lambda jac_block: jnp.einsum("...i,i->...", jac_block, vector), jac_pytree)
        # compute actual jacobian instead of vjp
        # jacobian_pytree = nkjax.jacobian(
        #     lambda w, sigma: jnp.exp(log_psi(w, sigma)),
        #     parameters["params"],
        #     sigma,
        #     model_state,
        #     mode = "holomorphic",
        #     chunk_size=chunk_size,
        #     dense=False
        # )
        jacobian_pytree = nkjax.jacobian(
            lambda w, sigma: log_psi(w, sigma),
            parameters["params"],
            sigma,
            model_state,
            mode = "holomorphic",
            chunk_size=chunk_size,
            dense=False
        )

        # jac_ct = jacobian_dense.conj().T - Z_ratio*jnp.exp(log_psi_sigma).conj()*nkstats.mean(jnp.exp(-2*jnp.abs(log_Hpsi_sigma))[None,:]*jacobian_dense.conj().T)

        # n_samples = log_phi_sigma.size * mpi.n_nodes

        # gradient
        # op_loc_sigma_R = jnp.sum(
        #     sigma_p_mels
        #     * jnp.exp(log_psi_sigma_p + jnp.expand_dims(log_psi_sigma.conj() - 2 * log_phi_sigma.real, -1)),
        #     axis=-1,
        # )
       
        # op_loc = jnp.sum(etap_mels * jnp.exp(log_psi_eta), axis=-1)/jnp.abs(jnp.exp(log_Hpsi_sigma))**2
        # op_loc_c = op_loc - (jnp.exp(log_psi_sigma)/jnp.abs(jnp.exp(log_Hpsi_sigma))**2)* Z_ratio * O_mean

        # jac_ct_pytree =  tree_map(
        #     lambda x: x - Z_ratio * jax.lax.broadcast_in_dim(jnp.exp(log_psi_sigma).conj(), x.T.shape, (0,)).T * jnp.expand_dims(nkstats.mean(jax.lax.broadcast_in_dim(1/jnp.abs(jnp.exp(log_Hpsi_sigma))**2, x.T.shape, (0,)).T  * x, axis = -1), axis=-1),
        #     dagger_pytree(jacobian_pytree))

        # grad_pytree = vjp_pytree(jac_ct_pytree, op_loc_c)
        # log version to check
        jacobians_avg = jax.tree_util.tree_map(
                partial(jnp.mean, axis=0, keepdims=True), _multiply_by_pdf(jacobian_pytree, w_is_sigma)
        )
        jacobians = jax.tree_util.tree_map(
            lambda x, y: x - Z_ratio*y, jacobian_pytree, jacobians_avg
        )

        # jacobians = jacobian_pytree = nkjax.jacobian(
        #     lambda w, sigma: log_psi(w, sigma),
        #     parameters["params"],
        #     sigma,
        #     model_state,
        #     mode = "holomorphic",
        #     chunk_size=chunk_size,
        #     dense=False,
        #     center=True
        # )
        # jax.debug.print("{x}", x=tree_map(lambda x,y: jnp.abs(x/y), jacobians_l, jacobians ))
        op_loc = jnp.sum(etap_mels * jnp.exp(log_psi_eta - jnp.expand_dims(log_psi_sigma,-1)), axis=-1)
        op_loc_c = w_is_sigma * (op_loc - Z_ratio * O_mean)

        # jac_ct_pytree =  tree_map(
        # lambda x: x - Z_ratio * jnp.expand_dims(nkstats.mean(jax.lax.broadcast_in_dim(w_is_sigma, x.T.shape, (0,)).T  * x, axis = -1), axis=-1),
        # dagger_pytree(jacobian_pytree))

        grad_pytree = vjp_pytree(dagger_pytree(jacobians), op_loc_c)
        # Final gradient (normalize by sample size)
        grad = tree_map(lambda g: Z_ratio * g.T / log_psi_sigma.size, grad_pytree)

        return nkstats.statistics(w_is_sigma*op_loc*Z_ratio), grad

# @partial(jax.jit, static_argnames=("logψ", "return_grad", "chunk_size"))
# def expect_grad_is(
#     log_psi, parameters, model_state, operator, sigma_phi, eta_psi, return_grad, chunk_size
# ):
#     O = operator.operator
#     psi_vars = {"params": parameters, **model_state}
#     log_phi, phi_vars = _prepare(log_psi, psi_vars, operator)

#     # Compute standard Expectation value
#     log_psi_eta = nkjax.apply_chunked(lambda x: log_psi(psi_vars, x), chunk_size=chunk_size)(eta_psi)

#     # Compute standard Expectation value
#     etap_psi, etap_mels = O.get_conn_padded(eta_psi)
#     _etap_psi = etap_psi.reshape(-1, etap_psi.shape[-1])
#     log_psi_etap = nkjax.apply_chunked(lambda x: log_psi(psi_vars, x), chunk_size=chunk_size)(
#         _etap_psi
#     )
#     del _etap_psi
#     log_psi_etap = log_psi_etap.reshape(etap_psi.shape[:-1])

#     O_loc_eta = jnp.sum(etap_mels * jnp.exp(log_psi_etap - jnp.expand_dims(log_psi_eta, -1)), axis=-1)
#     O_eta = nkstats.mean(O_loc_eta)

#     if operator.square_fast:
#         # use fast, approximate, bad variance formula
#         O2_eta = nkstats.mean(jnp.abs(O_loc_eta) ** 2)
#     else:
#         # use exact expression
#         etap2_psi, etap2_mels = operator._operator_squared.get_conn_padded(eta_psi)
#         log_psi_etap2 = log_psi(psi_vars, etap2_psi)
#         O2_loc_eta = jnp.sum(
#             etap2_mels * jnp.exp(log_psi_etap2 - jnp.expand_dims(log_psi_eta, -1)), axis=-1
#         )
#         O2_eta = nkstats.mean(O2_loc_eta)

#     K = O2_eta

#     # End of compute standard expectation value
#     log_phi_sigma = nkjax.apply_chunked(lambda x: log_phi(phi_vars, x), chunk_size=chunk_size)(sigma_phi)

#     # Compute standard Expectation value
#     sigmap_phi, sigmap_mels = O.get_conn_padded(sigma_psi)
#     _sigmap_phi = sigmap_phi.reshape(-1, sigmap_phi.shape[-1])
#     log_psi_sigmap = nkjax.apply_chunked(lambda x: log_psi(psi_vars, x), chunk_size=chunk_size)(
#         _sigmap_phi
#     )
#     del _sigmap_phi
#     log_phi_sigmap = log_phi_sigmap.reshape(sigmap_phi.shape[:-1])

#     if not return_grad:
#         log_phi_sigma = nkjax.apply_chunked(lambda x: log_psi(psi_vars, x), chunk_size=chunk_size)(
#             sigma_phi
#         )

#         op_loc_sigma_R = jnp.sum(
#             sigma_p_mels
#             * jnp.exp(log_psi_sigma_p + jnp.expand_dims(log_psi_sigma.conj() - 2 * log_phi_sigma.real, -1)),
#             axis=-1,
#         )

#         op_loc = op_loc_sigma_R * K

#         return nkstats.statistics(op_loc)

#     else:
#         log_psi_sigma = nkjax.apply_chunked(lambda x: log_psi(psi_vars, x), chunk_size=chunk_size)(
#             sigma_phi
#         )

#         vjp_fun = nkjax.vjp_chunked(
#             lambda w, sigma: log_psi({"params": w, **model_state}, sigma),
#             parameters,
#             sigma_phi,
#             conjugate=True,
#             chunk_size=chunk_size,
#             chunk_argnums=1,
#             nondiff_argnums=1,
#         )

#         n_samples = log_phi_sigma.size * mpi.n_nodes

#         # gradient
#         op_loc_sigma_R = jnp.sum(
#             sigma_p_mels
#             * jnp.exp(log_psi_sigma_p + jnp.expand_dims(log_psi_sigma.conj() - 2 * log_phi_sigma.real, -1)),
#             axis=-1,
#         )

#         op_loc = op_loc_sigma_R * K
#         R = jnp.exp(2 * (log_psi_sigma.real - log_phi_sigma.real))
#         R_O = nkstats.mean(op_loc) * R
#         vec = (op_loc_sigma_R - R_O) * K / n_samples

#         (O_grad,) = vjp_fun(jnp.conjugate(vec))

#         return nkstats.statistics(op_loc), jax.tree_util.tree_map(
#             lambda x: mpi.mpi_sum_jax(x)[0], O_grad
#         )
