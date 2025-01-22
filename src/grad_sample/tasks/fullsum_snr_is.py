from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
# from hydra.utils import call, instantiate
# import copy
import flax
import netket as nk
import netket.jax as nkjax
import jax.numpy as jnp
import jax

from tqdm import tqdm
from grad_sample.tasks.base import Problem
from grad_sample.utils.utils import cumsum, find_closest_saved_vals
from grad_sample.utils.distances import fs_dist, dot_prod

import json
from grad_sample.is_hpsi.qgt import QGTJacobianDenseImportanceSampling, QGTJacobianDefaultConstructorIS
from grad_sample.is_hpsi.operator import IS_Operator

from grad_sample.is_hpsi.expect import *
from functools import partial
from grad_sample.utils.tree_op import shape_tree, snr_tree, flatten_tree_to_array

class FullSumIS(Problem):
    def __init__(self, cfg: DictConfig):
        # instantiate parent class
        super().__init__(cfg)
        self.alpha_l = [0.0, 0.5, 1.0, 1.5, 2.0]
        # self.alpha_l = [-1.0]
        self.dimH = self.model.hi.n_states
        self.save_dir = self.output_dir + "/is_analysis"
        self.chunk_size_vmap = self.dimH // (self.dimH // self.chunk_size_vmap)

        self.out_dict = {}
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))
        E=  data["Energy"]["Mean"]["real"]
        E_err = jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs)
        
        self.out_dict["commons"] = {"save_every": self.save_every, "E_gs": self.E_gs, "E_err": E_err}
        
        # eval_s will be relative errors at every magnitude
        self.eval_s, self.error_s = find_closest_saved_vals(E_err, jnp.arange(len(E_err//10)), self.save_every, n_vals_per_scale=2)
        print(self.eval_s)
        print(self.error_s)
        self.sample_size = 9
        self.Nsample = 2**self.sample_size
        self.sampler = nk.sampler.ExactSampler(hilbert= self.model.hi)
        self.chunk_size = self.chunk_size_jac
        self.vstate = nk.vqs.MCState(sampler= self.sampler, model=self.ansatz, chunk_size= self.chunk_size, n_samples= self.Nsample, seed=0)

        self.vstate_fs = nk.vqs.FullSumState(hilbert=self.model.hi, model=self.ansatz, chunk_size= None)
        self.sr_fs = nk.optimizer.SR(solver=self.solver_fn, diag_shift=self.diag_shift, holomorphic= self.mode == "holomorphic")

        self.num_resample = 1000
        self.chunk_size_resample = 50

        self.model.H_jax._setup()

    def __call__(self):
        k=0
        for alpha in self.alpha_l :
            alpha_out = {'sample_size': self.sample_size,  
                         'n_batches': self.num_resample,
                         'mean_e': [],
                         "var_e": [],
                         'snr_e': [],
                         'snr_ng': None,
                         'snr_grad': None,
                         'fs_dist': None,
                         "rel_err": self.error_s}
            snr_e_l = []
            snr_ng_l = []
            snr_grad_l = []
            fs_dist_l = []
    
            for state_idx in tqdm(self.eval_s):   
                # load vstate with required variables
                variables = None
                with open(self.state_dir + "/%d.mpack"%state_idx, 'rb') as file:
                    self.vstate.variables = flax.serialization.from_bytes(variables,file.read())
                with open(self.state_dir + "/%d.mpack"%state_idx, 'rb') as file:
                    self.vstate_fs.variables = flax.serialization.from_bytes(variables,file.read())

                self.is_op = IS_Operator(operator = self.model.H_jax, is_mode=alpha)
                vmc_fs = nk.VMC(hamiltonian=self.model.H_jax, optimizer=self.opt, variational_state=self.vstate_fs, preconditioner=self.sr_fs)
                e_fs, fs_grad = self.vstate_fs.expect_and_grad(self.model.H_jax)
                fs_ng_t = vmc_fs._forward_and_backward()
                fs_ng_leaves, _ = jax.tree_util.tree_flatten(fs_ng_t)
                fs_ng = jnp.concatenate([leaf.flatten() for leaf in fs_ng_leaves])
                log_q, log_q_vars = self.is_op.get_log_importance(self.vstate)

                if self.diag_exp == 'schedule':
                    diag_shift = self.diag_shift(state_idx*self.save_every)
                else: 
                    diag_shift = 1e-6
                self.compute_S_F = jax.jit(nkjax.vmap_chunked(lambda s : _compute_S_F(s, self.vstate._apply_fun, self.vstate.parameters, self.vstate.model_state, log_q, log_q_vars, self.chunk_size_jac, self.is_op, self.solver_fn, diag_shift), in_axes=0, chunk_size = self.chunk_size_resample))
                # with jax.checking_leaks():
                samples = self.vstate.sample_distribution(
                    log_q,
                    variables=log_q_vars, n_samples = self.Nsample * self.num_resample
                )

                batch_sample = samples.reshape((self.num_resample, 1, self.Nsample, -1))
                e, grad_e , ng = self.compute_S_F(batch_sample)
                ng_dense = flatten_tree_to_array(ng)
                print(fs_ng_t.keys())
                fs_dist_l.append(jax.vmap(lambda x : dot_prod(fs_ng, x), in_axes=0)(ng_dense))
                alpha_out['snr_e'].append(jnp.sqrt(e.size*jnp.abs(jnp.mean(e))**2/(jnp.var(e))))
                alpha_out['mean_e'].append(jnp.mean(e))
                alpha_out['var_e'].append(jnp.var(e))
                snr_grad_l.append(snr_tree(grad_e, fs_grad))
                snr_ng_l.append(snr_tree(ng[0], fs_ng_t))
    
            # stack snr and save 
            alpha_out['snr_ng'] = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *snr_ng_l)
            alpha_out['snr_grad'] = jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays, axis=0), *snr_grad_l)
            alpha_out["fs_dist"] = jnp.stack(fs_dist_l)
            jnp.savez(self.output_dir + f"/out_analysis_{alpha}_{diag_shift}.npz", alpha_out)

    #@partial(jax.jit, static_argnums=0)
    def compute_updated_state(self, dp):
        # update params and compute update vstate
        return _compute_updated_state(self.vstate.model, self.vstate.parameters, self.vstate.hilbert.all_states(), self.delta, dp)
    
@partial(jax.jit, static_argnames=('model'))
def _compute_updated_state(model, parameters, all_states, delta, dp):
    # update params and compute update vstate
    params, tree_def = jax.tree_util.tree_flatten(parameters)
    leaf_sizes = [leaf.size for leaf in params]
    partitioned_dp = jnp.split(dp, cumsum(leaf_sizes)[:-1])
    new_leaves = [params[i] - delta*partitioned_dp[i].reshape(params[i].shape) for i in range(len(params))]
    new_pars = jax.tree_util.tree_unflatten(tree_def, new_leaves)
    log_psi_updated = model.apply({"params": new_pars}, all_states)
    psi_new = jnp.exp(log_psi_updated)
    return psi_new

def _compute_S_F(samples, log_psi, parameters, model_state, log_q, log_q_vars, chunk_size, O, solver_fn, diag_shift):
    # sample and return estimators for S, F, NG, and energy
    # force call sample distribution to reset the batch of samples even though there may be cached ones (by calling sample_distribution instead of samples_distribution)

    # Hpsi_samples = vstate.samples_distribution(
    #     apply_fun, variables=variables, resample_fraction=O.resample_fraction
    # )

    O_exp, O_grad = expect_grad_is(
        log_psi,
        parameters,
        log_q,
        log_q_vars,
        model_state,
        O,
        samples,
        return_grad=True,
        chunk_size=chunk_size
    )

    O_grad = force_to_grad(O_grad, parameters)
    # ng = O_grad
    S = QGTJacobianDefaultConstructorIS(log_psi,
        parameters,
        model_state,
        samples,
        log_q,
        log_q_vars,
        importance_operator = O,
        holomorphic=True,
        chunk_size=chunk_size,
        diag_shift= diag_shift)
    
    ng = solver_fn(S, O_grad)
    return O_exp.mean.real, O_grad, ng
