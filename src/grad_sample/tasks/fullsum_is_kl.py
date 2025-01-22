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
from grad_sample.utils.distances import fs_dist

import json
from grad_sample.is_hpsi.qgt import QGTJacobianDenseImportanceSampling, QGTJacobianDefaultConstructorIS
from grad_sample.is_hpsi.operator import IS_Operator

from grad_sample.is_hpsi.expect import *
from functools import partial
from grad_sample.utils.tree_op import shape_tree, snr_tree, flatten_tree_to_array

class FullSumISKL(Problem):
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
        out = {
                'sample_size': self.sample_size,
                'n_batches': self.num_resample,
                "rel_err": self.error_s,
                }  
        
        for alpha in self.alpha_l:
            out[alpha] = {'klr' : [],
                'klf' : []}
            
        for state_idx in tqdm(self.eval_s): 
            
            # load vstate with required variables
            variables = None
            with open(self.state_dir + "/%d.mpack"%state_idx, 'rb') as file:
                self.vstate_fs.variables = flax.serialization.from_bytes(variables,file.read())

            vmc_fs = nk.VMC(hamiltonian=self.model.H_jax, optimizer=self.opt, variational_state=self.vstate_fs, preconditioner=self.sr_fs)

            # fs_ng = vmc_fs._forward_and_backward()
            # fs_ng_leaves, _ = jax.tree_util.tree_flatten(fs_ng)
            # fs_ng = jnp.concatenate([leaf.flatten() for leaf in fs_ng_leaves])

            self.vstate_arr = self.vstate_fs.to_array()
            self.pdf = self.vstate_fs.probability_distribution()
            self.Hloc = self.model.H_sp @ self.vstate_arr / self.vstate_arr
            self.Hloc_c = jnp.sqrt(self.pdf)*(self.Hloc - jnp.sum(self.Hloc*self.pdf))
            self.mode = "holomorphic"
            # uncentered jacobian
            self.jacobian_c = nkjax.jacobian(
                self.vstate._apply_fun,
                self.vstate.parameters,
                self.vstate.hilbert.all_states(), #in MC state, this is vstate.samples
                self.vstate.model_state,
                pdf=self.pdf,
                mode=self.mode,
                dense=True,
                center=False,
                chunk_size=10,
                _sqrt_rescale=True, # rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
            )
            self.force_fs_unrolled = jnp.abs(self.jacobian_c.transpose().conj() * self.Hloc_c)
            self.S_fs_unrolled = jnp.abs(jnp.outer(self.jacobian_c.transpose().conj(), self.jacobian_c))
            print(self.S_fs_unrolled.shape)
            psi_sq = jnp.abs(jnp.exp(self.vstate_arr))**2
            self.prob_target = (psi_sq[None,:] * self.force_fs_unrolled) / jnp.sum(psi_sq[None,:] * self.force_fs_unrolled, axis=1)[:,None]

            for alpha in self.alpha_l :
                
                alpha_distrib = jnp.abs(jnp.exp(self.vstate_arr))**alpha / jnp.sum(jnp.abs(jnp.exp(self.vstate_arr))**alpha)
                kl_f, kl_r = kl(self.prob_target, alpha_distrib[None,:])
                if self.diag_exp == 'schedule':
                    diag_shift = self.diag_shift(state_idx*self.save_every)
                else: 
                    diag_shift = 1e-6
                out[alpha]['klr'].append((jnp.mean(kl_r), jnp.var(kl_r)))
                out[alpha]['klf'].append((jnp.mean(kl_f), jnp.var(kl_f)))

        
            jnp.savez(self.output_dir + f"/out_analysis_kl.npz", out)

    def compute_jac_hloc(self):
        # compute local energies
        self.Hloc = self.model.H_sp @ self.vstate_arr / self.vstate_arr
        self.Hloc_c = jnp.sqrt(self.pdf)*(self.Hloc - jnp.sum(self.Hloc*self.pdf))
        self.mode = "holomorphic"
        # uncentered jacobian
        self.jacobian_orig = nkjax.jacobian(
            self.vstate._apply_fun,
            self.vstate.parameters,
            self.vstate.hilbert.all_states(), #in MC state, this is vstate.samples
            self.vstate.model_state,
            pdf=self.pdf,
            mode=self.mode,
            dense=True,
            center=False,
            chunk_size=self.chunk_size_jac,
            _sqrt_rescale=False, #(not) rescaled by sqrt[π(x)], but in MC this rescales by 1/sqrt[N_mc]
        )


    def compute_dp(self):
        # computes QGT, force and associated solution
        jac_c = self.jacobian_orig - jnp.sum(self.jacobian_orig*self.pdf[:,None],axis=0)
        jac_c = jnp.sqrt(self.pdf[:,jnp.newaxis])*jac_c
        Hloc_c = jnp.sqrt(self.pdf)*(self.Hloc - jnp.sum(self.Hloc*self.pdf))

        # force and QGT in fullsum, dense
        self.force_fs = jac_c.transpose().conj() @ Hloc_c 
        self.S_fs = (jac_c.transpose().conj() @ jac_c) + self.diag_shift*jnp.eye(jac_c.shape[1])

        # solve system
        self.exact_dp = self.solver_fn(self.S_fs, self.force_fs)[0]
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

def kl(p, q):
        # returns KL(p || q) and KL(q || p)
        return jnp.sum(p * jnp.log(p/q), axis=1), jnp.sum(q * jnp.log(q/p), axis=1)