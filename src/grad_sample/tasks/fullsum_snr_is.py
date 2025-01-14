from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import copy
import flax
import netket as nk
import netket.jax as nkjax
import jax.numpy as jnp
import jax
from tqdm import tqdm
from grad_sample.tasks.base import Problem
from grad_sample.utils.utils import cumsum, find_closest_saved_vals
import json
from grad_sample.is_hpsi.qgt import QGTJacobianDenseImportanceSampling
from grad_sample.is_hpsi.operator import IS_Operator

from grad_sample.is_hpsi.expect import *
from functools import partial
from grad_sample.utils.tree_op import shape_tree, snr_tree

class FullSumIS(Problem):
    def __init__(self, cfg: DictConfig):
        # instantiate parent class
        super().__init__(cfg)
        self.alpha_l = [0.0, 0.5, 1.0, 1.5, 2.0]
        self.dimH = self.model.hi.n_states
        self.save_dir = self.output_dir + "/is_analysis"
        self.chunk_size_vmap = self.dimH // (self.dimH // self.chunk_size_vmap)
        # self.chunk_size_jac = self.dimH // self.chunk_size_jac

        # self.vmap_change = nkjax.vmap_chunked(self.compute_S_F, in_axes=0, chunk_size = self.chunk_size_vmap) 
        self.out_dict = {}
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))
        E=  data["Energy"]["Mean"]["real"]
        E_err = jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs)
        
        self.out_dict["commons"] = {"save_every": self.save_every, "E_gs": self.E_gs, "E_err": E_err}
        
        # eval_s will be relative errors at every magnitude
        # self.eval_s = jnp.linspace(1, self.n_iter//self.save_every - 1, 10).astype(int)
        self.eval_s = find_closest_saved_vals(E_err, jnp.arange(len(E_err//10)), self.save_every)
        print(self.eval_s)
        self.diag_shift = 1e-6
        self.sample_size = 9
        self.Nsample = 2**self.sample_size
        self.sampler = nk.sampler.ExactSampler(hilbert= self.model.hi)
        self.chunk_size = self.chunk_size_jac
        self.vstate = nk.vqs.MCState(sampler= self.sampler, model=self.ansatz, chunk_size= self.chunk_size, n_samples= self.Nsample, seed=0)
        self.num_resample = 250
        self.compute_S_F = nkjax.vmap_chunked(self._compute_S_F, in_axes=0, chunk_size = self.num_resample)

        self.model.H_jax._setup()

    def __call__(self):
        k=0
        for alpha in self.alpha_l :
            alpha_out = {'sample_size': self.sample_size, 
                         'diag_shift': self.diag_shift, 
                         'n_batches': self.num_resample,
                         'snr_e': [],
                         'snr_ng': None,
                         'snr_grad': None,
                         "rel_err": []}
            snr_e_l = []
            snr_ng_l = []
            snr_grad_l = []
            for state_idx in tqdm(self.eval_s):   
                # load vstate with required variables
                variables = None
                with open(self.state_dir + "/%d.mpack"%state_idx, 'rb') as file:
                    self.vstate.variables = flax.serialization.from_bytes(variables,file.read())

                self.is_op = IS_Operator(operator = self.model.H_jax, is_mode=alpha)
                self.sr = nk.optimizer.SR(qgt = QGTJacobianDenseImportanceSampling(importance_operator=self.is_op, chunk_size=self.chunk_size_jac), solver=self.solver_fn, diag_shift=self.diag_shift, holomorphic = self.mode == "holomorphic")
                self.vmc = nk.VMC(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)

                # with jax.checking_leaks():
                e, grad_e, S, ng = self.compute_S_F(jnp.arange(self.num_resample))
                alpha_out['snr_e'].append(jnp.sqrt(jnp.abs(jnp.mean(e))/(e.size*jnp.var(e))))
                snr_grad_l.append(snr_tree(grad_e))
                snr_ng_l.append(snr_tree(ng))
                # self.out_dict[int(state_idx)] = {"in_idx": in_idx, "fid_vals": fid_vals, 
                #                             "dp_dist_ev": dp_dist_ev, "infid_ev": infid_ev, "vs_arr" : self.vstate_arr,
                #                             "pdf" : self.pdf, "Hloc" : self.Hloc,
                #                             "jac" : self.jacobian_orig, "delta": self.delta}
            # save   
            
            
        # jnp.savez(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}.npz", self.out_dict)
        # print("file saved at %s"%(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}.npz"))
    
    def _compute_S_F(self, i):
        # sample and return estimators for S, F, NG, and energy
        # force call sample distribution to reset the batch of samples even though there may be cached ones (by calling sample_distribution instead of samples_distribution)
        log_Hpsi, Hpsi_vars = self.is_op.get_log_importance(self.vstate)

        _ = self.vstate.sample_distribution(
            log_Hpsi,
            variables=Hpsi_vars,
        )
        e, grad_e = self.vstate.expect_and_grad(self.is_op)
        S = self.sr.lhs_constructor(self.vstate)
        ng = self.solver_fn(S, grad_e)
        return e.mean.real, grad_e, S.to_dense(), ng
    
    #@partial(jax.jit, static_argnums=0)
    def compute_updated_state(self, dp):
        # update params and compute update vstate
        return _compute_updated_state(self.vstate.model, self.vstate.parameters, self.vstate.hilbert.all_states(), self.delta, dp)

    # @partial(jax.jit, static_argnums=0)
    def get_change(self, indices):
        fid_vals = self.vmap_change(indices)
        in_idx = jnp.argsort(fid_vals)
        return in_idx, fid_vals
    
    # @partial(jax.jit, static_argnums=[0,1])
    def compute_change(self, alpha, vmap_version=True):
        # set to 0 row j in jacobian and component j in Hloc
        if vmap_version:
            return 1
        
        else: #used when we want to look at the updated dp for other purposes when we found the samples to remove
            return 1
    
    # @partial(jax.jit, static_argnums=0)
    def compute_change_prune(self, pdf, j):
        # set to 0 row j in jacobian and component j in Hloc
       
        pdf_new = pdf.at[j].set(0)
        # renormalize pdf
        pdf_new = pdf_new/(jnp.sum(pdf_new))
        
        # center and rescale according to new pdf
        jac_new = jnp.sqrt(pdf_new[:,None])*(self.jacobian_orig - jnp.sum(self.jacobian_orig*pdf_new[:,None],axis=0))
        Hloc_new = jnp.sqrt(pdf_new)*(self.Hloc - jnp.sum(self.Hloc*pdf_new))
   
        # # recompute qgt and rhs
        # new_rhs = jac_new.transpose().conj() @ Hloc_new
        # # almost same val
        # # print(new_rhs / (jac_new.transpose().conj() @ (jnp.sqrt(pdf)*(Hloc_new-jnp.sum(Hloc_new*pdf)))))
        # new_qgt = (jac_new.transpose().conj() @ jac_new) + self.diag_shift*jnp.eye(jac_new.shape[1])  
        # # Solve system
        # dp_sol =  self.solver_fn(new_qgt, new_rhs)[0]
        dp_sol = self.get_new_dp(jac_new, Hloc_new)
        # if vmap_version:
        # else: #used when we want to look at the updated dp for other purposes when we found the samples to remove
        return dp_sol, pdf_new
    
    # @partial(jax.jit, static_argnums=0)
    def get_new_dp(self, jac_new, Hloc_new):
        # recompute qgt and rhs
        new_rhs = jac_new.transpose().conj() @ Hloc_new
        # almost same val
        # print(new_rhs / (jac_new.transpose().conj() @ (jnp.sqrt(pdf)*(Hloc_new-jnp.sum(Hloc_new*pdf)))))
        new_qgt = (jac_new.transpose().conj() @ jac_new) + self.diag_shift*jnp.eye(jac_new.shape[1])  
        # Solve system
        dp_sol =  self.solver_fn(new_qgt, new_rhs)[0]
        return dp_sol
    
    def prune_to(self, pdf, i):
        # remove the sample
        # jax.lax.dynamic_slice(in_idx, (0,), (i,))
        dp_sol, pdf_new = self.compute_change_prune(pdf, i)
        psi_updated = self.compute_updated_state(dp_sol)
        return pdf_new, (curved_dist(self.exact_dp, dp_sol, self.S_fs), fs_dist(psi_updated, self.im_t_ev)/self.norm_infid)

    def prune(self):
        if self.mode=="oneshot":
            in_idx, fid_vals = self.get_change(jnp.arange(self.jacobian_orig.shape[0])) #compute infidelity change on entire basis
            pdf_out, out = jax.lax.scan(self.prune_to, self.pdf, in_idx)
            # out = prune_vmap(jnp.arange(1,self.dimH))
            return in_idx, fid_vals, out[0], out[1]
            return in_idx, fid_vals, jnp.array(dp_dist_ev), jnp.array(infid_ev)
            # return in_idx, fid_vals, jnp.array(dp_dist_ev), jnp.array(infid_ev)

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