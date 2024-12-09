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
from grad_sample.utils.distances import curved_dist, fs_dist
from grad_sample.utils.utils import cumsum, find_closest_saved_vals
import json

from functools import partial

class FullSumPruning(Problem):
    # pruning class, default to curved_dist method (distance in parameter space defined by the QGT)
    def __init__(self, cfg: DictConfig, mode = "oneshot", recompute_stride = None, deltadep=False):
        # instantiate parent class
        super().__init__(cfg)
        self.delta = 1e-4 #delta used in imaginary time evolution

        self.delta_l = [10**n for n in range(-5,0)]
        self.mode = mode
        self.deltadep = deltadep
        self.dimH = self.model.hi.n_states
        self.recompute_stride = recompute_stride
        self.save_dir = self.output_dir + "/{mode}"
        self.chunk_size_vmap = self.dimH // (self.dimH // self.chunk_size_vmap)
        # self.chunk_size_jac = self.dimH // self.chunk_size_jac

        self.vmap_change = nkjax.vmap_chunked(self.compute_change, in_axes=0, chunk_size = self.chunk_size_vmap) 
        self.out_dict = {}
        log_opt = self.output_dir + ".log"
        data = json.load(open(log_opt))
        E=  data["Energy"]["Mean"]["real"]
        E_err = jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs)
        self.out_dict["commons"] = {"save_every": self.save_every, "E_gs": self.E_gs, "E_err": E_err, "delta": self.delta}
        
        # eval_s will be relative errors at every magnitude
        # self.eval_s = jnp.linspace(1, self.n_iter//self.save_every - 1, 10).astype(int)
        self.eval_s = find_closest_saved_vals(E_err, jnp.arange(len(E_err//10)), self.save_every)
        self.delta_eval = jnp.ones(len(self.eval_s)) * 1e-5
        self.delta_eval = self.delta_eval.at[:5].set(1e-5)
        print(self.eval_s)
        self.diag_shift = 1e-10

    def __call__(self):
        self.set_strategy()
        if self.deltadep:
            print("Analyzing delta dependency")
            for delta in self.delta_l:
                self.delta = delta
                self.out_dict[delta] = {}
                for state_idx in tqdm(self.eval_s):
                    # todo : add delta dependency
                    self.load_state(state_idx)
                    in_idx, fid_vals, dp_dist_ev, infid_ev = self.prune()

                    self.out_dict[delta][int(state_idx)] = {"in_idx": in_idx, "fid_vals": fid_vals, 
                                                "dp_dist_ev": dp_dist_ev, "infid_ev": infid_ev, "vs_arr" : self.vstate_arr,
                                                "pdf" : self.pdf, "Hloc" : self.Hloc,
                                                "jac" : self.jacobian_orig}
                jnp.savez(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}_deltadep.npz", self.out_dict)
                print("file saved at %s"%(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}_deltadep.npz"))

        else:
            k=0
            for state_idx in tqdm(self.eval_s):
                # todo : add delta dependency
                if k > 4:
                    self.delta = 1e-3
                # self.delta = self.delta_eval[k].astype(float)
                k+=1
                self.load_state(state_idx)
                in_idx, fid_vals, dp_dist_ev, infid_ev = self.prune()
                self.out_dict[int(state_idx)] = {"in_idx": in_idx, "fid_vals": fid_vals, 
                                            "dp_dist_ev": dp_dist_ev, "infid_ev": infid_ev, "vs_arr" : self.vstate_arr,
                                            "pdf" : self.pdf, "Hloc" : self.Hloc,
                                            "jac" : self.jacobian_orig, "delta": self.delta}
            # save   
            
            jnp.savez(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}.npz", self.out_dict)
            print("file saved at %s"%(self.output_dir + f"/out_analysis_{self.mode}_{self.strategy}.npz"))

    def set_strategy(self):
        self.strategy = "curved_dist" 

    def load_state(self, state_idx):
        # change vstate
        variables = None
        with open(self.state_dir + "/%d.mpack"%state_idx, 'rb') as file:
            self.vstate.variables = flax.serialization.from_bytes(variables,file.read())

        # update relevant fields
        self.vstate_arr = self.vstate.to_array()
        self.pdf = self.vstate.probability_distribution()

        self.compute_jac_hloc()
        self.compute_dp()
        self.compute_im_t_ev()

    def compute_im_t_ev(self):
        self.im_t_ev = self.vstate_arr - self.delta* self.model.H_sp @ self.vstate_arr
        self.norm_infid = fs_dist(self.vstate_arr, self.im_t_ev)

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

    # @partial(jax.jit, static_argnums=0)
    def get_change(self, indices):
        fid_vals = self.vmap_change(indices)
        in_idx = jnp.argsort(fid_vals)
        return in_idx, fid_vals
    
    # @partial(jax.jit, static_argnums=[0,1])
    def compute_change(self, j, vmap_version=True):
        # set to 0 row j in jacobian and component j in Hloc
       
        pdf_new = self.pdf.at[j].set(0)
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
        if vmap_version:
            return curved_dist(self.exact_dp, dp_sol, self.S_fs)
        
        else: #used when we want to look at the updated dp for other purposes when we found the samples to remove
            return dp_sol
    
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

class InfidelityPruning(FullSumPruning):
    # def __init__(self, cfg, mode="oneshot", recompute_stride=None):
    #     super.__init__(cfg, mode, recompute_stride)
    def set_strategy(self):
        self.strategy = "infidelity" 

    def compute_change(self, j, vmap_version = True):
        # overrides compute_change function to state infidelity (FS distance) between first order imaginary time evolution and updated state
        # set sample prob to 0
        pdf_new = self.pdf.at[j].set(0)
        # renormalize pdf
        pdf_new = pdf_new/(jnp.sum(pdf_new))

        # compute new jacobian: recenter and sqrt rescale
        jac_new = jnp.sqrt(pdf_new[:,None])*(self.jacobian_orig - jnp.sum(self.jacobian_orig*pdf_new[:,None],axis=0))
        # center and scale H_loc too
        Hloc_new = jnp.sqrt(pdf_new)*(self.Hloc - jnp.sum(self.Hloc*pdf_new))

        # recompute qgt and rhs
        new_rhs = jac_new.transpose().conj() @ Hloc_new
        new_qgt = (jac_new.transpose().conj() @ jac_new) + self.diag_shift*jnp.eye(jac_new.shape[1])

        # Solve system
        dp_sol =  self.solver_fn(new_qgt, new_rhs)[0]
        psi_new = self.compute_updated_state(dp_sol)
       
        if vmap_version:
            return fs_dist(psi_new, self.im_t_ev)
        else: #used when we want to look at the updated dp for other purposes when we found the samples to remove
            return dp_sol, fs_dist(psi_new, self.im_t_ev)

