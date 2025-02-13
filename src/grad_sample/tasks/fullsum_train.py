import jax
import jax.numpy as jnp

import netket as nk
import os
# import netket_pro as nkp
from jax.tree import structure as tree_structure
from grad_sample.tasks.base import Problem
from omegaconf import DictConfig, OmegaConf

from grad_sample.utils.utils import save_cb
from netket.vqs import FullSumState
import json
import matplotlib.pyplot as plt
from grad_sample.utils.utils import save_rel_err_fs, save_snr, save_rel_err_large, save_alpha
from functools import partial
import advanced_drivers as advd
import optax

def add_module(old_params: dict, new_params: dict, max_attempts: int = 10):
    """
    Modify old_params to match new_params by recursively adding the key {"module": ...} until the dictionaries match.
    If all keys of the dictionary already match at the beginning we do not attempt this.
    Returns the values of old_params with the new key structure.
    If the structures do not match after max_attempts iterations raise an error
    """
    for i in range(max_attempts):
        if tree_structure(old_params) != tree_structure(new_params):
            old_params = {"module": old_params}
        else:
            return old_params

    raise RuntimeError(
        f"Exceed maximum number of attempts to match params structures ({max_attempts})"
    )

class Trainer(Problem):
    def __init__(self, cfg: DictConfig, plot_training_curve=True):
        super().__init__(cfg)
        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        # if self.is_mode != None:
        #     self.gs = nk.VMC(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        # else:
        #     self.gs = nk.VMC(hamiltonian=self.model.hamiltonian.to_jax_operator(), optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        
        if self.use_symmetries:
            self.n_symm_stages = len(self.model.symmetrizing_functions)
            self.lr_factor = 0.2
            self.diag_shift_factor = 1e-2
            self.lr_schedulers = [
                    optax.cosine_decay_schedule(
                        init_value=self.lr,
                        decay_steps=self.n_iter,
                        alpha=self.lr_factor,
                        exponent=10,
                    )
                    for i in range(self.n_symm_stages)
                            ]   
            self.diag_shift_schedulers = [
                optax.cosine_decay_schedule(
                    init_value=self.diag_shift,
                    decay_steps=self.n_iter,
                    alpha=self.diag_shift_factor,
                    exponent=10,
                )
                for i in range(self.n_symm_stages)
            ]
            # symmetrized networks
            self.nets = [f(self.ansatz) for f in self.model.symmetrizing_functions]
            # implementation of vmc such that the schedulers can be changed for each optimization stage

        if self.sample_size !=0:
            if self.is_mode != None:
                # try out vmc_ng driver to use auto diagshift callback
                self.gs = advd.driver.VMC_NG_IS(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, diag_shift=self.diag_shift, auto_is=self.auto_is)
            else:
                self.gs = advd.driver.VMC_NG(hamiltonian=self.model.hamiltonian.to_jax_operator(), optimizer=self.opt, variational_state=self.vstate, diag_shift=self.diag_shift)
        else: #use netket vmc bc advd not compatible with FS State yet
            self.gs = nk.VMC(hamiltonian=self.model.hamiltonian.to_jax_operator(), optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        
        self.plot_training_curve = True
        self.fs_state_rel_err = FullSumState(hilbert = self.gs.state.hilbert, model = self.gs.state.model, chunk_size=None, seed=0)
        
        self.autodiagshift = advd.callbacks.PI_controller_diagshift(diag_shift_max=0.01)
        
        if self.save_every != None:
            self.out_log = (self.json_log, self.state_log)
        else :
            self.out_log = (self.json_log,)

        if self.E_gs != None:
            self.save_rel_err_cb = partial(save_rel_err_fs, e_gs = self.E_gs, fs_state = self.fs_state_rel_err, save_every=25, output_dir=self.output_dir)
        else:
            self.save_rel_err_cb  = partial(save_rel_err_large, e_ref=self.E_ref, n_sites=self.model.graph.n_nodes, save_every=50, output_dir=self.output_dir)
            
        if self.is_mode != None:
            self.callbacks=(self.save_rel_err_cb, save_alpha)
        else:
            self.callbacks=(self.save_rel_err_cb)

    def __call__(self):

        if not self.use_symmetries:
            self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=self.callbacks)

        else:
            old_vars = None  # dummy
            for i in range(self.n_symm_stages):
                print(
                    f"Symmetry stage {i}/{self.n_symm_stages-1}:"
                )

                self.vstate = nk.vqs.MCState(
                    self.sampler,
                    model=self.nets[i],
                    n_samples_per_rank=self.Nsample,
                    # seed=self.seed,
                    # sampler_seed=self.seed,
                    # n_discard_per_chain=self.n_discard_per_chain,
                    chunk_size=self.chunk_size,
                )
                self.fs_state_rel_err = FullSumState(hilbert = self.vstate.hilbert, model = self.vstate.model, chunk_size=None, seed=0)
       
                if self.E_gs != None:
                    self.callbacks=(self.save_rel_err_cb(self.fs_state_rel_err),)
                else:
                    self.callbacks = None
                if i > 0:
                    updated_params = add_module(
                        old_params=old_vars["params"],
                        new_params=self.vstate.variables["params"],
                    )
                    old_vars["params"] = updated_params
                    self.vstate.variables = old_vars
                    assert old_vars == self.vstate.variables

                optimizer = nk.optimizer.Sgd(learning_rate=self.lr_schedulers[i])
                
                driver = self.gs_func(
                    optimizer,
                    self.diag_shift_schedulers[i],
                    self.vstate,
                )
                if self.E_gs != None :
                    driver.run(
                        n_iter=self.n_iter,
                        out=self.out_log,
                        callback=self.callbacks,
                    )
                else :
                    driver.run(
                        n_iter=self.n_iter,
                        out=self.out_log,
                    )
                old_vars = self.vstate.variables

        if self.plot_training_curve and (self.E_gs != None or self.E_ref != None):
            log_opt = self.output_dir + ".log"
            data = json.load(open(log_opt))
            E=  data["Energy"]["Mean"]["real"]
            plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs), label= "MC")
            e_r_fs = data["rel_err"]
            plt.plot(e_r_fs["iters"], e_r_fs["value"], label= "FullSum")
            try :
                plt.title(f"Relative error w.r.t. exact GS during training, {self.Nsample} samples")
            except: 
                plt.title(f"Relative error w.r.t. exact GS during training")
            plt.xlabel("iteration")
            plt.ylabel("Relative error")
            plt.yscale("log")
        
        elif self.plot_training_curve:
            log_opt = self.output_dir + ".log"
            data = json.load(open(log_opt))
            E=  data["Energy"]["Mean"]["real"]
            plt.plot(E, label= "MC Energy")
            try :
                plt.title(f"Energy during training, {self.Nsample} samples")
            except: 
                plt.title(f"Energy during training")
            plt.xlabel("iteration")
            plt.ylabel("Energy")
            
        plt.legend()
        plt.savefig(self.output_dir + '/training.png')