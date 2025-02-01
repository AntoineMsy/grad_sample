import jax
import jax.numpy as jnp

import netket as nk
import os
# import netket_pro as nkp
from grad_sample.tasks.base import Problem
from omegaconf import DictConfig, OmegaConf

from grad_sample.utils.utils import save_cb
from netket.vqs import FullSumState
import json
import matplotlib.pyplot as plt
from grad_sample.utils.utils import save_rel_err_fs, save_snr
from functools import partial
import advanced_drivers as advd

class Trainer(Problem):
    def __init__(self, cfg: DictConfig, plot_training_curve=True):
        super().__init__(cfg)
        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if self.is_mode != None:
            self.gs = nk.VMC(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        else:
            self.gs = nk.VMC(hamiltonian=self.model.hamiltonian.to_jax_operator(), optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)

        # if self.is_mode != None:
        #     # try out vmc_ng driver to use auto diagshift callback
        #     self.gs = advd.driver.VMC_NG(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, diag_shift=self.diag_shift)
        # else:
        #     self.gs = advd.driver.VMC_NG(hamiltonian=self.model.H_jax, optimizer=self.opt, variational_state=self.vstate, diag_shift=self.diag_shift)
        
        self.plot_training_curve = True
        self.fs_state_rel_err = FullSumState(hilbert = self.gs.state.hilbert, model = self.gs.state.model, chunk_size=None, seed=0)
        self.save_rel_err_cb = partial(save_rel_err_fs, e_gs = self.E_gs, fs_state = self.fs_state_rel_err, save_every =25)

        self.autodiagshift = advd.callbacks.PI_controller_diagshift(diag_shift_max=0.01)

        if self.save_every != None:
            self.out_log = (self.json_log, self.state_log)
        else :
            self.out_log = (self.json_log,)

    def __call__(self):
        if self.E_gs != None:
            self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=(self.save_rel_err_cb,))
        else:
            self.gs.run(n_iter=self.n_iter, out=self.out_log)
        # self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=(self.save_rel_err_cb, self.autodiagshift))

        if self.plot_training_curve and self.E_gs != None:
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
            plt.plot(jnp.abs(E), label= "MC Energy")
            try :
                plt.title(f"Energy during training, {self.Nsample} samples")
            except: 
                plt.title(f"Energy during training")
            plt.xlabel("iteration")
            plt.ylabel("Energy")
            
        plt.legend()
        plt.savefig(self.output_dir + '/training.png')