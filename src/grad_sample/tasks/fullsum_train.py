import jax
import jax.numpy as jnp

import netket as nk
import os
# import netket_pro as nkp
from grad_sample.tasks.base import Problem
from omegaconf import DictConfig, OmegaConf

from grad_sample.utils.utils import save_cb

import json
import matplotlib.pyplot as plt
from grad_sample.utils import save_rel_err

class Trainer(Problem):
    def __init__(self, cfg: DictConfig, plot_training_curve=True):
        super().__init__(cfg)
        # Save the current config to the custom path
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        if self.hpsi_is:
            self.gs = nk.VMC(hamiltonian=self.is_op, optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        else:
            self.gs = nk.VMC(hamiltonian=self.model.H, optimizer=self.opt, variational_state=self.vstate, preconditioner=self.sr)
        self.plot_training_curve = True

    def __call__(self):
        self.gs.run(n_iter=self.n_iter, out=(self.json_log, self.state_log))

        if self.plot_training_curve:
            log_opt = self.output_dir + ".log"
            data = json.load(open(log_opt))
            E=  data["Energy"]["Mean"]["real"]
            plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs))
            plt.title("Relative error w.r.t. exact GS during training")
            plt.xlabel("iteration")
            plt.ylabel("Relative error")
            plt.yscale("log")
            plt.savefig(self.output_dir + '/training.png')