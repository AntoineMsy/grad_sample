from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import os
import netket as nk
import optax
import jax.numpy as jnp

from utils.utils import save_cb, e_diag
from utils.plotting_setup import *

class Problem:
    def __init__(self, cfg : DictConfig):
        self.cfg = deepcopy(cfg)

        self.device = self.cfg.get("device")
        # set working device
        os.environ["CUDA_VISIBLE_DEVICES"]= str(self.device)

        # Instantiate model class (Ising, Heisenberg...)
        self.model = instantiate(self.cfg.model)

        # Instantiate ansatz
        self.ansatz = instantiate(self.cfg.ansatz)

        # set hparams and relevant variables
        self.solver_fn = call(self.cfg.solver_fn)
        self.lr = self.cfg.get("lr")
        self.diag_shift = self.cfg.get("diag_shift")
        self.n_iter = self.cfg.get('n_iter')
        self.chunk_size_jac = self.cfg.get("chunk_size_jac")
        self.chunk_size_vmap = self.cfg.get("chunk_size_vmap")
        self.save_every = self.cfg.get("save_every")
        self.base_path = self.cfg.get("base_path")

        self.holomorphic = True

        self.vstate = nk.vqs.FullSumState(hilbert=self.model.hi, model=self.ansatz, chunk_size=self.chunk_size_jac, seed=0)
        self.opt = optax.sgd(learning_rate=self.lr)

        self.sr = nk.optimizer.SR(solver=self.solver_fn, diag_shift=self.diag_shift, holomorphic=self.holomorphic)
        self.diag_exp = int(-jnp.log10(self.diag_shift)+1)
        
        self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.L}/RBM/alpha{self.ansatz.alpha}/saved_{self.save_every}_{self.diag_exp}"
        
        # create dir if it doesn't already exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.state_dir = self.output_dir + "/state"

        self.E_gs = e_diag(self.model.H_sp)
        print("The ground state energy is:", self.E_gs)
        
        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
        self.state_log = nk.logging.StateLog(output_prefix=self.state_dir, save_every=self.save_every)

    
