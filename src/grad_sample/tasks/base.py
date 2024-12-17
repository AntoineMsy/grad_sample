from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import os
import netket as nk
import optax
import scipy
import jax.numpy as jnp
import jax
from grad_sample.utils.utils import save_cb, e_diag
# from grad_sample.utils.plotting_setup import *
from grad_sample.ansatz.cnn import final_actfn
from deepnets.net.patches import extract_patches2d

from grad_sample.is_hpsi.qgt import QGTJacobianDenseImportanceSampling
from grad_sample.is_hpsi.operator import IS_Operator

from grad_sample.is_hpsi.expect import *

class Problem:
    def __init__(self, cfg : DictConfig):
        self.cfg = deepcopy(cfg)

        # self.device = self.cfg.get("device")
        # # set working device
        # os.environ["CUDA_VISIBLE_DEVICES"]= str(self.device)

        # Instantiate model class (Ising, Heisenberg...)
        self.model = instantiate(self.cfg.model)

        # Instantiate ansatz
        print(self.cfg.ansatz)
        self.mode = "holomorphic"
        if "LogStateVector" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hi)
            self.alpha = 0

        elif "LSTMNet" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hi)
            self.alpha = self.cfg.ansatz.layers

        elif "cnn" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, graph=self.model.lattice, final_actfn=final_actfn)
            self.alpha = self.cfg.ansatz.depth
            self.mode = "complex" 

        elif "ViT" in self.cfg.ansatz._target_:
            self.ansatz = call(self.cfg.ansatz, extract_patches = extract_patches2d)
            self.alpha = self.cfg.ansatz.d_model
            self.mode = "real"
        else:
            self.ansatz = instantiate(self.cfg.ansatz)
            self.alpha = self.ansatz.alpha

        dict_name = {"netket.models.RBM": "RBM", 
        "netket.models.LogStateVector": "log_state",
         "netket.experimental.models.LSTMNet": "RNN",
         "grad_sample.ansatz.cnn.ConvReLU": "CNN",
         "deepnets.net.ViT.net.ViT_Vanilla": "ViT"}
         
        self.ansatz_name = dict_name[self.cfg.ansatz._target_]

        # set hparams and relevant variables
        self.solver_fn = call(self.cfg.solver_fn)
        self.lr = self.cfg.get("lr")
        self.diag_shift = self.cfg.get("diag_shift")
        self.n_iter = self.cfg.get('n_iter')
        self.chunk_size_jac = self.cfg.get("chunk_size_jac")
        self.chunk_size_vmap = self.cfg.get("chunk_size_vmap")
        self.save_every = self.cfg.get("save_every")
        self.base_path = self.cfg.get("base_path")

        # self.holomorphic = True
        self.sample_size = self.cfg.get("sample_size")
        self.is_mode = self.cfg.get("is_mode")
        
        if self.sample_size == 0:
            if self.chunk_size_jac < self.model.hi.n_states:
                self.chunk_size = self.model.hi.n_states // (self.model.hi.n_states//self.chunk_size_jac)
            else:
                self.chunk_size = self.model.hi.n_states
            print(self.model.hi.n_states // self.chunk_size)
            print(self.chunk_size)
            print(self.model.hi.n_states )
            self.vstate = nk.vqs.FullSumState(hilbert=self.model.hi, model=self.ansatz, chunk_size=self.chunk_size, seed=0)
        else:
            self.Nsample = 2**self.sample_size
            self.chunk_size = self.chunk_size_jac
            self.sampler = nk.sampler.ExactSampler(hilbert= self.model.hi)
            self.vstate = nk.vqs.MCState(sampler= self.sampler, model=self.ansatz, chunk_size= self.chunk_size, n_samples= self.Nsample, seed=0)
            print("MC state loaded, num samples %d"%self.Nsample)
        self.opt = nk.optimizer.Sgd(learning_rate=self.lr)

        if self.is_mode != None:
            self.is_op = IS_Operator(operator = self.model.H_jax, mode=self.is_mode)
            self.sr = nk.optimizer.SR(qgt = QGTJacobianDenseImportanceSampling(importance_operator=self.is_op, chunk_size=self.chunk_size_jac), solver=self.solver_fn, diag_shift=self.diag_shift, holomorphic= self.mode == "holomorphic")
        else:
            self.sr = nk.optimizer.SR(solver=self.solver_fn, diag_shift=self.diag_shift, holomorphic= self.mode == "holomorphic")
        # self.sr = nk.optimizer.SR(diag_shift=self.diag_shift, holomorphic= self.mode == "holomorphic")
        
        self.diag_exp = int(-jnp.log10(self.diag_shift)+1)
        
        if "heisenberg" in self.model.name or "J1J2" in self.model.name:
            self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}_s{int(self.model.sign_rule)}/L{self.model.L}/{self.ansatz_name}/alpha{self.alpha}/saved_{self.save_every}_{self.diag_exp}"
        else:
            if self.sample_size == 0:
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.L}/{self.ansatz_name}/alpha{self.alpha}/saved_{self.save_every}_{self.diag_exp}"
            elif self.is_mode != None: 
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.L}/{self.ansatz_name}/alpha{self.alpha}/MC_{self.sample_size}_{self.is_mode}/saved_{self.save_every}_{self.diag_exp}"
            else:
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.L}/{self.ansatz_name}/alpha{self.alpha}/MC_{self.sample_size}/saved_{self.save_every}_{self.diag_exp}"
            
        # create dir if it doesn't already exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.state_dir = self.output_dir + "/state"

        self.E_gs = e_diag(self.model.H_sp)
        print("The ground state energy is:", self.E_gs)
        
        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
        self.state_log = nk.logging.StateLog(output_prefix=self.state_dir, save_every=self.save_every)

    
