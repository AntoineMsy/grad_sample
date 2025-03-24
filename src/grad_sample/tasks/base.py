from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate
import os
import json
import netket as nk
import optax
import scipy
import jax.numpy as jnp
import numpy as np
import jax
from grad_sample.utils.utils import save_cb, e_diag

from grad_sample.is_hpsi.qgt import QGTJacobianDenseImportanceSampling
from grad_sample.is_hpsi.operator import IS_Operator
from grad_sample.is_hpsi.expect import *
import auto_importance as advd

from typing import Sequence
def to_sequence(arg):
    # tranforms arguments into sequences if they're just single values
    if not isinstance(arg, Sequence):
        return (arg,)
    else:
        return arg
    
class Problem:
    def __init__(self, cfg : DictConfig):
        self.cfg = deepcopy(cfg)
        nk.config.netket_enable_x64 = True

        # Instantiate model class (Ising, Heisenberg...)
        self.model = instantiate(self.cfg.model)

        # Instantiate ansatz
       
        self.mode = "holomorphic"

        if "LogStateVector" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hilbert_space)
            self.alpha = 0

        elif 'RBMSymm' in self.cfg.ansatz._target_:
            sym_group = self.model.graph.translation_group()
            self.ansatz = nk.models.RBMSymm(alpha = self.cfg.ansatz.alpha, param_dtype=complex, symmetries = sym_group)
            self.alpha = self.ansatz.alpha

        elif "LSTMNet" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hilbert_space)
            self.alpha = self.cfg.ansatz.layers

        elif "cnn" in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, lattice=self.model.graph)
            self.alpha = len(self.cfg.ansatz.channels)
            self.mode = "complex" 

        elif "ViT" in self.cfg.ansatz._target_:
            # only works with rajah's model in models/system !
            self.ansatz = call(self.cfg.ansatz, system = self.model).network
            self.alpha = f"{self.cfg.ansatz.depth}_{self.cfg.ansatz.d_model}_{self.cfg.ansatz.heads}" #unique identifier for vit
            self.mode = "complex"

        elif 'MLP' in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hidden_activations=[nk.nn.log_cosh]*len(self.cfg.ansatz.hidden_dims_alpha))
            self.alpha = self.ansatz.hidden_dims_alpha[0]

        elif 'LogNeuralBackflow' in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz, hilbert = self.model.hilbert_space)
            self.alpha = self.cfg.ansatz.hidden_units
            self.mode = 'complex'

        elif 'RBM' in self.cfg.ansatz._target_:
            self.ansatz = instantiate(self.cfg.ansatz)
            self.alpha = self.ansatz.alpha
            if self.cfg.ansatz.param_dtype == 'complex':
                self.mode = 'holomorphic'
            else :
                self.mode = 'real'
            
            

        dict_name = {"netket.models.RBM": "RBM",
        'netket.models.RBMSymm': 'RBMSymm',             
        "netket.models.LogStateVector": "log_state",
         "netket.experimental.models.LSTMNet": "RNN",
         "grad_sample.ansatz.cnn.CNN": "CNN",
         "deepnets.net.ViT2D": "ViT2D",
         "deepnets.net.ViT1D": "ViT1D",
         'netket.models.MLP': 'MLP',
         'grad_sample.ansatz.nnbf.LogNeuralBackflow' : 'NNBF'}
         
        self.ansatz_name = dict_name[self.cfg.ansatz._target_]

        # set hparams and relevant variables
        self.solver_fn = call(self.cfg.solver_fn)
        self.lr = self.cfg.get("lr")
        self.diag_shift = self.cfg.get("diag_shift")
        self.diag_exp = self.diag_shift
        self.n_iter = self.cfg.get('n_iter')
        
        self.chunk_size_jac = self.cfg.get("chunk_size_jac")
        self.chunk_size_vmap = self.cfg.get("chunk_size_vmap")
        self.chunk_size_vstate = self.cfg.get('chunk_size_vstate', None)

        self.save_every = self.cfg.get("save_every")
        self.base_path = self.cfg.get("base_path")
        
        try:
            self.base_path = os.environ['SCRATCH'] + self.base_path
        except:
            print('scratch folder not specified in env variables')
            self.base_path = "/scratch/.amisery" + self.base_path

        self.sample_size = self.cfg.get("sample_size")
        self.is_distrib = instantiate(self.cfg.is_distrib)
        self.auto_is = self.cfg.get("auto_is", False)
        self.use_symmetries = self.cfg.get("use_symmetries", False)
        
        if self.diag_shift == 'schedule':
            start_diag_shift, end_diag_shift = 1e-2, 1e-4

            # Define a linear schedule for diag_shift using optax

            self.diag_shift = optax.cosine_decay_schedule(
                init_value=start_diag_shift,
                decay_steps=2000,
                alpha=0.001
            ) #moves the diagonal shift from 1e-2 to 1e-4
    
        if self.lr == "schedule":
            lr_schedule = optax.cosine_decay_schedule(
                                                    init_value=3e-3,
                                                    decay_steps=3000,
                                                    alpha=0.1
                                                ) #moves the lr from 1e-3 to 1e-4
            # for qchem
            lr_schedule = optax.cosine_decay_schedule(
                                        init_value=0.1,
                                        decay_steps=300,
                                        alpha=0.01
                                        )
            self.opt = optax.sgd(learning_rate=lr_schedule)
        else:
            self.opt = optax.sgd(learning_rate=self.lr)
        # define optimizer
        # self.opt = optax.inject_hyperparams(optax.sgd)(learning_rate=self.lr) #used with autodiagshift
        # Choose between SR and SRt automatically
        rng_key_pars = jax.random.PRNGKey(np.random.randint(10000))
        # rng_key_pars = jax.random.PRNGKey(5)
        params = self.ansatz.init(
            rng_key_pars, jnp.zeros((1, self.model.graph.n_nodes))
        )
        max_nparams = nk.jax.tree_size(params)
        print(f"Nparams = {max_nparams}")
        
        if self.sample_size == 0:
            if self.chunk_size_jac < self.model.hilbert_space.n_states:
                self.chunk_size = self.model.hilbert_space.n_states // (self.model.hilbert_space.n_states//self.chunk_size_jac)
            else:
                self.chunk_size = self.model.hilbert_space.n_states
            self.chunk_size = None
            # print(self.model.hi.n_states // self.chunk_size)
            # print(self.chunk_size)
            # print(self.model.hi.n_states )
            self.vstate = nk.vqs.FullSumState(hilbert=self.model.hilbert_space, model=self.ansatz, chunk_size=self.chunk_size, seed=0)
        else:
            self.Nsample = 2**self.sample_size
            self.chunk_size = self.chunk_size_jac
            self.use_ntk = max_nparams > self.Nsample
            if "Exact" in self.cfg.sampler._target_:
                self.sampler = instantiate(self.cfg.sampler, hilbert= self.model.hilbert_space)
            elif 'Exchange' in self.cfg.sampler._target_:
                self.sampler = instantiate(self.cfg.sampler, hilbert=self.model.hilbert_space, 
                                                             graph=self.model.graph, 
                                                             sweep_size=self.model.graph.n_nodes, 
                                                             n_chains_per_rank=self.Nsample // 2
                                                             )
            self.vstate = nk.vqs.MCState(sampler= self.sampler, 
                                         model=self.ansatz, 
                                         chunk_size= self.chunk_size, 
                                         n_samples= self.Nsample,
                                        #  seed=0
                                        )
            print("MC state loaded, num samples %d"%self.Nsample)

        if "LogStateVector" in self.cfg.ansatz._target_:
            self.vstate.init_parameters()

        self.sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianDense, 
                                    solver=self.solver_fn, 
                                    diag_shift=self.diag_shift, 
                                    holomorphic= self.mode == "holomorphic")
        
        self.gs_func = lambda opt, dshift, vstate : advd.driver.VMC_NG(hamiltonian=self.model.hamiltonian.to_jax_operator(), 
                                                                        optimizer=opt, 
                                                                        sampling_distribution=self.is_distrib,
                                                                        variational_state=vstate, 
                                                                        diag_shift = dshift, 
                                                                        use_ntk = self.use_ntk,
                                                                        on_the_fly = False,
                                                                        auto_is = self.auto_is)
      
        # code only support default and overdispersed distribution for naming right now
        if self.sample_size == 0:
            self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/{self.lr}_{self.diag_exp}"
        elif self.is_distrib.name == 'overdispersed':
            if self.auto_is: 
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}_isauto/{self.lr}_{self.diag_exp}"
            else:
                self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}_{self.is_distrib.q_variables['alpha'].item()}/{self.lr}_{self.diag_exp}"
        elif self.is_distrib.name == 'default':
            self.output_dir = self.base_path + f"/{self.model.name}_{self.model.h}/L{self.model.graph.n_nodes}/{self.ansatz_name}/{self.alpha}/MC_{self.sample_size}/{self.lr}_{self.diag_exp}"
        else:
            raise NotImplementedError()
        
        # create dir if it doesn't already exist, if not in analysis mode
        self.run_index = self.cfg.get("run_index")
        if self.run_index == None:
            run_index = 0
            while True:
                run_dir = os.path.join(self.output_dir, f"run_{run_index}")
                if not os.path.exists(run_dir):
                    os.makedirs(run_dir)
                    self.output_dir = run_dir  # Update the output_dir to point to the newly created run_N folder
                    break
                run_index += 1
        else :
            self.output_dir = self.output_dir + '/run_%d'%self.run_index

        os.makedirs(self.output_dir, exist_ok=True)
        print(self.output_dir)
        self.state_dir = self.output_dir + "/state"
        if hasattr(self.model, 'E_fci') and self.model.E_fci is not None:
            self.E_gs = self.model.E_fci
        else:
            try :
                try:
                    self.e_dict = self.ref_energies[self.model.name][str(self.model.h)][str(int(self.model.graph.n_nodes**(1/self.model.graph.ndim)))]
                    if 'exact' in self.e_dict.keys():
                        self.E_ref = self.e_dict['exact']
                    
                except:
                    self.E_gs = e_diag(self.model.hamiltonian.to_sparse())
                    print("The ground state energy is:", self.E_gs)
            except : 
                self.E_gs = None
                print('Hilbert space too large for exact diag, loading reference energy from litterature')
                self.ref_energies = json.load(open("../../energy_ref_litt.json"))
            
                self.e_dict = self.ref_energies[self.model.name][str(self.model.h)][str(int(self.model.graph.n_nodes**(1/self.model.graph.ndim)))]
                if 'exact' in self.e_dict.keys():
                    self.E_ref = self.e_dict['exact']
                elif 'qmc' in self.e_dict.keys():
                    self.E_ref = self.e_dict['qmc']
                elif 'rbm+pp' in self.e_dict.keys():
                    self.E_ref = self.e_dict['rbm+pp']
                else :
                    self.E_ref = self.e_dict['aochen']
                print('Ref energy %.4f'%(self.E_ref*self.model.graph.n_nodes*4))
                self.E_gs = self.E_ref*self.model.graph.n_nodes*4

        self.E_gs_per_site = self.E_gs/self.model.graph.n_nodes/4
            # except:
            #     raise(FileNotFoundError(f'Error while retrieving reference energy for {self.model.name}, at coupling {self.model.h} and L {self.model.graph.n_nodes**(1/self.model.graph.ndim)} '))
            

        self.json_log = nk.logging.JsonLog(output_prefix=self.output_dir)
        if self.save_every != None:
            self.state_log = nk.logging.StateLog(output_prefix=self.state_dir, save_every=self.save_every)

    
