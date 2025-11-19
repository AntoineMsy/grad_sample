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
from netket.callbacks import InvalidLossStopping
import json
import matplotlib.pyplot as plt
from grad_sample.utils.utils import save_rel_err_fs, save_snr, save_rel_err_large, save_alpha, compute_snr_callback, save_sampler_state
from functools import partial
import advanced_drivers as advd
import optax
import netket_checkpoint as nkc
from netket_pro.distributed import declare_replicated_array
import orbax.checkpoint as ocp

from flax import serialization

from netket_checkpoint._src.utils.orbax_utils import state_dict_to_restoreargs

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
        if jax.process_index() == 0:
            with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))

        if self.use_symmetries:
            self.n_symm_stages = len(self.model.symmetrizing_functions)
            print(self.n_symm_stages)
            self.lr_factor = 0.1
            self.diag_shift_factor = 1e-2
            self.lr_schedulers = [
                    optax.cosine_decay_schedule(
                        init_value=self.lr,
                        decay_steps=self.n_iter,
                        alpha=self.lr_factor,
                        exponent=1,
                    )
                    for i in range(self.n_symm_stages)
                            ]   
            self.diag_shift_schedulers = [
                optax.exponential_decay(
                    init_value=self.diag_shift,
                    transition_steps=self.n_iter//2,
                    decay_rate=0.96,
                    end_value = self.diag_shift * self.diag_shift_factor
                )
                for i in range(self.n_symm_stages)
            ]
            # symmetrized networks
            self.nets = [f(self.ansatz) for f in self.model.symmetrizing_functions]
            # implementation of vmc such that the schedulers can be changed for each optimization stage
        print('creating driver')
        if self.sample_size !=0:
            self.gs = advd.driver.VMC_NG(hamiltonian=self.model.hamiltonian.to_jax_operator(), 
                                            optimizer=self.opt, 
                                            importance_sampling_distribution=self.is_distrib,
                                            variational_state=self.vstate, 
                                            diag_shift=self.diag_shift, 
                                            auto_is=self.auto_is,
                                            use_ntk=self.use_ntk,
                                            momentum=self.momentum,
                                            # collect_gradient_statistics=self.collect_gradient_statistics,
                                            on_the_fly=False)
        else: #use netket vmc bc advd not compatible with FS State yet
            self.gs = nk.VMC(hamiltonian=self.model.hamiltonian.to_jax_operator(), 
                             optimizer=self.opt, 
                             variational_state=self.vstate, 
                             preconditioner=self.sr)
        
        self.plot_training_curve = True
        self.fs_state_rel_err = FullSumState(hilbert = self.gs.state.hilbert, 
                                             model = self.gs.state.model, 
                                             chunk_size=None, 
                                             seed=0)
        print('finished creating driver')
        # self.autodiagshift = advd.callbacks.PI_controller_diagshift(diag_shift_max=0.01, diag_shift_min=1e-6, safety_fac=1.0, clip_min=0.99, clip_max=1.01)
        
        if self.save_every != None:
            self.out_log = (self.json_log, self.state_log)
        else :
            self.out_log = (self.json_log,)
        # print(self.vstate.hilbert.n_states)
        # try:
        #     n_states = self.vstate.hilbert.n_states
        #     self.save_rel_err_cb = partial(save_rel_err_fs, 
        #                                    e_gs = self.E_gs, 
        #                                    fs_state = self.fs_state_rel_err, 
        #                                    save_every=2, 
        #                                    output_dir=self.output_dir)
        # except:
        #     print('Hilbert space too large to be indexed, using reference energy callback')
            
        # if self.sample_size == 0:
        #     self.callbacks = (lambda *x: True)  
        # else:  
        #     self.callbacks=(self.save_rel_err_cb,)
        # self.save_rel_err_cb  = partial(save_rel_err_large, 
        #                                     e_ref=self.E_gs, 
        #                                     n_sites=None, 
        #                                     save_every=50, 
        #                                     output_dir=self.output_dir)
        # self.save_sampler = partial(save_sampler_state, out_prefix=self.output_dir, save_every=250)
       
        
        # self.compute_snr_cb = partial(compute_snr_callback, 
        #                               fs_state = self.fs_state_rel_err,
        #                               H_sp = self.gs._ham.to_sparse(),
        #                               save_every=50
        #                              )                             
        
        # self.callbacks = (self.save_rel_err_cb)     
        self.callbacks = (InvalidLossStopping())
        self.restored_stage = cfg.get('restored_stage',2)
    def __call__(self):
        if not self.use_symmetries:
            print('calling run')
            options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=self.n_iter//10, keep_period=20)
            if jax.process_index() == 0:
                os.makedirs(os.path.join(self.output_dir, f"ckpt"), exist_ok=True)
            ckpt = nkc.checkpoint.CheckpointManager(directory=os.path.join(self.output_dir, f"ckpt"), options=options)
            ckpt_cb = advd.callbacks.CheckpointCallback(ckpt)

            self.callbacks = (InvalidLossStopping(), ckpt_cb)
            self.gs._optimizer_state = jax.tree.map(
                                    lambda x: declare_replicated_array(x),
                                    driver._optimizer_state)
            self.gs.run(n_iter=self.n_iter, out=self.out_log, callback=self.callbacks)
    
        else:
            run_checkpointed = False
            if self.ckpt_path is not None:
                options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=self.n_iter//5, keep_period=20)
                # sym_restore = int(self.ckpt_path[-1])
                
                self.vstate = nk.vqs.MCState(
                    self.sampler,
                    model=self.nets[self.restored_stage],
                    n_samples=self.Nsample,
                    # seed=self.seed,
                    # sampler_seed=self.seed,
                    # n_discard_per_chain=1,
                    chunk_size=self.chunk_size,
                )
                # optimizer = nk.optimizer.Sgd(learning_rate=self.lr)
                # driver = self.gs_func(
                #     optimizer,
                #     self.diag_shift,
                #     self.vstate,
                #     self.is_distrib
                # )

                self.ckpt_restore = nkc.checkpoint.CheckpointManager(directory=os.path.join(self.ckpt_path, f'ckpt{self.restored_stage}'), options=options)
                # OLD CODE
                # chkptr = self.ckpt_restore.orbax_checkpointer()
                # driver_serialized = serialization.to_state_dict(driver)
                # state_serialized = driver_serialized.pop("state")
                # serialized_args = {
                #     "state": ocp.args.PyTreeRestore(
                #         state_dict_to_restoreargs(state_serialized, strict=False)
                #     ),
                # }
                # restored_data = chkptr.restore(
                #     chkptr.latest_step(), args=ocp.args.Composite(**serialized_args)
                # )
                # restored_data["driver"]["state"] = restored_data["state"]
                # restored_state = serialization.from_state_dict(driver.state, restored_data["state"])
                # driver.state = restored_state
                # initialise the sampler
                distribution = self.is_distrib
    
                chain_name = distribution.name

                if chain_name not in self.vstate.sampler_states:
                    log_prob_p_fun, variables_p = distribution(self.vstate._apply_fun, self.vstate.variables)
                    self.vstate.init_sampler_distribution(
                        log_prob_p_fun,
                        variables=variables_p,
                        chain_name=chain_name,
                    )
                restored_state = self.ckpt_restore.restore_state(self.vstate)
                old_vars = restored_state.variables
                old_sampler_states = restored_state.sampler_states
                is_distrib = self.is_distrib
                if is_distrib.name == 'overdispersed':
                    alpha = 1.4402 #hard coded bc don't know how to retrieve from sampler
                    is_distrib.q_variables['alpha'] = jnp.array([alpha])
                # driver = self.gs_func(
                #     optimizer,
                #     self.diag_shift,
                #     restored_state,
                #     is_distrib
                # )
                run_checkpointed=True

            else:
                old_vars = None # dummy
                is_distrib = self.is_distrib

            for i in range(self.first_sym_stage, self.n_symm_stages):
                print(
                    f"Symmetry stage {i}/{self.n_symm_stages-1}:"
                )

                self.vstate = nk.vqs.MCState(
                    self.sampler,
                    model=self.nets[i],
                    n_samples=self.Nsample,
                    # seed=self.seed,
                    # sampler_seed=self.seed,
                    # n_discard_per_chain=1,
                    chunk_size=self.chunk_size,
                )
                # self.fs_state_rel_err = FullSumState(hilbert = self.vstate.hilbert, model = self.vstate.model, chunk_size=None, seed=0)

                if i > 0 or run_checkpointed:
                    updated_params = add_module(
                        old_params=old_vars["params"],
                        new_params=self.vstate.variables["params"],
                    )
                    old_vars["params"] = updated_params
                    self.vstate.variables = old_vars
                    self.vstate.sampler_states = old_sampler_states
                    assert old_vars == self.vstate.variables
                
                options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=self.n_iter//5, keep_period=20)
                # if jax.process_index() == 0:
                os.makedirs(os.path.join(self.output_dir, f"ckpt{i}"), exist_ok=True)
                ckpt = nkc.checkpoint.CheckpointManager(directory=os.path.join(self.output_dir, f"ckpt{i}"), options=options)
                ckpt_cb = advd.callbacks.CheckpointCallback(ckpt)

                self.callbacks = (InvalidLossStopping(), ckpt_cb)
                # self.callbacks = (InvalidLossStopping())
                optimizer = nk.optimizer.Sgd(learning_rate=self.lr)
                driver = self.gs_func(
                    optimizer,
                    self.diag_shift,
                    self.vstate,
                    is_distrib
                )
                driver._optimizer_state = jax.tree.map(
                                    lambda x: declare_replicated_array(x),
                                    driver._optimizer_state)
                # jax.debug.print('new is distrib {x}', x = driver.importance_sampling_distribution.q_variables)
                # if self.E_gs != None :
                driver.run(
                    n_iter=self.n_iter,
                    out=self.out_log,
                    callback=self.callbacks,
                )
                # else :
                #     driver.run(
                #         n_iter=self.n_iter,
                #         out=self.out_log,
                #     )
                old_vars = self.vstate.variables
                # jax.debug.print('old is distrib {x}', x = driver.importance_sampling_distribution.q_variables)
                is_distrib = driver.importance_sampling_distribution
        if jax.process_index() == 0:
            log_opt = self.output_dir + ".log"
            data = json.load(open(log_opt))
            E=  jnp.array(data["Energy"]["Mean"]["real"])
            
            if self.plot_training_curve and (self.E_gs != None):
                
                plt.plot(jnp.abs(E-self.E_gs)/jnp.abs(self.E_gs), label= "MC")
                
                try :
                    plt.title(f"Relative error w.r.t. exact GS during training, {self.Nsample} samples")
                    e_r_fs = data["rel_err"]
                    plt.plot(e_r_fs["iters"], e_r_fs["value"], label= "FullSum")
                except: 
                    plt.title(f"Relative error w.r.t. exact GS during training")
                plt.xlabel("iteration")
                plt.ylabel("Relative error")
                plt.yscale("log")
            
            elif self.plot_training_curve:
                
                plt.plot(E, label= "MC Energy")
                try :
                    plt.title(f"Energy during training, {self.Nsample} samples")
                except: 
                    plt.title(f"Energy during training")
                plt.xlabel("iteration")
                plt.ylabel("Energy")
                
            plt.legend()
            plt.savefig(self.output_dir + '/training.png')