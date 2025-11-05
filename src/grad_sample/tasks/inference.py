from omegaconf import DictConfig, OmegaConf
from grad_sample.tasks.fullsum_train import Trainer
import netket as nk
import netket_checkpoint as nkc
import os
import orbax.checkpoint as ocp

from flax import serialization
from hydra.core.global_hydra import GlobalHydra
from netket_checkpoint._src.utils.orbax_utils import state_dict_to_restoreargs
from netket_checkpoint._src.utils.iter_utils import cleanup_checkpointer

from jax.tree import structure as tree_structure

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

class Inference():
    def __init__(self, cfg: DictConfig):
          # Pass directly to Trainer
        nk.config.netket_enable_x64 = True
        ckpt_path = os.path.expandvars(cfg.ckpt_path)
        self.sym_ckpt = cfg.sym_ckpt
        self.sym_eval = cfg.sym_eval

        # Assume ckpt_path is the absolute path to your YAML config file
        # Example: ckpt_path = "/home/user/project/outputs/2025-10-28/config.yaml"
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()

        cfg_orig = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))

        # (Optional) make it structured, i.e., immutable for safety
        OmegaConf.set_struct(cfg_orig, True)

        self.trainer = Trainer(cfg_orig)

        options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=self.trainer.n_iter//5, keep_period=20)
        # sym_restore = int(self.trainer.ckpt_path[-1])
        nets = [f(self.trainer.ansatz) for f in self.trainer.model.symmetrizing_functions]
        self.vstate = nk.vqs.MCState(
            self.trainer.sampler,
            model=nets[self.sym_ckpt],
            n_samples=self.trainer.Nsample,
            # seed=self.trainer.seed,
            # sampler_seed=self.trainer.seed,
            # n_discard_per_chain=1,
            chunk_size=self.trainer.chunk_size,
        )

        # optimizer = nk.optimizer.Sgd(learning_rate=0.1)
        # driver = self.trainer.gs_func(
        #     optimizer,
        #     1e-4,
        #     self.trainer.vstate,
        #     self.trainer.is_distrib
        # )
        # chkptr = self.trainer.ckpt_restore.orbax_checkpointer()

        # driver_serialized = serialization.to_state_dict(driver)
        # state_serialized = driver_serialized.pop("state")
        
        # serialized_args = {
        #     "state": ocp.args.PyTreeRestore(
        #         state_dict_to_restoreargs(state_serialized, strict=False)
        #     ),
        # }
        # restored_data = chkptr.restore(
        #         chkptr.latest_step(), args=ocp.args.Composite(**serialized_args)
        #     )
        # restored_data['state'].keys()
        # samples_overdisp = restored_data['state']['sampler_states']['overdispersed']['σ']
        # self.restored_state = serialization.from_state_dict(driver.state, restored_data["state"])

        self.ckpt_restore = nkc.checkpoint.CheckpointManager(directory=os.path.join(ckpt_path, f'ckpt{self.sym_ckpt}'), options=options)
        distribution = self.trainer.is_distrib
    
        chain_name = distribution.name

        if chain_name not in self.vstate.sampler_states:
            log_prob_p_fun, variables_p = distribution(self.vstate._apply_fun, self.vstate.variables)
            self.vstate.init_sampler_distribution(
                log_prob_p_fun,
                variables=variables_p,
                chain_name=chain_name,
            )

        self.restored_state = self.ckpt_restore.restore_state(self.vstate)
        
        old_vars = self.restored_state.variables
        # old_sampler_states = self.restored_state.sampler_states

        # Create arbitrarly symmetrized vstate with the checkpointed variables
        self.vstate = nk.vqs.MCState(
            self.trainer.sampler,
            model=nets[self.sym_eval],
            n_samples=self.trainer.Nsample,
            # seed=self.trainer.seed,
            # sampler_seed=self.trainer.seed,
            # n_discard_per_chain=1,
            chunk_size=self.trainer.chunk_size,
        )
        updated_params = add_module(
                        old_params=old_vars["params"],
                        new_params=self.vstate.variables["params"],
                    )
        old_vars["params"] = updated_params
        self.vstate.variables = old_vars
        # self.vstate.sampler_states = old_sampler_states
        assert old_vars == self.vstate.variables
        self.vstate.n_samples = cfg.get('n_samples',2**15)
        self.vstate.n_chains = cfg.get('n_chains', 32)
        self.vstate.n_samples_per_chain = self.vstate.n_samples / self.vstate.n_chains
        self.vstate.chunk_size = cfg.get('chunk_size', None) #or 1024
        self.vstate.n_discard_per_chain = self.vstate.n_samples_per_chain

        self._ham = self.trainer.model.hamiltonian.to_jax_operator()
        # old_vars = self.trainer.vstate.variables
        # is_distrib = driver.importance_sampling_distribution
        # run_checkpointed=True

    def __call__(self):
        print('Expecting the variational energy of the restored state ...')
        self.exp_value = self.vstate.expect(self._ham)
        print(self.exp_value)

