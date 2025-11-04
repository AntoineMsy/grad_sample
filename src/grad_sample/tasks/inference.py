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

class Inference():
    def __init__(self, cfg: DictConfig):
          # Pass directly to Trainer
        nk.config.netket_enable_x64 = True
        ckpt_path = os.path.expandvars(cfg.ckpt_path)

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
        sym_restore = 0
        nets = [f(self.trainer.ansatz) for f in self.trainer.model.symmetrizing_functions]
        self.trainer.vstate = nk.vqs.MCState(
            self.trainer.sampler,
            model=nets[sym_restore],
            n_samples=self.trainer.Nsample,
            # seed=self.trainer.seed,
            # sampler_seed=self.trainer.seed,
            # n_discard_per_chain=1,
            chunk_size=self.trainer.chunk_size,
        )

        optimizer = nk.optimizer.Sgd(learning_rate=0.1)
        driver = self.trainer.gs_func(
            optimizer,
            1e-4,
            self.trainer.vstate,
            self.trainer.is_distrib
        )

        self.trainer.ckpt_restore = nkc.checkpoint.CheckpointManager(directory=os.path.join(ckpt_path, 'ckpt'), options=options)
        chkptr = self.trainer.ckpt_restore.orbax_checkpointer()

        driver_serialized = serialization.to_state_dict(driver)
        state_serialized = driver_serialized.pop("state")
        
        serialized_args = {
            "state": ocp.args.PyTreeRestore(
                state_dict_to_restoreargs(state_serialized, strict=False)
            ),
        }
        restored_data = chkptr.restore(
                chkptr.latest_step(), args=ocp.args.Composite(**serialized_args)
            )
        restored_data['state'].keys()
        samples_overdisp = restored_data['state']['sampler_states']['overdispersed']['σ']
        
        self.restored_state = serialization.from_state_dict(driver.state, restored_data["state"])
        self.restored_state.n_samples = cfg.get('n_samples',2**15)
        self.restored_state.n_chains = cfg.get('n_chains', 32)
        self.restored_state.n_samples_per_chain = self.restored_state.n_samples / self.restored_state.n_chains
        self.restored_state.chunk_size = cfg.get('chunk_size', None) #or 1024
        self.restored_state.n_discard_per_chain = self.restored_state.n_samples_per_chain

        self._ham = driver._ham
        # old_vars = self.trainer.vstate.variables
        # is_distrib = driver.importance_sampling_distribution
        # run_checkpointed=True

    def __call__(self):
        print('Expecting the variational energy of the restored state ...')
        self.exp_value = self.restored_state.expect(self._ham)
        print(self.exp_value)

