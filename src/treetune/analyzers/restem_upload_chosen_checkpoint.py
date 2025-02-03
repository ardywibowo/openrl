import json

from wandb.sdk.wandb_run import Run

from treetune.analyzers.analyzer import Analyzer
from treetune.common import logging_utils
from treetune.trainers.policy_trainer import PolicyTrainer

logger = logging_utils.get_logger(__name__)

@Analyzer.register("restem_upload_chosen_checkpoint")
class RestemUploadChosenCheckpoint(Analyzer):
    def __init__(
        self,
        cloud_logger: Run,
        runtime,
        **kwargs,
    ):
        from treetune.runtime.policy_iteration_runtime import \
            PolicyIterationRuntime

        assert isinstance(runtime, PolicyIterationRuntime)
        self.runtime: PolicyIterationRuntime = runtime
        super().__init__(cloud_logger, runtime, **kwargs)

    def analyze(self, force_rerun: bool = False, **kwargs):
        if not self.distributed_state.is_main_process:
            return

        super().analyze()
        self.get_analysis_root_dir().mkdir(parents=True, exist_ok=True)

        if not force_rerun and (self.get_analysis_root_dir() / "done").exists():
            return

        checkpoint_dir = self.runtime.exp_root / "checkpoints"
        ckpts = self.runtime._get_list_of_evaluation_checkpoints(checkpoint_dir, every_n_checkpoints=1, ignore_worker_vars=True)

        chosen_checkpoints = []
        for ckpt in ckpts:
            if (ckpt / "chosen_checkpoint").exists():
                chosen_checkpoints.append(ckpt)

        # now parse their names, each iteration should exactly have one chosen
        iter_to_chosen_ckpt = {}
        for ckpt in chosen_checkpoints:
            iteration = PolicyTrainer.parse_checkpoint_name(ckpt.name)[0] # (iteration, epoch, step)[0]
            assert iteration not in iter_to_chosen_ckpt, f"iteration {iteration} has multiple chosen checkpoints"
            iter_to_chosen_ckpt[iteration] = ckpt.name

        # now upload the chosen checkpoints as json
        with open(self.get_analysis_root_dir() / "chosen_checkpoints.json", "w") as f:
            json.dump(iter_to_chosen_ckpt, f, indent=4)

        # now upload the chosen checkpoints to the cloud
        self.cloud_logger.save(str((self.get_analysis_root_dir() / "chosen_checkpoints.json").absolute()), policy="now")

        (self.get_analysis_root_dir() / "done").touch()
