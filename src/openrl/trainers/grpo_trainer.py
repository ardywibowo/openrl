import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from weakref import WeakValueDictionary

import numpy as np
import torch
from accelerate.checkpointing import load_custom_state, save_custom_state
from accelerate.utils import gather, pad_across_processes, release_memory
from datasets import Dataset
from deepspeed import DeepSpeedEngine
from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.integrations import HfTrainerDeepSpeedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_pt_utils import get_model_param_count

from openrl.common import JsonDict, Lazy
from openrl.common.deepspeed_utils import (prepare_data_loader_for_inference,
                                           prepare_data_loader_for_training)
from openrl.common.logging_utils import get_logger
from openrl.common.py_utils import need_to_minimize_stored_files
from openrl.common.wandb_utils import get_repo_dir
from openrl.models.base_model import Model
from openrl.trainers.arguments import TrainingArguments
from openrl.trainers.base_trainer import Trainer
from openrl.trainers.data_collator import (COLUMN_ACTOR_SHIFTED_LOGPS,
                                           COLUMN_REF_SHIFTED_LOGPS,
                                           COLUMN_VALUES, PPODataCollator)
from openrl.trainers.deepspeed_policy_trainer import DeepSpeedPolicyTrainer
from openrl.trainers.policy_trainer import Checkpoint
from openrl.trainers.utils import (DeepSpeedRunningMoments,
                                   close_to_zero_percentage,
                                   entropy_from_logits, masked_mean,
                                   masked_var, monitor_tensor_anomalies)

from .ppo_trainer import AdaptiveKLController, FixedKLController

logger = get_logger(__name__)


@dataclass
class GRPOHParams:
    """
    Configuration class for GRPOTrainer.

    Parameters:
        adap_kl_ctrl (bool):
            Use adaptive KL control, otherwise linear.
        init_kl_coef (Optional[float]):
            Initial KL penalty coefficient (used for adaptive and linear control).
        kl_penalty (Literal["kl", "abs", "mse", "full"]):
            KL penalty options. 'kl': model_logp - ref_logp, 'abs': abs(kl),
            'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution.
        target (Optional[float]):
            Target KL value for adaptive KL control.
        gamma (float):
            Gamma parameter for advantage calculation.
        lam (float):
            Lambda parameter for advantage calculation.
        cliprange (float):
            Range for clipping in PPO policy gradient loss.
        cliprange_value (float):
            Range for clipping values in loss calculation.
        vf_coef (float):
            Scaling factor for value loss.
        early_stopping (bool):
            Whether to stop the PPO optimization loop early if the KL is too high.
        target_kl (float):
            Stop early if we exceed this value by over 50%.
        compare_steps (int):
            Number of steps between comparison of the current reward with the best seen so far.
        ratio_threshold (float):
            Skip mini-batches with high PPO ratios that can cause loss spikes.
        temperature (float):
            The temperature used for sampling.
    """
    
    adap_kl_ctrl: bool = True
    init_kl_coef: Optional[float] = 0.2
    kl_penalty: Literal["kl", "abs", "mse", "full", "control_variate"] = "kl"
    kl_penalty_loss_type: Optional[Literal["kl", "abs", "mse", "control_variate"]] = (
        None
    )
    kl_penalty_loss_clip_max: float = 10000
    kl_penalty_loss_clip_min: float = 0
    force_disable_kl_penalty: bool = False
    target: Optional[float] = 6.0
    horizon: Optional[int] = 10000
    gamma: float = 1
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    early_stopping: bool = False
    target_kl: float = 1
    compare_steps: int = 1
    ratio_threshold: float = 10.0
    temperature: float = 1.0
    
    def __post_init__(self):
        assert self.temperature > 0, "Temperature should be positive."


@Trainer.register("grpo")
class GRPOTrainer(DeepSpeedPolicyTrainer):
    def __init__(
        self,
        num_episodes_per_iteration: int,
        params: JsonDict,
        actor_model: Lazy[Model],
        actor_deepspeed_config: JsonDict,
        general_training_args: JsonDict,
        reference_model: Optional[Lazy[Model]] = None,
        reference_deepspeed_config: Optional[JsonDict] = None,
        num_iterations: int = 1,
        num_epochs_per_iteration: int = 1,
        report_entropy: bool = True,
        align_skipping_on_overflow: bool = True,
        enable_exponential_moving_average_actor: bool = False,
        cache_reference_model_on_temp_storage: bool = False,
        temp_checkpoint_dir: Optional[str] = None,
        profile_torch_memory: bool = False,
        cache_deepspeed_engines: bool = False,
        move_reference_model_to_cpu: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._set_process_log_level(logger)
        
        self.grpo_hparams = GRPOHParams(**params)
        self.args = TrainingArguments(**general_training_args)
        self.align_skipping_on_overflow = align_skipping_on_overflow
        
        self.num_iterations = num_iterations
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self._compute_batch_size_and_steps()
        
        self.actor_lazy = actor_model
        self.actor_deepspeed_config = actor_deepspeed_config
        
        self.reference_lazy = reference_model
        self.reference_deepspeed_config = reference_deepspeed_config
        if self.reference_lazy is None:
            logger.info("No reference model provided. We then assume no KL penalty.")

        if enable_exponential_moving_average_actor:
            self.ema_actor_model = self.actor_lazy
        else:
            self.ema_actor_model = None

        # This operation is done on the same data across all processes
        # So, there is no need to synchronize the operation
        self.running_scores = DeepSpeedRunningMoments(force_no_sync=True)

        if self.grpo_hparams.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                self.grpo_hparams.init_kl_coef,
                self.grpo_hparams.target,
                horizon=(
                    self.total_num_training_steps * self.global_batch_size
                ),  # Total number of episodes
            )
        else:
            self.kl_ctl = FixedKLController(self.grpo_hparams.init_kl_coef)
        
        from deepspeed.utils import logger as ds_logger
        
        ds_logger.setLevel(logging.DEBUG)
        
        # We are very conservative with memory (both CPU and GPU) and we want to avoid OOM errors.
        # We load the models in the memory whenever needed. This is why this variable exists.
        self.checkpoint_path_to_load = None
        
        if temp_checkpoint_dir is not None:
            self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
        else:
            self.temp_checkpoint_dir = get_repo_dir() / "temp_ppo_checkpoints"
            logger.info(
                f"No temporary checkpoint directory provided. Using {self.temp_checkpoint_dir}"
            )
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_reference_model = cache_reference_model_on_temp_storage
        self.profile_torch_memory = profile_torch_memory
        self.cache_deepspeed_engines = cache_deepspeed_engines
        self.move_reference_model_to_cpu = move_reference_model_to_cpu
        self.report_entropy = report_entropy
        
        if self._is_main_process():
            if getattr(self.cloud_logger, "define_metric", None):
                self.cloud_logger.define_metric("train/global_iteration")
                self.cloud_logger.define_metric(
                    "episodes_metric/*",
                    step_metric="train/global_iteration",
                    step_sync=True,
                )
        
    def _is_kl_penalty_enabled(self):
        return (
            not self.grpo_hparams.force_disable_kl_penalty
            and self.reference_lazy is not None
        )

    def _compute_batch_size_and_steps(self):
        if self.args.target_train_batch_size is not None:
            if (
                self.args.per_device_train_batch_size is None
                and self.args.gradient_accumulation_steps is None
            ):
                raise ValueError(
                    "Either per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )
            if (
                self.args.per_device_train_batch_size is not None
                and self.args.gradient_accumulation_steps is not None
            ):
                raise ValueError(
                    "Only one of per_device_train_batch_size or gradient_accumulation_steps "
                    "should be provided."
                )
            
            if self.args.per_device_train_batch_size is not None:
                self.args.gradient_accumulation_steps = (
                    self.args.target_train_batch_size
                    // self.args.per_device_train_batch_size
                    // self.distributed_state.num_processes
                )
            elif self.args.gradient_accumulation_steps is not None:
                self.args.per_device_train_batch_size = (
                    self.args.target_train_batch_size
                    // self.args.gradient_accumulation_steps
                    // self.distributed_state.num_processes
                )

        self.global_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.distributed_state.num_processes
        )
        self.total_num_training_steps = (
            self.num_iterations
            * self.num_epochs_per_iteration
            * self.num_episodes_per_iteration
            // self.global_batch_size
        )
        logger.info(f"Per device batch size: {self.args.per_device_train_batch_size}")
        logger.info(
            f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"Num of total processes: {self.distributed_state.num_processes}")
        logger.info(
            f"Global batch size (w. parallel, distributed & accumulation): {self.global_batch_size}"
        )
        logger.info(
            f"Total number of training steps (Gradient Updates): {self.total_num_training_steps}"
        )

    def get_models(
        self,
    ) -> WeakValueDictionary[str, Union[PreTrainedModel, DeepSpeedEngine]]:
        weak_dict = WeakValueDictionary()
        if getattr(self, "_actor_engine", None) is not None:
            weak_dict["actor"] = self._actor_engine

        if getattr(self, "_reference_engine", None) is not None:
            weak_dict["reference"] = self._reference_engine

        return weak_dict

    def step(self, episodes_dataset: Dataset) -> Optional[Path]:
        """
        Perform a single step of PPO training. A Single step of policy training amounts to possibly multi epochs
        of training on the episodes.

        Distributed Note:
            This function is called on each process. i.e., they all receive a full copy of the episodes_dataset.

        Considering our conservative memory approach, here is a general idea of each step:
        1. Initialize (load into CPU/GPU) the reference model if it exists.
        2. Compute the reference log probabilities if the reference model exists.
        3. Remove the reference model from the memory.
        4. Initialize (load into CPU/GPU ) the actor model and its optimizer.
        6. Load the checkpoint if needed (i.e. self.checkpoint_path_to_load is not None).
        9. Remove the actor from the memory. (Including the optimizer states)
        10. (Outside of this function) Generate new episodes by sampling from the actor.
        11. (Outside of this function) go back to step 1.

        Args:
            episodes_dataset (Dataset):
                A HuggingFace Dataset containing the episodes to train on.
                It should have the following columns:
                    - "query_token_ids": The token ids of the query.
                    - "response_token_ids": The token ids of the response.
                    - "score": The reward of the response (single scalar per response)
                    - "advantages": The advantages of the response. (Optional)

        Returns:
            Optional[Path]:
                The path to the latest policy (actor) checkpoint.
        """
        episodes_dataset = self._filter_episodes(episodes_dataset)
        if self._is_kl_penalty_enabled():
            # Compute or reload from disk the episodes with reference log probabilities
            # It takes care initializing and destroying the reference model
            episodes_dataset = self._get_episodes_w_ref_logps(episodes_dataset)

        # Initialize the actor models along with their optimizers
        logger.info("Initializing the actor model.")
        actor_engine = self._init_actor_model()

        # Load from checkpoint if specified
        need_to_save_temp_checkpoint = not self.cache_deepspeed_engines
        if self.checkpoint_path_to_load is not None:
            logger.info(f"Loading checkpoint from {self.checkpoint_path_to_load}...")
            self._load_checkpoint_to_ds_engines(
                self.checkpoint_path_to_load, actor_engine
            )
            self.checkpoint_path_to_load = None
        elif not need_to_save_temp_checkpoint:
            logger.info(f"Resuming from {self.state.state_dict()}")
        else:
            assert (
                self.state.state_dict() == self.state.INITIAL_STATE_DICT
            ), f"State should be INITIAL. Got: {self.state.state_dict()}"
            logger.info("No checkpoint to load. Training will start from scratch.")
        
        # Compute or reload from disk the episodes with current actor log probabilities and values
        episodes_dataset = self._get_episodes_w_curr_logps_and_values(episodes_dataset, actor_engine)
        
        # Train the actor models using PPO
        self._train_actor(episodes_dataset, actor_engine)
        
        # Save the models' state if needed
        should_save_full_ckpt = (
            self.args.save_steps != -1
            and self.state.iteration % self.args.save_steps == 0
        )
        temp_ckpt_path = (
            self.temp_checkpoint_dir / self._get_automatic_checkpoint_name()
        )
        if need_to_save_temp_checkpoint:
            self._save_checkpoint(temp_ckpt_path, actor_engine)
            self.checkpoint_path_to_load = temp_ckpt_path
            if should_save_full_ckpt:
                self._copy_to_permanent_storage(temp_ckpt_path)
        else:
            # Just save the actor for inference
            self._save_hf_pretrained(actor_engine, temp_ckpt_path / "hf_pretrained")
            if should_save_full_ckpt:
                self._save_automatic_checkpoint(actor=actor_engine)
        self._clean_old_temp_checkpoints(exclude=[temp_ckpt_path])
        self._clean_episodes(exclude=[temp_ckpt_path.name])

        # Clean up models and their optimizers from memory
        see_memory_usage("Before cleaning up deepspeed from memory", force=True)
        self._destroy_ds_engine(actor_engine)
        del actor_engine
        release_memory()
        see_memory_usage("After cleaning up deepspeed from memory", force=True)
        if not self.cache_deepspeed_engines:
            self.log_tensors_on_gpu()

        path_to_latest_policy = temp_ckpt_path / "hf_pretrained"
        return path_to_latest_policy

    def _train_actor(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine
    ):
        """
        Train the actor models using PPO.

        Args:
            episodes (Dataset):
                The episodes to train on (possibly with reference log probabilities).
            actor (DeepSpeedEngine):
                The actor model to train.
        """
        kls = self._log_episodes_metrics(episodes)

        # Step 2: The actual PPO training loop
        dataloader = prepare_data_loader_for_training(
            episodes,
            per_device_batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        steps_in_epoch = len(dataloader)
        optim_steps_in_epoch = steps_in_epoch // self.args.gradient_accumulation_steps
        optim_steps_in_epoch = max(optim_steps_in_epoch, 1)
        num_optimization_steps_in_iteration = (
            self.num_epochs_per_iteration * optim_steps_in_epoch
        )
        total_num_optimization_steps = (
            self.num_iterations * num_optimization_steps_in_iteration
        )

        logger.info(f"***** Running a GRPO training step: {self.state.iteration}  *****")

        logger.info(f"  Num Episodes = {len(episodes):,}")
        logger.info(f"  Num Epochs Per Iteration = {self.num_epochs_per_iteration:,}")
        logger.info(f"  Num Dataloader Steps in an Epoch = {steps_in_epoch:,}")
        logger.info(f"  Num Optim. steps in an Epoch = {optim_steps_in_epoch:,}")
        logger.info(
            f"  Num Optim. steps in an iteration "
            f"(#epoch x #optim_step_per_epoch) = {num_optimization_steps_in_iteration:,}"
        )
        logger.info(
            f"  Total Num Optim. steps (#iteration x #epoch x #optim_step_per_epoch) "
            f"= {total_num_optimization_steps:,}"
        )
        logger.info(f"  World Size = {actor.world_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Per device batch size = {self.args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Global batch size (w. parallel, distributed & accumulation) = {self.global_batch_size}"
        )
        logger.info(f"  -------- Model Parameters --------")

        actor_num_trainable_params = get_model_param_count(actor, trainable_only=True)
        logger.info(
            f"  Number of trainable parameters (actor) = {actor_num_trainable_params:,}"
        )
        if self._can_log_to_cloud():
            self.cloud_logger.summary["actor/num_trainable_params"] = (
                actor_num_trainable_params
            )

        logger.info(f"  ---------------------------------")
        logger.info(f"  Current Global Step = {self.state.global_step}")

        # Create a new dataloader iterator
        dataloader_iter = iter(dataloader)

        # Check if we're resuming training in the middle of an iteration
        completed_optim_steps_in_this_iteration = (
            self.state.global_step % num_optimization_steps_in_iteration
        )
        assert (
            completed_optim_steps_in_this_iteration == 0
        ), "We don't support resuming training in the middle of an iteration. "

        progress_bar = tqdm(
            total=total_num_optimization_steps,
            disable=not self._is_main_process(),
            desc=f"Iteration {self.state.iteration}: Training",
            dynamic_ncols=True,
        )
        progress_bar.update(self.state.global_step)

        globalstep_last_logged = self.state.global_step

        actor.train()
        
        running_metrics = {}
        accumulated_metrics = {}

        dist.barrier()

        starting_epoch = 0
        for epoch in range(starting_epoch, self.num_epochs_per_iteration):
            for step, inputs in enumerate(dataloader_iter):
                # Store the grad_acc_boundary before engine.step() is called
                # as the engine.step() will reset the grad_acc_boundary
                is_grad_acc_boundary = actor.is_gradient_accumulation_boundary()

                # Perform the training step, LR scheduler step, zero_grad, and accumulation of gradients
                # noinspection PyTypeChecker
                metrics = self._training_step(inputs, actor)

                self._update_metrics(running_metrics, accumulated_metrics, metrics)

                if is_grad_acc_boundary:
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    progress_bar.update(1)

                    should_log = self.state.global_step % self.args.logging_steps == 0
                    if should_log:
                        self._log_training_metrics(
                            globalstep_last_logged,
                            accumulated_metrics=accumulated_metrics,
                            progress_bar=progress_bar,
                            actor=actor
                        )
                        globalstep_last_logged = self.state.global_step

            # Recreate the dataloader iterator
            dataloader_iter = iter(dataloader)

        dist.barrier()

        for key, value in running_metrics.items():
            value = torch.tensor(value, device=actor.device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value = value.cpu().item() / dist.get_world_size()
            running_metrics[key] = value

        if len(running_metrics) > 0:
            logger.info(f"Running metrics: {running_metrics}")

        if self._is_kl_penalty_enabled():
            assert isinstance(kls, float)
            self.kl_ctl.update(kls, self.num_episodes_per_iteration)

        self.state.iteration += 1

        progress_bar.close()

    def _training_step(
        self,
        inputs: Dict[str, torch.Tensor],
        actor: DeepSpeedEngine
    ) -> Dict[str, Union[float, torch.Tensor]]:
        
        inputs = {k: v.to(actor.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]  # Shape: (batch_size, max_seq_len)
        attention_mask = inputs["attention_mask"]  # Shape: (batch_size, max_seq_len)
        labels = inputs["labels"]  # Shape: (batch_size, max_seq_len)
        
        shifted_labels = labels[
            ..., 1:
        ].contiguous()  # Shape: (batch_size, max_seq_len-1)
        shifted_labels_mask = (shifted_labels != -100).to(
            attention_mask.dtype
        )  # Shape: (batch_size, max_seq_len-1)
        
        # Note that this is the log probability of the actor model
        # in the beginning of this iteration (aka the old log probs)
        shifted_actor_logprobs = inputs[COLUMN_ACTOR_SHIFTED_LOGPS]  # Shape: (batch_size, max_seq_len-1)
        assert shifted_actor_logprobs.shape == shifted_labels_mask.shape
        
        # Step 1: Compute the rewards, advantages, and returns
        with torch.no_grad():
            if self._is_kl_penalty_enabled():
                shifted_ref_logprobs = inputs[COLUMN_REF_SHIFTED_LOGPS]
            else:
                shifted_ref_logprobs = None
            
            kls = self._compute_kl(shifted_actor_logprobs, shifted_ref_logprobs)
            
            assert "advantages" in inputs
            precomputed_advantages = inputs["advantages"]  # Shape: (batch_size, max_seq_len)
            
            # Shift the advantages to left to match the actions
            advantages = precomputed_advantages[:, 1:]  # Shape: (batch_size, max_seq_len-1)
            returns = None
            
            assert advantages.shape == shifted_actor_logprobs.shape
        
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Step 2: Compute the policy/actor loss
        actor_loss, is_skipped, actor_metrics, approx_ref_kl = self._compute_actor_loss(
            actor,
            model_inputs=model_inputs,
            shifted_labels_mask=shifted_labels_mask,
            old_logprobs=shifted_actor_logprobs,
            ref_logprobs=shifted_ref_logprobs,
            advantages=advantages,
        )
        actor.backward(actor_loss)
        self._check_overflow(actor)
        actor.step()
        # Get rid of actor's activations to free up memory
        actor_loss = actor_loss.detach().clone()
        release_memory()
        
        metrics = {
            "advantages/mean": masked_mean(advantages, shifted_labels_mask).detach(),
            "advantages/std": (
                masked_var(advantages, shifted_labels_mask).detach().sqrt()
            ),
            "advantages/close_to_zero_perc": close_to_zero_percentage(
                advantages, shifted_labels_mask, threshold=1e-8
            ).detach(),
            "num_tokens": shifted_labels_mask.sum().detach(),
            "_num_participating_tokens": shifted_labels_mask.sum().detach(),
            **actor_metrics
        }
        if returns is not None:
            metrics["returns"] = masked_mean(returns, shifted_labels_mask).detach()
        if kls is not None or approx_ref_kl is not None:
            if approx_ref_kl is not None:
                kls = approx_ref_kl
            
            metrics["kls"] = (kls * shifted_labels_mask).sum(dim=1).mean().detach()
            metrics["kl_coef"] = self.kl_ctl.value
        
        metrics["actor/loss"] = actor_loss
        metrics["actor/grad_norm"] = actor.get_global_grad_norm()
        
        return metrics

    def _compute_actor_loss(
        self,
        actor: DeepSpeedEngine,
        model_inputs: Dict[str, torch.Tensor],
        shifted_labels_mask: torch.LongTensor,
        old_logprobs: torch.FloatTensor,
        ref_logprobs: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, bool, Dict[str, torch.Tensor], Optional[torch.FloatTensor]
    ]:
        """
        Compute the actor loss using PPO.
        
        Args:
            actor (`DeepSpeedEngine`):
                The actor model.
            model_inputs (`Dict[str, torch.Tensor]`):
                The model inputs for the actor model. Contains the following keys:
                - "input_ids": The input token ids, shape (`batch_size`, `max_seq_len`).
                - "attention_mask": The attention mask, shape (`batch_size`, `max_seq_len`).
                - "labels": The labels, shape (`batch_size`, `max_seq_len`).
            shifted_labels_mask (`torch.LongTensor`):
                The shifted labels mask (aka action_mask), shape (`batch_size`, `max_seq_len-1`).
            old_logprobs (`torch.FloatTensor`):
                The log probabilities of the actor model for the previous iteration,
                shape (`batch_size`, `max_seq_len-1`).
            advantages (`torch.FloatTensor`):
                The advantages of the responses, shape (`batch_size`, `max_seq_len-1`).
        
        Returns:
            `torch.FloatTensor`: The actor loss.
            `bool`: Whether the batch was skipped.
            `Dict[str, torch.Tensor]`: Metrics from the training step.
            `Optional[torch.FloatTensor]`: The approx_kls tensor.
        """
        # Switch to RL terminology for more clarity
        action_mask = shifted_labels_mask  # Shape: (batch_size, max_seq_len-1)
        
        # Compute the log probabilities of the actor
        outputs = self._forward_pass_actor(
            actor,
            model_inputs,
            return_all_logps=True,
            return_entropy=self.report_entropy,
        )
        logprobs = outputs["all_logps"]  # Shape: (batch_size, seq_len-1)
        assert logprobs.shape == old_logprobs.shape
        assert action_mask.shape == logprobs.shape
        
        # Compute the PPO-clip loss
        log_ratio = (logprobs - old_logprobs) * action_mask
        ratio = torch.exp(log_ratio)
        
        pg_losses1 = -advantages * ratio
        with torch.no_grad():
            pg_losses1_anomalies = monitor_tensor_anomalies(
                pg_losses1.detach(), action_mask
            )
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.grpo_hparams.cliprange, 1.0 + self.grpo_hparams.cliprange
        )
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)
        
        if self.grpo_hparams.kl_penalty_loss_type is not None:
            assert ref_logprobs is not None
            ref_kl = self._compute_kl_penalty(
                logprobs,
                ref_logprobs,
                estimation_type=self.grpo_hparams.kl_penalty_loss_type,
            )
            ref_kl = torch.clamp(
                ref_kl * action_mask,
                min=self.grpo_hparams.kl_penalty_loss_clip_min,
                max=self.grpo_hparams.kl_penalty_loss_clip_max,
            )
            
            ref_kl_loss = self.kl_ctl.value * ref_kl.sum(dim=1).mean()
            pg_loss = pg_loss + ref_kl_loss
            ref_kl = ref_kl.detach()
        else:
            ref_kl = None
            ref_kl_loss = None
        
        is_skipped = False
        avg_ratio = masked_mean(ratio, action_mask)
        if avg_ratio.item() > self.grpo_hparams.ratio_threshold:
            logger.warning(
                f"High PPO ratio detected: {avg_ratio.item():.2f}. Skipping this batch."
            )
            pg_loss = pg_loss * 0.0
            is_skipped = True
        
        pg_clip_frac = masked_mean(
            torch.gt(pg_losses2, pg_losses1).float(), action_mask
        )
        approx_kl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, action_mask)
        policy_kl = masked_mean(old_logprobs - logprobs, action_mask)
        
        metrics = {
            "actor/approx_kl": approx_kl.detach(),
            "actor/policy_kl": policy_kl.detach(),
            "actor/clip_frac": pg_clip_frac.detach(),
            "actor/ratio": avg_ratio.detach(),
            **{
                f"actor/pg_losses1_anomalies__{i}": v
                for i, v in pg_losses1_anomalies.items()
            },
        }
        if "entropy" in outputs:
            metrics["actor/logit_entropy"] = outputs["entropy"].detach()
        if ref_kl_loss is not None:
            metrics["actor/ref_kl_loss"] = ref_kl_loss.detach()
        
        return pg_loss, is_skipped, metrics, ref_kl

    def _compute_kl(
        self,
        shifted_actor_logprobs: torch.FloatTensor,
        shifted_ref_logprobs: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute per token rewards from scores and KL-penalty.
        
        Args:
            scores (`torch.FloatTensor`):
                Scores from the episodes; one scalar per episode, shape (`batch_size`)
            shifted_actor_logprobs (`torch.FloatTensor`):
                Log probabilities of the actor, shape (`batch_size`, `max_seq_len-1`)
            shifted_ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `max_seq_len-1`)
            attention_mask (`torch.LongTensor`):
                Mask for the input, shape (`batch_size`, `max_seq_len`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: Non-score rewards, shape (`batch_size`, `max_seq_len-1`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `max_seq_len-1`)
        """
        if (
            shifted_ref_logprobs is not None
            and self.grpo_hparams.kl_penalty_loss_type is None
        ):
            kl = self._compute_kl_penalty(shifted_actor_logprobs, shifted_ref_logprobs)
        else:
            # KL penalty is not part of the reward
            kl = None

        if kl is not None:
            kl = kl.detach()

        return kl

    def _compute_kl_penalty(
        self,
        logprob: Union[torch.FloatTensor, np.ndarray],
        ref_logprob: Union[torch.FloatTensor, np.ndarray],
        estimation_type: Optional[str] = None,
    ) -> Union[torch.FloatTensor, np.ndarray]:
        """
        Compute the per-token KL penalty between the log probabilities of the actor and the reference model.

        Args:
            logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the actor, shape (`batch_size`, T)
            ref_logprob (`Union[torch.FloatTensor, np.ndarray]`):
                Log probabilities of the reference model, shape (`batch_size`, T)

        Returns:
            `Union[torch.FloatTensor, np.ndarray]`: KL penalty, shape (`batch_size`, `T`)
        """

        if estimation_type is None:
            estimation_type = self.grpo_hparams.kl_penalty

        if estimation_type == "kl":
            return logprob - ref_logprob

        if estimation_type == "abs":
            return (logprob - ref_logprob).abs()

        if estimation_type == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if estimation_type == "control_variate":
            # Compute the per-token approximate KL penalty between the log probabilities of the actor
            # and the reference model as suggested by Schulman in http://joschu.net/blog/kl-approx.html
            #
            # D_KL [π_θ || π_ref] =
            #    π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t) - log(π_ref(y_t | x, y_<t) / π_θ(y_t | x, y_<t)) - 1
            #

            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                kl = torch.exp(log_ratio) - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                kl = np.exp(log_ratio) - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "seq_control_variate":
            log_ratio = ref_logprob - logprob
            if isinstance(log_ratio, torch.Tensor):
                prob_ratio = torch.exp(log_ratio.sum(dim=-1, keepdim=True))
                kl = prob_ratio - log_ratio - 1
            elif isinstance(log_ratio, np.ndarray):
                prob_ratio = np.exp(log_ratio.sum(axis=-1, keepdims=True))
                kl = prob_ratio - log_ratio - 1
            else:
                raise ValueError("Unsupported type for log_ratio.")
            return kl

        if estimation_type == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(
                ref_logprob, logprob, log_target=True, reduction="none"
            ).sum(-1)

        raise NotImplementedError

    log_keys_to_store_in_running_metrics = [
        "_num_participating_tokens",
    ]

    log_keys_weighed_by_num_participating_tokens = [
        "advantages/mean",
        "advantages/std",
        "rewards/mean",
        "returns",
        "non_score_rewards",
        "actor/loss",
        "actor/logit_entropy",
        "actor/approx_kl",
        "actor/policy_kl",
        "actor/clip_frac",
        "ratio"
    ]

    def _log_episodes_metrics(self, episodes: Dataset) -> Optional[float]:
        """
        Log scores, advantages, logprobs, and values of the episodes.

        Args:
            episodes (Dataset): The episodes dataset.

        Returns:
            Optional[float]: The KL from reference policy
        """
        if len(episodes) == 0:
            return

        def compute_seq_logp(
            episode: Dict[str, Any], logprobs_w_query: List[float]
        ) -> float:
            query_len = len(episode["query_token_ids"])
            logprobs = logprobs_w_query[query_len - 1 :]
            seq_logprob = sum(logprobs)
            return seq_logprob
        
        scores = []
        response_lengths = []
        advantages = []
        ref_logprobs = []
        actor_logprobs = []
        kls = []
        control_variate_kls = []
        for e in episodes:
            scores.append(e["scores"])
            response_lengths.append(len(e["response_token_ids"]))
            if "advantages" in e:
                advantages += e["advantages"]
            if COLUMN_REF_SHIFTED_LOGPS in e:
                ref_logprobs.append(compute_seq_logp(e, e[COLUMN_REF_SHIFTED_LOGPS]))
            if COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_logprobs.append(
                    compute_seq_logp(e, e[COLUMN_ACTOR_SHIFTED_LOGPS])
                )
            if COLUMN_REF_SHIFTED_LOGPS in e and COLUMN_ACTOR_SHIFTED_LOGPS in e:
                actor_lp = np.array(e[COLUMN_ACTOR_SHIFTED_LOGPS])
                ref_lp = np.array(e[COLUMN_REF_SHIFTED_LOGPS])
                kl = self._compute_kl_penalty(actor_lp, ref_lp).sum()
                kls.append(kl)

                # This is unbiased & low variance
                control_variate_kl = self._compute_kl_penalty(
                    actor_lp, ref_lp, estimation_type="control_variate"
                ).sum()
                control_variate_kls.append(control_variate_kl)

            if COLUMN_VALUES in e:
                values = e[COLUMN_VALUES]
                values_without_query = values[
                    len(e["query_token_ids"]) - 1 : -1
                ]  # Skip the last token (</s>)
                if len(values_without_query) == 0:
                    logger.warning(
                        f"Empty values for episode: {json.dumps(e, indent=2)}"
                    )

        scores = np.array(scores)
        response_lengths = np.array(response_lengths)
        actor_logprobs = np.array(actor_logprobs)
        metrics = {
            "scores/mean": np.mean(scores),
            "scores/std": np.std(scores),
            "scores/dist": scores,
            "response_lengths/mean": np.mean(response_lengths),
            "response_lengths/std": np.std(response_lengths),
            "response_lengths/dist": response_lengths,
            "actor_logprobs/sum": np.mean(actor_logprobs),
            "actor_logprobs/normalized_by_response_len": np.mean(
                actor_logprobs / response_lengths
            ),
            "actor_logprobs/dist": actor_logprobs,
        }

        if len(kls) > 0:
            kls = np.array(kls)
            metrics["kls/mean"] = np.mean(kls)
            metrics["kls/dist"] = kls
            kls = float(metrics["kls/mean"])
        else:
            kls = None

        if len(control_variate_kls) > 0:
            control_variate_kls = np.array(control_variate_kls)
            metrics["kls/crtl_var__mean"] = np.mean(control_variate_kls)
            metrics["kls/crtl_var__dist"] = control_variate_kls

        if len(advantages) > 0:
            advantages = np.array(advantages)
            metrics["advantages/mean"] = np.mean(advantages)
            metrics["advantages/std"] = np.std(advantages)
            metrics["advantages/dist"] = advantages

        if len(ref_logprobs) > 0:
            ref_logprobs = np.array(ref_logprobs)
            metrics["ref_logprobs/sum"] = np.mean(ref_logprobs)
            metrics["ref_logprobs/normalized_by_response_len"] = np.mean(
                ref_logprobs / response_lengths
            )
            metrics["ref_logprobs/dist"] = ref_logprobs

        non_array_metrics = {
            k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)
        }
        logger.info(f"Episode Metrics: {non_array_metrics}")

        logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
        self._cloud_log(
            {
                **logs,
                "train/global_step": self.state.global_step,
                "train/global_iteration": self.state.iteration,
            }
        )

        return kls

    def _log_training_metrics(
        self,
        _globalstep_last_logged: int,
        accumulated_metrics: Dict[str, Union[float, torch.Tensor]],
        progress_bar: tqdm,
        actor: DeepSpeedEngine
    ):
        # Wait for all processes to reach this point
        dist.barrier()

        logs: Dict[str, float] = {}
        
        # Compute the log values over all processes
        num_steps_since_last_log = (
            self.state.global_step - _globalstep_last_logged
        ) * self.args.gradient_accumulation_steps

        if "_num_participating_tokens" in accumulated_metrics:
            num_participating_tokens = accumulated_metrics["_num_participating_tokens"]
            dist.all_reduce(num_participating_tokens, op=dist.ReduceOp.SUM)
            num_participating_tokens = num_participating_tokens.item()
        else:
            num_participating_tokens = 1

        for metric_name, metric_value in accumulated_metrics.items():
            if metric_name.startswith("_"):
                continue
            if metric_value is None:
                continue

            is_weighed_by_num_actions = (
                metric_name in self.log_keys_weighed_by_num_participating_tokens
            )

            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.to(actor.device)
                dist.all_reduce(metric_value, op=dist.ReduceOp.SUM)
                metric_value = metric_value.item()
                divisor = dist.get_world_size()
            else:
                assert not is_weighed_by_num_actions
                divisor = 1
            
            if is_weighed_by_num_actions:
                metric_value /= num_participating_tokens
            else:
                metric_value /= divisor * num_steps_since_last_log
            
            logs[metric_name] = round(metric_value, 8)

        logs["actor/lr"] = self._get_learning_rate(actor)
        
        logs["epoch"] = round(self.state.epoch, 4)
        logs["step"] = self.state.global_step
        logs["actor/ds_step"] = actor.global_steps
        
        # First log the metrics on the progress bar
        progress_bar.set_postfix(logs)

        # Add "train/" prefix for clarity.
        logs = {f"train/{k}": v for k, v in logs.items()}

        self._cloud_log({**logs, "train/global_step": self.state.global_step})

        # Reset the accumulated metrics
        for key in accumulated_metrics.keys():
            accumulated_metrics[key] -= accumulated_metrics[key]

    def _update_metrics(
        self,
        running_metrics: Dict[str, Union[torch.Tensor, float]],
        accumulated_metrics: Dict[str, Union[torch.Tensor, float]],
        step_metrics: Dict[str, Union[torch.Tensor, float]],
    ):
        dist.barrier()

        def get_initial_value(
            val: Union[float, torch.Tensor]
        ) -> Union[float, torch.Tensor]:
            if isinstance(val, torch.Tensor):
                return torch.tensor(0.0, dtype=val.dtype, device=val.device)
            return 0.0

        # Initialize running metrics if not already initialized
        for key in step_metrics.keys():
            if key in accumulated_metrics:
                continue
            accumulated_metrics[key] = get_initial_value(step_metrics[key])
            if key in self.log_keys_to_store_in_running_metrics:
                if key not in running_metrics:
                    running_metrics[key] = get_initial_value(step_metrics[key])

        num_tokens = step_metrics["_num_participating_tokens"].item()

        for key, value in step_metrics.items():
            if value is None:
                continue

            if key in self.log_keys_weighed_by_num_participating_tokens:
                weight = num_tokens
            else:
                weight = 1

            value = value * weight
            accumulated_metrics[key] += value

        # Update Running Metrics
        running_metrics["_num_participating_tokens"] += num_tokens

    def _get_episodes_w_ref_logps(self, episodes: Dataset) -> Dataset:
        logger.info(f"Computing the reference log probabilities.")

        ds_w_ref_logprobs_path = (
            self.checkpoints_dir
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_refLogp"
        )

        # Initialize and use the reference model to compute log probabilities for the dataset
        ref_engine = self._init_reference_model()
        t0 = time.time()
        aug_ds = self._update_episodes_with_log_probs(
            ref_engine, episodes, COLUMN_REF_SHIFTED_LOGPS
        )
        self._cloud_log(
            {
                "timing/train/computing_ref_logprobs": time.time() - t0,
                "train/global_step": self.state.global_step,
            }
        )

        if self._is_main_process():
            # We reread the dataset from disk afterward. This is done to
            # ensure that the dataset is memory-mapped and that the reference
            # log probs are not actually loaded in CPU memory.
            aug_ds.save_to_disk(ds_w_ref_logprobs_path)
        self.distributed_state.wait_for_everyone()

        self._destroy_reference_engine(ref_engine)
        del ref_engine
        del aug_ds
        release_memory()
        if not self.cache_deepspeed_engines:
            self.log_tensors_on_gpu()

        episodes = Dataset.load_from_disk(str(ds_w_ref_logprobs_path))
        return episodes

    def _get_episodes_w_curr_logps_and_values(
        self,
        episodes: Dataset,
        actor: DeepSpeedEngine,
    ) -> Dataset:
        metrics = {}

        logger.info(f"Computing the current actor logprobs.")
        ds_w_actor_logps_path = (
            self.checkpoints_dir
            / f"episodes__iter{self.state.iteration:04d}"
            / "w_actLogp"
        )
        t0 = time.time()
        aug_ds = self._update_episodes_with_log_probs(
            actor, episodes, COLUMN_ACTOR_SHIFTED_LOGPS
        )
        metrics["timing/train/computing_actor_logprobs"] = time.time() - t0
        if self._is_main_process():
            aug_ds.save_to_disk(ds_w_actor_logps_path)
        self.distributed_state.wait_for_everyone()

        del aug_ds
        release_memory()

        episodes = Dataset.load_from_disk(str(ds_w_actor_logps_path))

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

        return episodes

    def _update_episodes_with_log_probs(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        dataset: Dataset,
        column_name: str,
    ) -> Dataset:
        # Create a distributed data loader such that the order of
        # episodes is preserved when batched are distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        model_engine.eval()

        dist.barrier()

        list_of_log_probs = []
        for inputs in tqdm(
            data_loader, desc="Computing log probs", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.args.per_device_train_batch_size
                ), (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.args.per_device_train_batch_size}."
                )

                inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}

                # Compute the sequence lengths as we need to extract
                # the log probs of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)  # Shape: (batch_size, 1)

                # Compute the log probabilities for each token
                outputs = self._forward_pass_actor(
                    model_engine, inputs, return_all_logps=True
                )
                logps = outputs["all_logps"].detach()
                assert logps.shape[1] == inputs["input_ids"].shape[1] - 1

                # Gather across all distributed processes
                # Note that after all_gather, the order in which the batches were
                # distributed across processes is preserved. So, concatenating
                # them along the batch dimension will give us the original order.
                seq_lengths = gather(seq_lengths).cpu()
                logps = gather(
                    pad_across_processes(logps, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                assert (
                    logps.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2d tensors to a list of lists
                logps_seq_lengths = seq_lengths - 1
                for i, seq_len in enumerate(logps_seq_lengths.squeeze().tolist()):
                    assert seq_len <= logps.shape[1]
                    list_of_log_probs.append(logps[i, :seq_len].tolist())

        # Remove any extra log probs that were added due to padding
        list_of_log_probs = list_of_log_probs[: len(dataset)]

        with self.distributed_state.main_process_first():
            dataset = dataset.add_column(name=column_name, column=list_of_log_probs)
        return dataset

    def _update_episodes_with_values(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        dataset: Dataset,
        column_name: str,
    ) -> Dataset:
        # Create a distributed data loader such that the order of
        # episodes is preserved when distributed across multiple processes.
        data_loader = prepare_data_loader_for_inference(
            dataset,
            per_device_batch_size=self.args.per_device_train_batch_size,
            data_loader_kwargs={
                "collate_fn": PPODataCollator(),
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            },
        )

        model_engine.eval()

        dist.barrier()

        list_of_values = []
        for inputs in tqdm(
            data_loader, desc="Computing values", disable=not self._is_main_process()
        ):
            with torch.no_grad():
                # Assume every sequence is padded from the right
                # noinspection DuplicatedCode
                assert torch.all(inputs["attention_mask"][:, 0] == 1)
                assert (
                    inputs["input_ids"].shape[0]
                    == self.args.per_device_train_batch_size
                ), (
                    f"We expect on all processes to have the same batch size of "
                    f"{self.args.per_device_train_batch_size}."
                )
                
                inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}
                
                # Compute the sequence lengths as we need to extract
                # the values of the non-padded tokens
                seq_lengths = inputs["attention_mask"].sum(dim=1).detach().clone()
                seq_lengths = seq_lengths.unsqueeze(1)
                
                # Compute the values for each token
                assert values.shape[1] == inputs["input_ids"].shape[1]

                # Gather across all distributed processes
                seq_lengths = gather(seq_lengths).cpu()
                values = gather(
                    pad_across_processes(values, dim=1, pad_index=0.0, pad_first=False)
                ).cpu()

                assert (
                    values.shape[0]
                    == inputs["input_ids"].shape[0] * dist.get_world_size()
                )

                # Convert 2d tensors to a list of lists
                for i, seq_len in enumerate(seq_lengths.squeeze().tolist()):
                    assert seq_len <= values.shape[1]
                    list_of_values.append(values[i, :seq_len].tolist())

        # Remove any extra values that were added due to padding
        list_of_values = list_of_values[: len(dataset)]

        with self.distributed_state.main_process_first():
            dataset = dataset.add_column(name=column_name, column=list_of_values)
        return dataset

    def _forward_pass_actor(
        self,
        model_engine: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False,
        return_sequence_logp: bool = False,
        return_all_logps: bool = False,
        return_entropy: bool = False,
        sequence_logp_reduction: Optional[Literal["mean"]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for the model.

        Args:
            model_engine (Union[DeepSpeedEngine, PreTrainedModel]): The model to forward pass.
            inputs (Dict[str, torch.Tensor]): The inputs to the model, containing the following keys:
                - "input_ids": The input ids of the sequence.
                - "labels": The labels for the sequence.
                - "attention_mask": The attention mask of the sequence.
        Returns:
            Dict[str, Any]: The outputs containing the following keys:
                - "logits": The logits for the sequence.
                - "logps": The log probabilities of the sequence.
                - "all_logps": The log probabilities of all tokens in the sequence.
        """
        input_ids: torch.Tensor = inputs["input_ids"]
        labels: torch.Tensor = inputs["labels"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        outputs: CausalLMOutputWithPast = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        logits = outputs.logits.float()  # Shape: (batch_size, max_seq_len, vocab_size)
        logits /= self.grpo_hparams.temperature

        # Shift so that tokens < n predict n
        # noinspection DuplicatedCode
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_label_mask = (shift_labels != -100).to(shift_logits.dtype)

        # Make sure all label indices are valid. i.e. convert -100 to 0
        shift_labels[shift_labels == -100] = 0

        log_probs = shift_logits.log_softmax(-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        )
        per_token_log_probs = per_token_log_probs.squeeze(2)

        # Multiply the log probs by the label mask to ignore the padding labels
        per_token_log_probs = per_token_log_probs * shift_label_mask

        output = {}
        if return_entropy:
            with torch.no_grad():
                mean_entropy = masked_mean(
                    entropy_from_logits(shift_logits.detach()), shift_label_mask
                )
                mean_entropy = mean_entropy.detach().clone()
                output["entropy"] = mean_entropy

        if return_logits:
            output["logits"] = logits

        if return_sequence_logp:
            sequence_log_probs = per_token_log_probs.sum(dim=-1)
            if sequence_logp_reduction == "mean":
                sequence_log_probs = sequence_log_probs / shift_label_mask.sum(dim=-1)
            output["sequence_logp"] = sequence_log_probs

        if return_all_logps:
            output["all_logps"] = per_token_log_probs

        return output

    def _init_reference_model(
        self, only_return_unwrapped_model: bool = False
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_reference_engine"):
            reference_model = (
                self._reference_engine.module
                if isinstance(self._reference_engine, DeepSpeedEngine)
                else self._reference_engine
            )
            reference_model.to(self.distributed_state.device)
            return self._reference_engine

        this_process_device = self.distributed_state.device

        metrics = {}
        t0 = time.time()

        # Load the reference model into GPU
        cache_path = self.temp_checkpoint_dir / ".reference"
        if not cache_path.exists():
            cache_path = None
        # noinspection PyTypeChecker
        reference_model: PreTrainedModel = self.reference_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
            runtime_hf_model_name=cache_path,
        )
        reference_model.eval()
        metrics["timing/reference/construct"] = time.time() - t0
        if (
            self.cache_reference_model
            and cache_path is None
            and self._is_main_process()
        ):
            # Since the reference model is used in every iteration, it makes
            # sense to cache it on the fast disk to avoid loading it from network storage
            reference_model.save_pretrained(
                self.temp_checkpoint_dir / ".reference", safe_serialization=False
            )

        if only_return_unwrapped_model:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})
            if self.cache_deepspeed_engines:
                self._reference_engine = reference_model

            return reference_model

        # Using a DeepSpeed engine for inference is useful for models that are
        # too large to fit in one GPU. DeepSpeed will automatically split
        # the model across multiple GPUs.
        ds_config = HfTrainerDeepSpeedConfig(self.reference_deepspeed_config)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, reference_model.config)

        engine = self._initialize_deepspeed_engine_for_inference(
            model=reference_model, deepspeed_config=ds_config.config
        )
        engine.eval()

        if self.cache_deepspeed_engines:
            self._reference_engine = engine

        return engine

    def _init_actor_model(
        self, only_return_unwrapped_model: bool = False
    ) -> Union[DeepSpeedEngine, PreTrainedModel]:
        if hasattr(self, "_actor_engine"):
            return self._actor_engine

        logger.info(f"Creating the actor deepspeed engine...")

        this_process_device = self.distributed_state.device

        metrics = {}
        t0 = time.time()

        # Load the actor model into GPU
        # noinspection PyTypeChecker
        actor_model: PreTrainedModel = self.actor_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
        )
        metrics["timing/actor/construct"] = time.time() - t0
        if only_return_unwrapped_model:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})
            return actor_model

        if self.args.gradient_checkpointing:
            actor_model.gradient_checkpointing_enable()

        t0 = time.time()

        ds_config = HfTrainerDeepSpeedConfig(self.actor_deepspeed_config)

        # Create the optimizer
        has_optimizer = ds_config.get_value("optimizer", None) is not None
        if has_optimizer:
            weight_decay = ds_config.get_value("optimizer.params.weight_decay", 0.0)
            if weight_decay == "auto":
                weight_decay = self.args.weight_decay

            optimizer = self.create_optimizer(actor_model, weight_decay)
        else:
            optimizer = None

        # Create the LR scheduler
        # noinspection DuplicatedCode
        has_deepspeed_scheduler = ds_config.get_value("scheduler", None) is not None
        warmup_steps = self.args.get_warmup_steps(self.total_num_training_steps)
        if has_deepspeed_scheduler:
            lr_scheduler = None
            self._patch_ds_config_for_lr_scheduler(
                ds_config,
                total_num_training_steps=self.total_num_training_steps,
                warmup_steps=warmup_steps,
                learning_rate=self.args.learning_rate,
            )
        elif self.args.lr_scheduler_type is not None:
            logger.info("Using non-DeepSpeed LR scheduler.")
            lr_scheduler = self.create_lr_scheduler(
                optimizer,
                name=self.args.lr_scheduler_type,
                warmup_steps=warmup_steps,
                num_training_steps=self.total_num_training_steps,
            )
        else:
            lr_scheduler = None

        self._patch_ds_config_for_optimizer(ds_config, self.args)
        self._patch_ds_config_for_batch_size(
            ds_config, self.args, self.global_batch_size
        )
        self._patch_ds_config_for_dtype(ds_config, self.args)
        self._patch_ds_config_for_bucket_size(ds_config, actor_model.config)

        engine = self._initialize_deepspeed_engine_for_training(
            actor_model,
            deepspeed_config=ds_config.config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        metrics["timing/actor/deepspeed_init"] = time.time() - t0
        self._cloud_log({**metrics, "train/global_step": self.state.global_step})

        if self.cache_deepspeed_engines:
            self._actor_engine = engine

        return engine

    def load_checkpoint(self, checkpoint: Union[Checkpoint, Path]) -> None:
        super().load_checkpoint(checkpoint)
        checkpoint_path = (
            checkpoint if isinstance(checkpoint, Path) else checkpoint.path
        )
        self._load_training_state(checkpoint_path)
        self.checkpoint_path_to_load = checkpoint_path

    def _load_checkpoint_to_ds_engines(
        self,
        checkpoint_path: Path,
        actor: Optional[DeepSpeedEngine] = None,
    ) -> None:
        metrics = {}
        if actor is not None:
            t0 = time.time()
            actor.load_checkpoint(str(checkpoint_path / "actor"))
            metrics["timing/actor/load_checkpoint"] = time.time() - t0

        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    def _save_hf_pretrained(self, engine: DeepSpeedEngine, path: Path) -> None:
        if self._is_main_process():
            # Only save on the main process
            assert engine.zero_optimization_stage() < 3
            logger.info(f"Saving HF pretrained weights to {path}")
            unwrapped_model = engine.module
            unwrapped_model.save_pretrained(path, safe_serialization=False)
        dist.barrier()

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        actor: Optional[DeepSpeedEngine] = None
    ) -> None:
        if self._is_main_process():
            if checkpoint_path.exists():
                logger.warning(
                    f"Checkpoint path {checkpoint_path} already exists. Overwriting."
                )
                shutil.rmtree(checkpoint_path)
            
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self._save_trainer_state(checkpoint_path)
        
        metrics = {}
        if actor is not None:
            t0 = time.time()
            self._save_hf_pretrained(actor, checkpoint_path / "hf_pretrained")
            metrics["timing/actor/save_hf_pretrained"] = time.time() - t0
            
            t0 = time.time()
            actor.save_checkpoint(str(checkpoint_path / "actor"))
            metrics["timing/actor/save_checkpoint"] = time.time() - t0
        
        if len(metrics) > 0:
            self._cloud_log({**metrics, "train/global_step": self.state.global_step})

    @staticmethod
    def is_checkpoint_resumable(checkpoint_path: Path) -> bool:
        if not checkpoint_path.exists():
            return False

        if not (checkpoint_path / "actor").exists():
            return False

        model_only_file_names = ["hf_pretrained", "pytorch_model.bin", "pytorch_model"]
        non_model_files = [
            file
            for file in (checkpoint_path / "actor").iterdir()
            if file.name not in model_only_file_names
        ]
        if len(non_model_files) == 0:
            return False
        
        return True

    def clean_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        if exclude is None:
            exclude = []

        # Remove unnecessary checkpoints
        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                if self.args.checkpoint_keep_steps is not None:
                    checkpoint_iteration = self.parse_checkpoint_name(checkpoint.name)[0]
                    if checkpoint_iteration % self.args.checkpoint_keep_steps == 0:
                        continue

                    logger.info(f"Removing checkpoint {checkpoint}")
                    shutil.rmtree(checkpoint)

        self.clean_non_model_checkpoints(exclude=exclude)

    def clean_non_model_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        """
        Clean all optimizer and scheduler states which are not needed for evaluation.
        """
        if exclude is None:
            exclude = []

        def clean_deepspeed_checkpoint(path: Path) -> List[str]:
            removed_files_and_dirs = []
            for file in path.iterdir():
                if file.name not in ["hf_pretrained", "pytorch_model.bin"]:
                    removed_files_and_dirs.append(file.name)
                    if file.is_dir():
                        shutil.rmtree(file, ignore_errors=True)
                    else:
                        file.unlink()
            return removed_files_and_dirs

        for checkpoint in self.checkpoints_dir.iterdir():
            if (
                checkpoint.is_dir()
                and checkpoint.name.startswith("ckpt--")
                and checkpoint not in exclude
            ):
                logger.info(f"Removing non-model files from checkpoint {checkpoint}")

                # Remove everything except the `hf_pretrained` folder
                all_removed_files_and_dirs = []
                all_removed_files_and_dirs += clean_deepspeed_checkpoint(
                    checkpoint / "actor"
                )
                logger.info(f"Removed files and dirs: {all_removed_files_and_dirs}")

    def _clean_episodes(self, exclude: List[str]) -> None:
        if not need_to_minimize_stored_files():
            return

        if self._is_main_process():
            keep_iterations = [
                int(self.parse_checkpoint_name(ckpt.name)[0])
                for ckpt in self.checkpoints_dir.iterdir()
                if ckpt.is_dir() and ckpt.name.startswith("ckpt--")
            ]
            keep_iterations += [0]  # Always keep the initial iteration
            keep_iterations += [
                int(self.parse_checkpoint_name(name)[0]) for name in exclude
            ]
            keep_iterations = set(keep_iterations)

            # Remove unnecessary episodes
            for episode in self.checkpoints_dir.glob("episodes__iter*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("__iter")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

            # Remove unnecessary episodes insided experiment_root
            for episode in self.root_dir.glob("episodes/episodes_*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("_")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing exp_root/episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

            # Remove unnecessary temp_episodes
            for episode in self.root_dir.glob("temp_episodes/iteration__*"):
                if not episode.is_dir():
                    continue

                episode_iter = int(episode.name.split("__")[1])
                if episode_iter in keep_iterations:
                    continue

                logger.info(
                    f"Removing temp episode {episode.name}; "
                    f"excluding iterations: {keep_iterations}"
                )
                shutil.rmtree(episode, ignore_errors=True)

        dist.barrier()

    def _save_trainer_state(self, checkpoint_path: Path) -> None:
        super()._save_trainer_state(checkpoint_path)
        if self._is_main_process():
            save_custom_state(self.running_scores, checkpoint_path, index=10)
            save_custom_state(self.kl_ctl, checkpoint_path, index=11)

    def _load_training_state(self, checkpoint_path: Path) -> None:
        super()._load_training_state(checkpoint_path)
        load_custom_state(self.running_scores, checkpoint_path, index=10)
        load_custom_state(self.kl_ctl, checkpoint_path, index=11)

    def save_final_checkpoint(self) -> None:
        last_checkpoint_path = self.get_last_checkpoint().path
        final_checkpoint_path = self.checkpoints_dir / "final"
        final_checkpoint_path.write_text(last_checkpoint_path.name)

    def _copy_to_permanent_storage(self, checkpoint_path: Path) -> None:
        if not self._is_main_process():
            return

        permanent_storage_path = self.checkpoints_dir
        copy_cmd = f"cp -r {checkpoint_path} {permanent_storage_path}/"
        logger.info(f"Copying checkpoint to permanent storage: {copy_cmd}")

        # Start the copy in background
        subprocess.Popen(copy_cmd, shell=True)

    def _clean_old_temp_checkpoints(self, exclude: Optional[List[Path]] = None) -> None:
        if exclude is None:
            exclude = []

        if self._is_main_process():
            for checkpoint in self.temp_checkpoint_dir.iterdir():
                if (
                    checkpoint.is_dir()
                    and checkpoint.name.startswith("ckpt--")
                    and checkpoint not in exclude
                ):
                    logger.info(f"Removing old temp checkpoint {checkpoint}")
                    shutil.rmtree(checkpoint)

        dist.barrier()

    def _destroy_ds_engine(self, ds_engine: DeepSpeedEngine):
        if self.cache_deepspeed_engines:
            return
        super()._destroy_ds_engine(ds_engine)

    def _destroy_reference_engine(
        self, model_or_engine: Union[PreTrainedModel, DeepSpeedEngine]
    ):
        if self.cache_deepspeed_engines:
            if self.move_reference_model_to_cpu:
                model = (
                    model_or_engine.module
                    if isinstance(model_or_engine, DeepSpeedEngine)
                    else model_or_engine
                )
                model.to("cpu")
            return

        if isinstance(model_or_engine, DeepSpeedEngine):
            super()._destroy_ds_engine(model_or_engine)

    def _filter_episodes(self, episodes_dataset: Dataset) -> Dataset:
        """
        Filter out episodes that are too long.
        """
        if self.args.max_seq_len is not None:
            max_seq_len = self.args.max_seq_len
            orig_len = len(episodes_dataset)

            def filter_fn(example):
                return (
                    len(example["query_token_ids"]) + len(example["response_token_ids"])
                    <= max_seq_len
                )

            with self.distributed_state.main_process_first():
                episodes_dataset = episodes_dataset.filter(filter_fn, desc="Filtering")

            logger.error(
                f"Filtered out {orig_len - len(episodes_dataset)} episodes "
                f"that are too long. Remaining: {len(episodes_dataset)}"
            )

        return episodes_dataset

    def _check_overflow(self, actor: DeepSpeedEngine):
        assert actor.bfloat16_enabled()
        if hasattr(actor.optimizer, "check_overflow"):
            assert (
                not actor.optimizer.check_overflow()
            ), "We don't expect overflow in BF16 training"
