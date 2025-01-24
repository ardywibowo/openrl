import gc
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
import torch
from accelerate.utils import release_memory
from datasets import Dataset
from deepspeed import get_accelerator
from deepspeed.runtime.utils import (torch_max_memory_reserved,
                                     torch_memory_reserved)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BatchEncoding, Pipeline, PreTrainedModel,
                          PreTrainedTokenizer, pipeline)

from treetune.common import Lazy
from treetune.common.logging_utils import get_logger
from treetune.episode_generators.base_episode_generator import (
    Episode, EpisodeGenerator)
from treetune.episode_generators.on_policy_episode_generator import \
    OnPolicyEpisodeGenerator
from treetune.episode_generators.tree_episode_generator import TreeEpisodeUtils
from treetune.models import Model
from treetune.reward_functions import RewardFunction
from treetune.tasks import GSM8K, Task
from treetune.tasks.math import MATH

logger = get_logger(__name__)


@RewardFunction.register("model_based_reward_function")
class ModelBasedRewardFunction(RewardFunction):
    def __init__(
        self,
        reward_model: Optional[Lazy[Model]] = None,
        reward_model_padding_side: str = "right",
        reward_pipeline_model_name: Optional[str] = None,
        reward_pipeline_task: str = "sentiment-analysis",
        reward_inference_per_device_batch_size: int = 128,
        cache_reward_model: bool = True,
        cache_reward_model_on_cpu: bool = False,
        temp_cache_dir: Optional[str] = None,
        unfinished_response_penalty: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # `reward_model` and `reward_pipeline_model_name` are mutually exclusive
        if reward_model is not None and reward_pipeline_model_name is not None:
            raise ValueError(
                "Only one of `reward_model` and `reward_pipeline_model_name` should be provided."
            )
        if reward_model is None and reward_pipeline_model_name is None:
            raise ValueError(
                "Either `reward_model` or `reward_pipeline_model_name` should be provided."
            )

        if reward_model is not None:
            assert reward_model_padding_side in ["left", "right"]

        self.reward_model_lazy = reward_model
        self.reward_model_padding_side = reward_model_padding_side
        self.reward_pipeline_model_name = reward_pipeline_model_name
        self.reward_pipeline_task = reward_pipeline_task
        self.reward_inf_per_device_batch_size = reward_inference_per_device_batch_size
        self.cache_reward_model = cache_reward_model
        self.cache_reward_model_on_cpu = cache_reward_model_on_cpu
        self.unfinished_response_penalty = unfinished_response_penalty

        if temp_cache_dir is not None:
            self.temp_model_cache_dir = Path(temp_cache_dir)
        else:
            from treetune.common.notebook_utils import get_repo_dir

            self.temp_model_cache_dir = get_repo_dir() / "temp_model_cache_dir"
            logger.info(
                f"No temporary model cache directory provided. Using {self.temp_model_cache_dir}"
            )
        self.temp_model_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_unfinished_response_penalty(self) -> float:
        return float(self.unfinished_response_penalty)

    def batch_compute_rewards(self, episodes: List[Episode]) -> List[Episode]:
        sequences = [
            self.tokenizer.decode(e.query_token_ids + e.response_token_ids)
            for e in episodes
        ]

        if self.reward_model_lazy is not None:
            rewards = self._compute_rewards_from_model(sequences)
        else:
            rewards = self._compute_rewards_from_pipeline(sequences)
        release_memory()

        episodes_with_reward = [
            Episode(
                query_token_ids=e.query_token_ids,
                response_token_ids=e.response_token_ids,
                scores=reward if e.scores is None else e.scores,
            )
            for reward, e in zip(rewards, episodes)
        ]

        return episodes_with_reward

    def _init_reward_model_pipeline(
        self,
    ) -> Pipeline:
        device = self.distributed_state.process_index
        sentiment_pipe = pipeline(
            self.reward_pipeline_task,
            model=self.reward_pipeline_model_name,
            device=device,
        )
        return sentiment_pipe

    def _init_reward_model(self) -> PreTrainedModel:
        this_process_device = self.distributed_state.device

        if hasattr(self, "_reward_model"):
            self._reward_model.to(this_process_device)
            return self._reward_model

        t0 = time.time()

        # Load the reward model into GPU
        cache_path = self.temp_model_cache_dir / ".reward_model"
        if not cache_path.exists():
            cache_path = None

        # noinspection PyTypeChecker
        reward_model: PreTrainedModel = self.reward_model_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
            runtime_hf_model_name=cache_path,
        )
        reward_model.to(this_process_device)
        self._cloud_log(
            {"timing/episode_generation/reward_model_construct": time.time() - t0}
        )

        if self.cache_reward_model and cache_path is None and self.is_main_process():
            # Since the reward model is used in every iteration, it makes
            # sense to cache it on the fast disk to avoid loading it from network storage
            cache_path = self.temp_model_cache_dir / ".reward_model"
            reward_model.save_pretrained(cache_path, safe_serialization=False)

        if self.cache_reward_model_on_cpu:
            self._reward_model = reward_model

        return reward_model

    def _compute_rewards_from_pipeline(self, sequences: List[str]) -> List[float]:
        reward_pipeline = self._init_reward_model_pipeline()
        pipe_outputs = reward_pipeline(
            sequences, return_all_scores=True, function_to_apply="none", batch_size=128
        )
        rewards = [
            (
                pos["score"].cpu().item()
                if isinstance(pos["score"], torch.Tensor)
                else pos["score"]
            )
            for _, pos in pipe_outputs
        ]
        del reward_pipeline
        release_memory()
        return rewards

    def _compute_rewards_from_model(self, sequences: List[str]) -> List[float]:
        reward_model = self._init_reward_model()
        reward_model.eval()

        # noinspection PyTypeChecker
        tokenizer: PreTrainedTokenizer = self.tokenizer
        tokenizer.padding_side = self.reward_model_padding_side

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0

        def collate_fn(examples: List[Dict[str, Any]]) -> BatchEncoding:
            return tokenizer(
                [e["text"] for e in examples],
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

        dataloader = DataLoader(
            Dataset.from_dict({"text": sequences}),
            batch_size=self.reward_inf_per_device_batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        has_logged = False

        all_rewards = []
        for inputs in tqdm(dataloader, desc="Computing rewards"):
            with torch.no_grad():
                if self.is_main_process() and not has_logged:
                    decoded = tokenizer.decode(
                        inputs["input_ids"][0], skip_special_tokens=False
                    )
                    logger.info(f"Decoded input: {decoded}")
                    has_logged = True

                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                outputs = reward_model(**inputs)

                # Extract the rewards from the last token
                if self.reward_model_padding_side == "right":
                    assert torch.all(
                        inputs["attention_mask"][:, 0] == 1
                    ), "Reward model expect the padding to be done on the right side."
                    # Compute the index of last token in the sequence lengths
                    seq_lengths = inputs["attention_mask"].sum(dim=1)
                    last_token_indices = seq_lengths - 1
                    rewards = outputs[range(outputs.shape[0]), last_token_indices]
                elif self.reward_model_padding_side == "left":
                    assert torch.all(
                        inputs["attention_mask"][:, -1] == 1
                    ), "Reward model expect the padding to be done on the left side."
                    rewards = outputs[:, -1]
                else:
                    raise ValueError(
                        f"Invalid padding side: {self.reward_model_padding_side}"
                    )

                all_rewards.extend(rewards.float().cpu().numpy().tolist())

        assert len(all_rewards) == len(sequences)

        if self.cache_reward_model_on_cpu:
            reward_model.to("cpu")

        return all_rewards
