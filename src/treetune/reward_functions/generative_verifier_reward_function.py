import copy
import gc
import json
import logging
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil
import torch
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets
from deepspeed import get_accelerator
from deepspeed.runtime.utils import (see_memory_usage,
                                     torch_max_memory_reserved,
                                     torch_memory_reserved)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BatchEncoding, Pipeline, PreTrainedModel,
                          PreTrainedTokenizer, pipeline)

from treetune.common import Lazy
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import find_n_free_ports
from treetune.common.vllm_server import VLLMServer, compute_vllm_stats
from treetune.episode_generators.base_episode_generator import (
    Episode, EpisodeGenerator)
from treetune.episode_generators.on_policy_episode_generator import \
    OnPolicyEpisodeGenerator
from treetune.episode_generators.tree_episode_generator import TreeEpisodeUtils
from treetune.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.models import Model
from treetune.reward_functions import RewardFunction
from treetune.tasks import GSM8K, Task
from treetune.tasks.math import MATH
from treetune.tokenization_utils import Tokenizer
from treetune.trainers.policy_trainer import Checkpoint
from treetune.pipelines import Pipeline

logger = get_logger(__name__)


@RewardFunction.register("generative_verifier_reward_function")
class GenerativeVerifierRewardFunction(RewardFunction):
    def __init__(
        self,
        generation_pipeline: Lazy[Pipeline],
        value_pipeline: Lazy[Pipeline],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.generation_pipeline = generation_pipeline.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir=self.root_dir / "generation_pipeline"
        )
        self.value_pipeline = value_pipeline.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir=self.root_dir / "value_pipeline"
        )

    def batch_compute_rewards(
        self, 
        episodes: List[Episode],
        iteration: Optional[int] = None
    ) -> List[Episode]:
        if iteration is None:
            iteration = 0
        
        sequences = [
            self.value_pipeline.tokenizer.decode(e.query_token_ids + e.response_token_ids)
            for e in episodes
        ]
        
        sequences_with_reasoning_traces = self.generation_pipeline.generate(sequences)
        sequences_with_reasoning_traces = self._convert_inference_outputs_to_str(
            sequences_with_reasoning_traces
        )
        rewards = self.value_pipeline.forward(sequences_with_reasoning_traces)
        
        episodes_with_reward = [
            Episode(
                query_token_ids=e.query_token_ids,
                response_token_ids=e.response_token_ids,
                query_text=e.query_text,
                response_text=e.response_text,
                scores=reward if e.scores is None else e.scores,
            )
            for reward, e in zip(rewards, episodes)
        ]

        return episodes_with_reward
