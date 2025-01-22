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

@Pipeline.register("forward_pipeline")
class ForwardPipeline(Pipeline):
    def __init__(
        self,
        model: Optional[Lazy[Model]] = None,
        tokenizer: Optional[Tokenizer] = None,
        padding_side: str = "right",
        per_device_batch_size: int = 128,
        do_cache_model: bool = True,
        do_cache_model_on_cpu: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        assert tokenizer is not None
        assert padding_side in ["left", "right"]
        
        self.model_lazy = model
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.per_device_batch_size = per_device_batch_size
        self.do_cache_model = do_cache_model
        self.do_cache_model_on_cpu = do_cache_model_on_cpu

        self.model_cache_dir = self.root_dir / "temp_model_cache_dir"
        logger.info(f"Using {self.model_cache_dir} as the temporary model cache directory.")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_reward_model(self) -> PreTrainedModel:
        this_process_device = self.distributed_state.device

        if hasattr(self, "_model"):
            self._model.to(this_process_device)
            return self._model

        t0 = time.time()

        # Load the reward model into GPU
        cache_path = self.model_cache_dir / ".reward_model"
        if not cache_path.exists():
            cache_path = None

        # noinspection PyTypeChecker
        model: PreTrainedModel = self.model_lazy.construct(
            device=this_process_device,
            disable_dropout=True,
            runtime_hf_model_name=cache_path,
        )
        model.to(this_process_device)
        self._cloud_log(
            {"timing/episode_generation/reward_model_construct": time.time() - t0}
        )

        if self.do_cache_model and cache_path is None and self.is_main_process():
            # Since the reward model is used in every iteration, it makes
            # sense to cache it on the fast disk to avoid loading it from network storage
            cache_path = self.model_cache_dir / ".reward_model"
            model.save_pretrained(cache_path, safe_serialization=False)

        if self.do_cache_model_on_cpu:
            self._reward_model = model

        return model

    def __call__(self, input_dataset: Dataset) -> Dataset:
        model = self._init_reward_model()
        model.eval()

        # noinspection PyTypeChecker
        tokenizer: PreTrainedTokenizer = self.tokenizer
        tokenizer.padding_side = self.padding_side

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
            Dataset.from_dict({"text": input_dataset}),
            batch_size=self.per_device_batch_size,
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

                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # Extract the rewards from the last token
                if self.padding_side == "right":
                    assert torch.all(
                        inputs["attention_mask"][:, 0] == 1
                    ), "Reward model expect the padding to be done on the right side."
                    # Compute the index of last token in the sequence lengths
                    seq_lengths = inputs["attention_mask"].sum(dim=1)
                    last_token_indices = seq_lengths - 1
                    rewards = outputs[range(outputs.shape[0]), last_token_indices]
                elif self.padding_side == "left":
                    assert torch.all(
                        inputs["attention_mask"][:, -1] == 1
                    ), "Reward model expect the padding to be done on the left side."
                    rewards = outputs[:, -1]
                else:
                    raise ValueError(
                        f"Invalid padding side: {self.padding_side}"
                    )

                all_rewards.extend(rewards.float().cpu().numpy().tolist())

        assert len(all_rewards) == len(input_dataset)

        if self.do_cache_model_on_cpu:
            model.to("cpu")
        
        release_memory()
        return all_rewards

    def _tokenize_query_and_response(
        self, query: str, response: str, allow_append_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        # This a legacy method that is not used anymore. It is kept here for reference.
        return self._tokenize_trajectory(
            {"query_text": query, "response_text": response},
            is_unfinished_response=not allow_append_eos,
            return_offsets=False,
        )

    def _tokenize_trajectory(
        self,
        trajectory: Dict[str, Any],
        is_unfinished_response: bool = False,
        return_offsets: bool = False,
        safety_check_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[List[int], List[int]],
        Tuple[List[int], List[int], List[Tuple[int, int]]],
    ]:
        safety_check_kwargs = safety_check_kwargs or {}
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]

        episode_text = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = episode_encoding["input_ids"]
        offsets = episode_encoding["offset_mapping"]

        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query_text)
        )
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        self._safety_check_tokenization(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query=query_text,
            response=response_text,
            episode_text=episode_text,
            **safety_check_kwargs,
        )

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if not is_unfinished_response and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        if return_offsets:
            return query_token_ids, response_token_ids, offsets
        else:
            return query_token_ids, response_token_ids

    def _safety_check_tokenization(
        self,
        query_token_ids: List[str],
        response_token_ids: List[str],
        query: str,
        response: str,
        episode_text: str,
        check_query_reconstruction: bool = True,
        check_response_reconstruction: bool = True,
    ):
        decoding_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        decoded_instance = self.tokenizer.decode(
            query_token_ids + response_token_ids, **decoding_kwargs
        )
        assert decoded_instance == episode_text, (
            f"Decoded instance does not match original instance.\n"
            f"Original instance: {episode_text}\n"
            f"Decoded instance: {decoded_instance}"
        )
        
        if check_query_reconstruction:
            decoded_query = self.tokenizer.decode(query_token_ids, **decoding_kwargs)
            assert decoded_query == query, (
                f"Decoded query does not match original query.\n"
                f"Original query: {query}\n"
                f"Decoded query: {decoded_query}"
            )
        
        if check_response_reconstruction:
            decoded_response = self.tokenizer.decode(
                response_token_ids, **decoding_kwargs
            )
            assert decoded_response == response, (
                f"Decoded response does not match original response.\n"
                f"Original response: {response}\n"
                f"Decoded response: {decoded_response}"
            )
