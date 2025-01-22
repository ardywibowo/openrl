import copy
import json
import logging
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.cuda
from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets
from deepspeed.runtime.utils import (see_memory_usage,
                                     torch_max_memory_reserved,
                                     torch_memory_reserved)

from treetune.common import Lazy, Component, VLLMServerHandler
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import find_n_free_ports
from treetune.common.vllm_server import VLLMServer, compute_vllm_stats
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.episodes import Episode
from treetune.pipelines import GenerationPipeline, DatasetPipeline
from treetune.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.tasks.base_task import Task

logger = get_logger(__name__)


class OnPolicyEpisodeGenerator(EpisodeGenerator):
    can_precompute_episodes: bool = False
    support_distributed: bool = True

    def __init__(
        self,
        initial_model_name_or_path: str,
        task: Task,
        generation_pipeline: Lazy[GenerationPipeline],
        dataset_pipeline: Lazy[DatasetPipeline],
        vllm_server_handler: Lazy[VLLMServerHandler],
        dataset_split: str = "train",
        save_generations_every_n_iteration: Optional[int] = None,
        fill_missing_episodes: bool = False,
        max_question_length: Optional[int] = None,
        question_template: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        The base class for episode generators that generate episodes by sampling from the model.
        It supports distributed environments.
        """
        super().__init__(**kwargs)
        
        self.initial_model_name_or_path = initial_model_name_or_path
        self.debug = debug
        
        self.vllm_server_handler = vllm_server_handler.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir=self.root_dir / "vllm_server_handler"
        )
        
        self.generation_pipeline = generation_pipeline.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir=self.root_dir / "generation_pipeline",
            vllm_server_handler=self.vllm_server_handler
        )
        self.dataset_pipeline = dataset_pipeline.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir=self.root_dir / "dataset_pipeline",
            vllm_server_handler=self.vllm_server_handler
        )
        
        self.save_generations_every_n_iteration = save_generations_every_n_iteration
        self.fill_missing_episodes = fill_missing_episodes
        
        self.task = task
        self.dataset_split = dataset_split
        self.max_question_length = max_question_length
        self.question_template = question_template
        
        self._orig_dataset = None
        self._latest_policy_path = None

    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        """
        release_memory()
        if iteration is None:
            self._log_on_main(
                logger,
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0
        
        self.init(iteration=iteration)
        
        # Prepare the dataset on all processes
        if self._orig_dataset is None:
            with self.distributed_state.main_process_first():
                self._orig_dataset = self._get_orig_dataset()
        
        dataset = self.dataset_pipeline(self._orig_dataset, cache=True)
        
        if self._latest_policy_path is None:
            self._latest_policy_path = self.initial_model_name_or_path
        else:
            self._latest_policy_path = str(latest_policy_path)
        
        self.generation_pipeline.set_model(self._latest_policy_path)
        
        infer_results = self._run_inference(dataset)
        
        ################
        metrics = {}
        process_index = self.distributed_state.process_index
        # Generate episodes from inference results. Each process generates its own episodes.
        t0 = time.time()
        episodes_lst = [
            self._convert_to_dict(e)
            for e in self._generate_episodes(infer_results, iteration)
        ]
        episodes_ds_shard = Dataset.from_list(episodes_lst)
        
        temp_dir = self.root_dir / f"iteration__{iteration:04d}"
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
        )
        del episodes_ds_shard
        release_memory()
        metrics["timing/episode_generation/inferResult_to_episodes"] = time.time() - t0
        
        self._cloud_log(metrics)

        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=self.seed + iteration)
                merged = merged.select(range(self.num_episodes_per_iteration))
            elif len(merged) < self.num_episodes_per_iteration:
                if self.fill_missing_episodes:
                    # Fill the missing episodes by repeating the existing ones
                    logger.warning(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration}). "
                        f"Repeating the existing episodes."
                    )
                    num_repeats = self.num_episodes_per_iteration // len(merged) + 1
                    merged = concatenate_datasets([merged] * num_repeats)
                    merged = merged.shuffle(seed=self.seed + iteration)
                    merged = merged.select(range(self.num_episodes_per_iteration))
                    logs = {f"episodes_metric/fill_missing_episodes": num_repeats}
                    self._cloud_log({**logs, "train/global_iteration": iteration})
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))

        see_memory_usage("After generating episodes", force=True)

        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()

        return episodes
        
        ##############
        
        # metrics = {}
        # t0 = time.time()
        # self._convert_and_save_episodes(infer_results)
        # metrics["timing/episode_generation/inferResult_to_episodes"] = time.time() - t0

        # self._cloud_log(metrics)
        # episodes = self._merge_episode_shards(temp_dir)

        # see_memory_usage("After generating episodes", force=True)

        # self._save_generations_to_cloud(temp_dir)
        # self._clean_up_temp_dir(temp_dir)

        # self.vllm_server_handler.kill_server()
        # self.distributed_state.wait_for_everyone()
        
        # return episodes

    def _run_inference(self, dataset: Dataset):
        # Generate episodes from inference results. Each process generates its own episodes.
        infer_results = self.generation_pipeline(dataset, cache=False)
        return infer_results
        
    def _get_orig_dataset(self):
        orig_dataset = self.task.get_datasets(self.dataset_split)
        self._log_on_main(logger, f"Initial Dataset Size: {len(orig_dataset)}")
        orig_dataset = self._filter_init_dataset(orig_dataset)
        
        orig_dataset = self.dataset_pipeline.process_init_dataset(orig_dataset)
        return orig_dataset

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        raise NotImplementedError

    def _merge_episode_shards(self, temp_dir: Path):
        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=self.seed + self._iteration)
                merged = merged.select(range(self.num_episodes_per_iteration))
            elif len(merged) < self.num_episodes_per_iteration:
                if self.fill_missing_episodes:
                    # Fill the missing episodes by repeating the existing ones
                    logger.warning(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration}). "
                        f"Repeating the existing episodes."
                    )
                    num_repeats = self.num_episodes_per_iteration // len(merged) + 1
                    merged = concatenate_datasets([merged] * num_repeats)
                    merged = merged.shuffle(seed=self.seed + self._iteration)
                    merged = merged.select(range(self.num_episodes_per_iteration))
                    logs = {f"episodes_metric/fill_missing_episodes": num_repeats}
                    self._cloud_log({**logs, "train/global_iteration": self._iteration})
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))
        return episodes

    def _convert_and_save_episodes(self, infer_results: Dataset) -> None:
        """
        Converts inference results into episodes and saves them to disk.
        """
        process_index = self.distributed_state.process_index
        shard_dir = temp_dir / "episodes" / f"shard_{process_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        infer_results.save_to_disk(shard_dir)
        release_memory()

        # Convert results to episodes
        episodes_list = [
            self._convert_to_dict(e)
            for e in self._generate_episodes(infer_results)
        ]
        episodes_ds_shard = Dataset.from_list(episodes_list)
        episodes_ds_shard.save_to_disk(shard_dir)

        del episodes_ds_shard

    def _save_generations_to_cloud(self, generations_dir: Path, iteration: int):
        if self.cloud_logger is None or not self.is_main_process():
            return

        if self.save_generations_every_n_iteration is None:
            # Saving generations is disabled
            return

        if iteration != 0 and iteration % self.save_generations_every_n_iteration != 0:
            # We only save generations every n iterations and the first iteration
            return

        temp_dir = Path(tempfile.mkdtemp())

        generations = temp_dir / f"iteration__{iteration:04d}.zip"
        shutil.make_archive(
            str(generations.with_suffix("")),
            format="zip",
            root_dir=generations_dir,
        )
        self.cloud_logger.save(str(generations.absolute()), policy="now")

    def _clean_up_temp_dir(self, temp_dir: Path) -> None:
        if not self.is_main_process():
            return

        try:
            # Remove all input_dataset__* directories
            for p in temp_dir.glob("input_dataset__*"):
                shutil.rmtree(p, ignore_errors=True)

            # Remove all episodes shards
            for p in (temp_dir / "episodes").glob("shard_*"):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error while cleaning up temp dir: {e}")

    def _filter_init_dataset(self, dataset: Dataset) -> Dataset:
        if self.max_question_length is None:
            return dataset
        
        tokenizer = self.tokenizer
        question_template = self.question_template
        max_question_length = self.max_question_length
        question_format_keys = []
        for column in dataset.column_names:
            if f"{{{column}}}" in self.question_template:
                question_format_keys.append(column)
        
        if len(question_format_keys) == 0:
            raise ValueError(
                "No columns found in the question template. "
                "Please add the column names in the question template."
            )
        
        def filter_out_long_questions(example):
            format_kwargs = {key: example[key] for key in question_format_keys}
            prompt = question_template.format(**format_kwargs)
            tokens = tokenizer(prompt).input_ids
            return len(tokens) <= max_question_length
        
        length_before = len(dataset)
        dataset = dataset.filter(
            filter_out_long_questions, num_proc=4, desc="Filtering long questions"
        )
        self._log_on_main(
            logger,
            f"Filtered out {length_before - len(dataset)} long questions from {length_before} questions."
        )
        return dataset
