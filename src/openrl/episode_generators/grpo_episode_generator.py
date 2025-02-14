import copy
import json
import logging
import math
import random
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets

from openrl.common import Lazy
from openrl.common.logging_utils import get_logger
from openrl.episode_generators.base_episode_generator import EpisodeGenerator
from openrl.episodes import Episode
from openrl.inference_servers import InferenceServer
from openrl.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from openrl.reward_functions import RewardFunction
from openrl.tasks.base_task import Task

logger = get_logger(__name__)

@EpisodeGenerator.register("grpo_episode_generator")
class GRPOEpisodeGenerator(EpisodeGenerator):
    can_precompute_episodes: bool = False
    support_distributed: bool = True

    def __init__(
        self,
        task: Task,
        initial_model_name_or_path: str,
        inference_server: Lazy[InferenceServer],
        inference_strategy: Lazy[InferenceStrategy],
        reward_function: Lazy[RewardFunction],
        dataset_split: str = "train",
        dataset_num_samples_per_iteration: Optional[int] = None,
        temp_dir_root: Optional[str] = None,
        save_generations_every_n_iteration: Optional[int] = None,
        **kwargs,
    ):
        """
        The base class for episode generators that generate episodes by sampling from the model.
        It supports distributed environments.
        """
        super().__init__(**kwargs)
        self._logger = logger

        self.inference_strategy_lazy = inference_strategy
        self.inference_server = inference_server.construct(**kwargs)
        self.task = task
        self.dataset_split = dataset_split
        self.initial_model_name_or_path = initial_model_name_or_path
        self.dataset_num_samples_per_iteration = dataset_num_samples_per_iteration
        self.save_generations_every_n_iteration = save_generations_every_n_iteration
        
        self.reward_function = reward_function.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir= self.root_dir,
            tokenizer=self.tokenizer
        )
        
        if temp_dir_root is None:
            self.temp_dir_root = self.root_dir / "temp_episodes"
            self._log_on_main(logger, f"Using default temp_dir_root: {self.temp_dir_root}")
        else:
            self.temp_dir_root = Path(temp_dir_root)
        self.temp_dir_root.mkdir(parents=True, exist_ok=True)

        self._orig_ds = None

    def _init_orig_ds(self):
        ds = self.task.get_datasets(self.dataset_split)
        self._log_on_main(logger, f"Initial Dataset Size: {len(ds)}")
        
        self._orig_ds = ds

    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        """
        release_memory()

        from deepspeed.runtime.utils import see_memory_usage

        see_memory_usage("Before generating episodes", force=True)

        if iteration is None:
            self._log_on_main(
                logger,
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0
        
        self.init(iteration)
        process_index = self.distributed_state.process_index
        
        # Prepare the dataset on all processes
        if self._orig_ds is None:
            with self.distributed_state.main_process_first():
                self._init_orig_ds()
        
        dataset = self._orig_ds
        num_samples = self.dataset_num_samples_per_iteration
        assert num_samples <= len(dataset)
        
        dataset = dataset.shuffle(seed=self.seed + iteration)
        dataset = dataset.select(range(num_samples))
        
        self._log_on_main(
            logger,
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(2, len(dataset)))], indent=2, sort_keys=True)}"
        )

        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save to disk so that it's memory efficient. Note that this is done on all processes.
        # to avoid any issues with distributed environment and funkiness of HF Datasets.
        inp_ds_path = temp_dir / f"input_dataset__{process_index}"
        dataset.save_to_disk(inp_ds_path)
        del dataset

        # The same dataset is loaded on all processes
        dataset = Dataset.load_from_disk(str(inp_ds_path))

        # Shard the dataset based on the number of processes
        dataset = dataset.shard(
            num_shards=self.distributed_state.num_processes,
            index=process_index,
            contiguous=True,
        )

        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)

        if latest_policy_path is None:
            hf_ckpt_path_or_model = self.initial_model_name_or_path
        else:
            hf_ckpt_path_or_model = str(latest_policy_path)
        
        t0 = time.time()
        episodes_ds_shard = self._run_inference(
            dataset_shard=dataset,
            model_name_or_path=hf_ckpt_path_or_model,
            results_root_dir=results_dir
        )
        self.distributed_state.wait_for_everyone()
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
        )
        del episodes_ds_shard
        self._metrics["timing/episode_generation/inference"] = time.time() - t0
        logger.info(f"Process {process_index} finished inference.")
        release_memory()
        
        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))
            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
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
        
        metrics = self.gather_metrics()
        if len(metrics) > 0:
            metrics["train/global_iteration"] = iteration
            self._cloud_log(metrics)
        
        self.distributed_state.wait_for_everyone()
        return episodes

    def _run_inference(
        self,
        dataset_shard: Dataset,
        model_name_or_path: str,
        results_root_dir: Path,
    ):
        """
        Potentially start an inference server and run inference to generate results needed for episode generation.

        Args:
            dataset_shard (Dataset):
                The shard of the prompt dataset to run inference on.
            model_name_or_path (str):
                The model to use for inference.
            results_root_dir (Path):
                The directory to save the results to (this is unique for each process).
        """
        infer_result_path = results_root_dir / "results_ds"
        self.inference_server.start_server(model_name_or_path, results_root_dir)

        # Initialize the inference strategy with the inference server URL
        inference_strategy = self.inference_strategy_lazy.construct(
            root_dir=results_root_dir,
            seed=self.get_process_seed(),
            cloud_logger=None,
            log_level=(
                logging.WARNING
                if not self.distributed_state.is_local_main_process
                else None
            ),
        )

        results = inference_strategy.generate(dataset_shard)
        results = self._generate_episodes(results)
        results.save_to_disk(str(infer_result_path))
        
        logger.info(f"Rank {self.distributed_state.process_index} finished inference.")
        del results
        
        self.inference_server.stop_server()
        # self.inference_server_handler.compute_server_stats(results_root_dir)
        
        results = Dataset.load_from_disk(str(results_root_dir / "results_ds"))
        return results
    
    def _generate_episodes(self, results: Dataset) -> Dataset:
        """
        Compute rewards and advantages for each response group in the given HuggingFace Dataset.
        For each row in `results` (which should include "response_group" and "full_text_group" columns),
        this function:
        1. Extracts the query text by removing the candidate response from the full text.
        2. Computes a reward for each candidate by calling:
                reward = self.reward_function(response, data_instance)
        3. Computes the group-average reward.
        4. Computes each candidate's advantage as (reward - group_avg).
        Finally, it flattens the data so that each row in the returned Dataset corresponds to a 
        single candidate response with its computed reward and advantage, and further converts 
        it into an Episode.
        """
        
        def compute_rewards(row: dict) -> dict:
            full_texts = row["full_text_group"]
            responses = row["response_group"]
            
            # Extract queries by removing the trailing response from each full text.
            queries = [
                full_text[:-len(response)] if response else full_text
                for full_text, response in zip(full_texts, responses)
            ]
            rewards = [self.reward_function(response, row) for response in responses]
            group_avg = sum(rewards) / len(rewards) if rewards else 0.0
            advantages = [reward - group_avg for reward in rewards]
            
            # Compute standard deviation.
            std = math.sqrt(sum((r - group_avg) ** 2 for r in rewards) / len(rewards)) if rewards else 0.0
            if std == 0.0:
                std = 1.0
            
            advantages = [(reward - group_avg) / std for reward in rewards]
            
            # Save computed lists in the row for later flattening.
            row["flat_queries"] = queries
            row["flat_response"] = responses
            row["flat_reward"] = rewards
            row["flat_advantage"] = advantages
            return row
        
        def flatten_row(row: dict) -> list:
            # For each candidate in the row, produce a separate dictionary.
            return [
                {
                    "query": query,
                    "response": response,
                    "reward": reward,
                    "advantage": advantage,
                }
                for query, response, reward, advantage in zip(
                    row["flat_queries"],
                    row["flat_response"],
                    row["flat_reward"],
                    row["flat_advantage"],
                )
            ]
        
        def process_episode(row: dict) -> dict:
            query_token_ids, response_token_ids = self._tokenize(row["query"], row["response"])
            episode = Episode(
                query_token_ids=query_token_ids,
                response_token_ids=response_token_ids,
                query_text=row["query"],
                response_text=row["response"],
                scores=row["reward"],
                advantages=[row["advantage"] for _ in range(len(response_token_ids))],
            )
            return self._convert_to_dict(episode)
        
        # Step 1: Compute rewards and advantages per row (non-batched).
        results = results.map(compute_rewards, num_proc=8, batched=False)
        
        # Step 2: Flatten the dataset by iterating over each row.
        flat_list = []
        for row in results:
            flat_list.extend(flatten_row(row))
        flattened_dataset = Dataset.from_list(flat_list)
        
        # Step 3: Process each flattened example into an Episode.
        episodes = flattened_dataset.map(process_episode, num_proc=8)
        
        return episodes

    def _tokenize(self, query_text: str, response_text: str) -> Tuple[List[int], List[int]]:
        """
        Tokenize the concatenated query and response texts.
        Uses the tokenizer's offset mapping to determine the split between query and response.
        Optionally appends BOS to the query and EOS to the response if configured.
        """
        episode_text: str = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens later.
            return_offsets_mapping=True,
        )
        
        token_ids: List[int] = episode_encoding["input_ids"]
        offsets: List[Tuple[int, int]] = episode_encoding["offset_mapping"]
        
        # Determine the index where the response begins.
        response_start_index = next(
            (i for i, (start, _) in enumerate(offsets) if start >= len(query_text)),
            None
        )
        if response_start_index is None:
            raise ValueError("Could not determine the start of the response based on query length.")
        
        query_token_ids: List[int] = token_ids[:response_start_index]
        response_token_ids: List[int] = token_ids[response_start_index:]
        
        if self.tokenizer.bos_token_id is not None:
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids
        
        if self.tokenizer.eos_token_id is not None:
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]
        
        return query_token_ids, response_token_ids

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

    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        if isinstance(episode_obj, dict):
            return episode_obj

        return asdict(episode_obj)
