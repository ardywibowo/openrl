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
from treetune.inference_pipelines.base_inference_pipeline import InferenceOutput
from treetune.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.models import Model
from treetune.reward_functions import RewardFunction
from treetune.tasks import GSM8K, Task
from treetune.tasks.math import MATH
from treetune.tokenization_utils import Tokenizer
from treetune.trainers.policy_trainer import Checkpoint
from treetune.inference_pipelines import InferencePipeline

logger = get_logger(__name__)

@InferencePipeline.register("simple_generation_pipeline")
class SimpleGenerationPipeline(InferencePipeline, TreeEpisodeUtils):
    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        vllm_server: Lazy[VLLMServer],
        vllm_gpu_memory_utilization: Union[float, str] = 0.9,
        vllm_min_available_gpu_memory_mb: Optional[int] = None,
        model_name_or_path: str = "",
        tokenizer: Tokenizer = None,
        save_generations_every_n_iteration: Optional[int] = None,
        wait_until_memory_release: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.inference_strategy_lazy = inference_strategy
        self.vllm_server_lazy = vllm_server
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_min_available_gpu_memory_mb = vllm_min_available_gpu_memory_mb
        
        self.save_generations_every_n_iteration = save_generations_every_n_iteration
        self.model_name_or_path = model_name_or_path
        self.wait_until_memory_release = wait_until_memory_release
        self._port_generator_rng = random.Random(self.seed)
        
        self.tokenizer = tokenizer

    def generate(self, sequences: List[str], iteration: int) -> Dataset:
        # 1. Prepare the dataset shard
        dataset_shard, temp_dir, results_dir, seed = self._prepare_sharded_dataset(
            sequences, iteration
        )

        # 2. Run inference
        inference_start_time = time.time()
        infer_results, metrics = self._run_inference_with_vllm(
            dataset_shard=dataset_shard,
            results_dir=results_dir,
            seed=seed,
            iteration=iteration
        )
        metrics["timing/episode_generation/inference"] = time.time() - inference_start_time

        logger.info(f"Process {self.distributed_state.process_index} finished inference.")

        # 3. Convert inference results to episodes and save them.
        self._convert_and_save_episodes(
            infer_results=infer_results,
            temp_dir=temp_dir
        )

        # Log intermediate metrics (including vLLM stats already inserted)
        self._cloud_log(metrics)

        # 4. Merge shards if main process, etc.
        episodes = self._merge_episode_shards(temp_dir)
        see_memory_usage("After generating episodes", force=True)

        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)

        self.distributed_state.wait_for_everyone()
        return episodes

    def _run_inference_with_vllm(
        self,
        dataset_shard: Dataset,
        results_dir: Path,
        seed: int,
        iteration: int
    ) -> Tuple[Dataset, Dict[str, float]]:
        """
        Initializes the vLLM server for this process, runs inference, handles cleanup,
        then computes and logs vLLM stats.
        """
        metrics: Dict[str, float] = {}

        # 1) Check GPU memory usage and define cleanup
        device = self.distributed_state.device
        release_memory()
        gpu_memory_before_mb = get_gpu_memory()[device.index]

        def vllm_cleanup_fn():
            if self.wait_until_memory_release:
                threshold_mb = gpu_memory_before_mb * 1.1  # 10% tolerance
                wait_for_memory_release(device.index, threshold_mb=threshold_mb)

        # 2) Initialize and run inference
        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=results_dir,
            hf_ckpt_path_or_model=self.model_name_or_path,
            process_index=self.distributed_state.process_index,
            seed=seed
        )
        infer_results = self._run_inference(
            dataset_shard=dataset_shard,
            vllm_init_fn=vllm_init_fn,
            vllm_cleanup_fn=vllm_cleanup_fn,
            results_root_dir=results_dir,
            seed=seed,
            iteration=iteration,
        )

        # 3) Once inference is done and server is stopped, compute vLLM stats (main process only)
        self._compute_and_log_vllm_stats(results_dir, metrics)

        return infer_results, metrics

    def _convert_and_save_episodes(
        self,
        infer_results: Dataset,
        temp_dir: Path
    ) -> None:
        """
        Converts inference results into InferenceOutput objects, then saves them to disk.
        """
        process_index = self.distributed_state.process_index
        shard_dir = temp_dir / "episodes" / f"shard_{process_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Convert results to episodes
        episodes_list = [
            self._convert_to_dict(e)
            for e in self._convert_results_to_inference_output(infer_results)
        ]
        episodes_ds_shard = Dataset.from_list(episodes_list)
        episodes_ds_shard.save_to_disk(shard_dir)

        del episodes_ds_shard
        release_memory()

    def _compute_and_log_vllm_stats(
        self,
        results_dir: Path,
        metrics: Dict[str, float]
    ) -> None:
        """
        If main process, parse vLLM logs from `results_dir / vllm_server.log`
        and update the metrics dict.
        """
        if not self.distributed_state.is_main_process:
            return
        try:
            log_path = results_dir / "vllm_server.log"
            vllm_stats = compute_vllm_stats(log_path)
        except Exception as e:
            logger.error(f"Error while computing vLLM stats: {e}")
            vllm_stats = {}

        if "avg_generation_throughput" in vllm_stats:
            vllm_stats["total_approx_generation_throughput"] = (
                vllm_stats["avg_generation_throughput"] 
                * self.distributed_state.num_processes
            )

        vllm_stats = {f"vllm_stats/{k}": round(v, 2) for k, v in vllm_stats.items()}
        logger.info(f"vLLM Stats: {vllm_stats}")
        metrics.update(vllm_stats)

    def _prepare_sharded_dataset(
        self, sequences: List[str], iteration: int
    ) -> Tuple[Dataset, Path, Path, int]:
        """
        Creates an iteration-specific directory, builds a dataset from sequences
        on all processes, saves it to disk, reloads, and returns the per-process shard.
        
        Returns:
            dataset_shard (Dataset): The portion of the dataset for this process.
            temp_dir (Path): Directory for storing iteration artifacts.
            results_dir (Path): Directory where this process will store results.
            seed (int): Process- and iteration-specific random seed.
        """
        process_index = self.distributed_state.process_index

        # Create the iteration directory
        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Derive a seed for reproducibility
        seed = self.seed + process_index * 100 + iteration

        # Directory for results from this rank
        results_dir = temp_dir / "results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset from sequences (main process first to avoid collisions)
        with self.distributed_state.main_process_first():
            dataset = Dataset.from_dict({"query": sequences})

        # Save to disk to avoid memory overhead in distributed settings
        inp_ds_path = temp_dir / f"input_dataset__{process_index}"
        dataset.save_to_disk(inp_ds_path)
        del dataset

        # Reload and shard
        dataset = Dataset.load_from_disk(str(inp_ds_path))
        dataset_shard = dataset.shard(
            num_shards=self.distributed_state.num_processes,
            index=process_index,
            contiguous=True,
        )

        return dataset_shard, temp_dir, results_dir, seed

    def _merge_episode_shards(self, temp_dir: Path) -> Dataset:
        """
        If main process, merges all episodes shards into one dataset; 
        otherwise waits until merging is complete.
        """
        self.distributed_state.wait_for_everyone()

        if self.is_main_process():
            shard_paths = sorted(
                (temp_dir / "episodes").glob("shard_*"),
                key=lambda x: int(x.name.split("shard_")[-1])
            )
            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        return Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))

    def _save_generations_to_cloud(self, generations_dir: Path, iteration: int):
        """
        If enabled, saves local generation files as a zip archive to the cloud.
        """
        if self.cloud_logger is None or not self.is_main_process():
            return

        if self.save_generations_every_n_iteration is None:
            return  # saving generations is disabled

        # Only save on multiples of n (including 0 if iteration == 0)
        if iteration != 0 and iteration % self.save_generations_every_n_iteration != 0:
            return

        temp_dir = Path(tempfile.mkdtemp())
        zip_path = temp_dir / f"iteration__{iteration:04d}.zip"
        shutil.make_archive(
            str(zip_path.with_suffix("")),
            format="zip",
            root_dir=generations_dir,
        )
        self.cloud_logger.save(str(zip_path.absolute()), policy="now")

    def _clean_up_temp_dir(self, temp_dir: Path) -> None:
        """
        Cleanup leftover dataset and shard directories after merging on the main process.
        """
        if not self.is_main_process():
            return
        try:
            # Remove saved input shards
            for p in temp_dir.glob("input_dataset__*"):
                shutil.rmtree(p, ignore_errors=True)
            # Remove per-process episodes shards
            for p in (temp_dir / "episodes").glob("shard_*"):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error while cleaning up temp dir: {e}")

    def _run_inference(
        self,
        dataset_shard: Dataset,
        vllm_init_fn: Callable[[], Tuple[VLLMServer, Dict[str, Any]]],
        vllm_cleanup_fn: Callable[[], None],
        results_root_dir: Path,
        seed: int,
        iteration: int,
    ) -> Dataset:
        """
        Runs the configured inference strategy against the dataset shard, 
        saves the results, and stops the vLLM server.
        """
        infer_result_path = results_root_dir / "results_ds"
        vllm_server, guidance_llm_kwargs = vllm_init_fn()

        # Inject the vLLM server info into the inference strategy
        inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
        inference_strategy_lazy._params["guidance_llm"].update(guidance_llm_kwargs)
        inference_strategy = inference_strategy_lazy.construct(
            result_dir=results_root_dir,
            seed=seed,
            cloud_logger=None,
            log_level=(
                logging.WARNING
                if not self.distributed_state.is_local_main_process
                else None
            ),
        )

        # Generate and save inference results
        results = inference_strategy.generate(dataset_shard)
        results.save_to_disk(str(infer_result_path))
        
        logger.info(f"Rank {self.distributed_state.process_index} finished inference.")
        vllm_server.stop_server()
        del results
        del vllm_server
        release_memory()
        logger.info(f"Rank {self.distributed_state.process_index} stopped vLLM server.")

        # Cleanup
        vllm_cleanup_fn()
        release_memory()

        # Reload final inference results from disk
        return Dataset.load_from_disk(str(infer_result_path))

    def _set_vllm_ports(self, seed: Optional[int] = None):
        """
        On the main process, find free ports for all ranks. 
        Broadcast them so each rank knows which port to use.
        """
        if self.distributed_state.process_index == 0:
            ports = find_n_free_ports(
                self.distributed_state.num_processes,
                generator=self._port_generator_rng
            )
            logger.info(f"Found free ports: {ports}")
        else:
            ports = [0] * self.distributed_state.num_processes

        from accelerate.utils import broadcast_object_list
        ports = broadcast_object_list(ports, from_process=0)
        release_memory()
        self._vllm_port = ports[self.distributed_state.process_index]
        logger.info(
            f"Rank {self.distributed_state.process_index} using vLLM port {self._vllm_port}"
        )

    def _get_vllm_init_fn(
        self,
        results_dir: Path,
        hf_ckpt_path_or_model: str,
        process_index: int,
        seed: int,
    ) -> Callable[[], Tuple[VLLMServer, Dict[str, Any]]]:
        """
        Returns a function that, when called, starts a vLLM server (with appropriate 
        ports & GPU memory) and returns both the server and config details.
        """
        # Possibly find and set vLLM ports
        self._set_vllm_ports(seed=seed)

        vllm_server_lazy = self.vllm_server_lazy
        vllm_gpu_memory_utilization = self.vllm_gpu_memory_utilization
        vllm_port = self._vllm_port

        # If set to "auto", compute usage as a fraction of remaining GPU memory
        if vllm_gpu_memory_utilization == "auto":
            allocated_mem_mb = get_gpu_memory()[process_index]
            total_mem_mb = torch.cuda.get_device_properties(process_index).total_memory / (1024 ** 2)
            remaining_mem_mb = (total_mem_mb - allocated_mem_mb) * 0.9
            vllm_gpu_memory_utilization = round(remaining_mem_mb / total_mem_mb, 2)

            logger.info(
                f"GPU #{process_index} Auto vLLM memory utilization: {vllm_gpu_memory_utilization}. "
                f"Allocated: {allocated_mem_mb} MB, Total: {total_mem_mb} MB, "
                f"Remaining*0.9: {remaining_mem_mb} MB."
            )

        def _init() -> Tuple[VLLMServer, Dict[str, Any]]:
            vllm_log_path = results_dir / "vllm_server.log"
            logger.info(
                f"Rank #{process_index} starting vLLM: "
                f"model={hf_ckpt_path_or_model}, port={vllm_port}, seed={seed}"
            )
            start_t = time.time()

            # Construct the vLLM server
            vllm_server = vllm_server_lazy.construct(
                seed=seed,
                port=vllm_port,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
            server_url = vllm_server.start_server(
                hf_ckpt_path_or_model=hf_ckpt_path_or_model,
                gpu_idx=process_index,
                wait_for_response=True,
                log_path=vllm_log_path,
                timeout=800,
            )

            # Log how long it took to start vLLM
            self._cloud_log({"timing/episode_generation/vllm_start": time.time() - start_t})
            return vllm_server, {
                "api_base": server_url,
                "model": hf_ckpt_path_or_model,
            }

        return _init

    def _convert_results_to_inference_output(
        self, inference_results: Dataset
    ) -> List[InferenceOutput]:
        """
        Flatten the inference results into a list of InferenceOutput objects.
        """
        return [
            output
            for instance in inference_results
            for output in self._convert_to_inference_output(instance)
        ]

    def _convert_to_inference_output(self, instance: Dict[str, Any]) -> List[InferenceOutput]:
        """
        For a single row (instance), parse the JSON tree and extract the paths 
        as InferenceOutput objects.
        """
        tree = json.loads(instance["_treetune__reasoning_tree"])
        paths = self.extract_paths_from_tree(tree)
        return [self._convert_path_to_inference_output(path) for path in paths]

    def _convert_path_to_inference_output(self, path: Dict[str, Any]) -> InferenceOutput:
        """
        Each path is a chain of nodes. We expect exactly 2: [query_node, response_node].
        """
        assert len(path["node_chain"]) == 2
        query_text = path["node_chain"][0]["text"]
        full_text = path["node_chain"][-1]["full_text"]
        response_text = full_text[len(query_text) :]
        return InferenceOutput(query=query_text, response=response_text)

    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        """
        Convert the final data structure to a dict for saving in HF Datasets.
        """
        if isinstance(episode_obj, dict):
            return episode_obj
        return asdict(episode_obj)
