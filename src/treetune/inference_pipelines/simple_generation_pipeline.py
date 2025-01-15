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
        process_index = self.distributed_state.process_index
        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        seed = self.seed + process_index * 100 + iteration
        
        results_dir = temp_dir / "results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)
        hf_ckpt_path_or_model = self.model_name_or_path
        
        # Prepare the dataset on all processes
        with self.distributed_state.main_process_first():
            dataset = Dataset.from_dict({"query": sequences})

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
        
        vllm_init_fn = self._get_vllm_init_fn(
            results_dir=results_dir,
            hf_ckpt_path_or_model=hf_ckpt_path_or_model,
            process_index=process_index,
            seed=seed,
        )

        metrics = {}
        t0 = time.time()

        this_process_device = self.distributed_state.device
        release_memory()
        gpu_memory_usage_before_mb = get_gpu_memory()[this_process_device.index]
        def vllm_cleanup_fn():
            if self.wait_until_memory_release:
                threshold_mb = (
                    gpu_memory_usage_before_mb * 1.1
                )  # Allow for 10% tolerance
                wait_for_memory_release(
                    this_process_device.index,
                    threshold_mb=threshold_mb,
                )

        infer_results = self._run_inference(
            dataset_shard=dataset,
            vllm_init_fn=vllm_init_fn,
            vllm_cleanup_fn=vllm_cleanup_fn,
            results_root_dir=results_dir,
            seed=seed,
            iteration=iteration,
        )

        metrics["timing/episode_generation/inference"] = time.time() - t0

        logger.info(f"Process {process_index} finished inference.")

        # Generate episodes from inference results. Each process generates its own episodes.
        episodes_lst = [
            self._convert_to_dict(e)
            for e in self._convert_results_to_inference_output(infer_results)
        ]
        episodes_ds_shard = Dataset.from_list(episodes_lst)
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
        )
        del episodes_ds_shard
        release_memory()

        # Log the vLLM stats
        if self.distributed_state.is_main_process:
            try:
                vllm_stats = compute_vllm_stats(results_dir / "vllm_server.log")
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

        self._cloud_log(metrics)

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

        return episodes

    def _convert_results_to_inference_output(
        self, inference_results: Dataset
    ) -> List[InferenceOutput]:
        return [output 
            for instance in inference_results 
            for output in self._convert_to_inference_output(instance)
        ]

    def _convert_to_inference_output(self, instance: Dict[str, Any]) -> List[InferenceOutput]:
        tree = json.loads(instance["_treetune__reasoning_tree"])
        paths = self.extract_paths_from_tree(tree)
        
        return [self._convert_path_to_inference_output(path) for path in paths]

    def _convert_path_to_inference_output(
        self, 
        path: Dict[str, Any]
    ) -> List[InferenceOutput]:
        assert len(path["node_chain"]) == 2
        
        query_text = path["node_chain"][0]["text"]
        full_text = path["node_chain"][-1]["full_text"]
        response_text = full_text[len(query_text) :]
        
        return InferenceOutput(
            query=query_text,
            response=response_text
        )

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
        Potentially start a vLLM server and run inference to generate results needed for episode generation.

        Args:
            dataset_shard (Dataset):
                The shard of the prompt dataset to run inference on.
            vllm_init_fn (Callable[[], Tuple[VLLMServer, Dict[str, Any]]]):
                A function that initializes the vLLM server and returns the server object and the server URL.
            results_root_dir (Path):
                The directory to save the results to (this is unique for each process).
            seed (int):
                The seed for this process to use for inference.
        """
        infer_result_path = results_root_dir / "results_ds"
        vllm_server, guidance_llm_kwargs = vllm_init_fn()

        # Initialize the inference strategy with the vLLM server URL
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

        results = inference_strategy.generate(dataset_shard)
        results.save_to_disk(str(infer_result_path))
        
        logger.info(f"Rank {self.distributed_state.process_index} finished inference.")
        vllm_server.stop_server()
        del results
        del vllm_server
        release_memory()
        logger.info(f"Rank {self.distributed_state.process_index} stopped vLLM server.")

        vllm_cleanup_fn()
        release_memory()

        results = Dataset.load_from_disk(str(results_root_dir / "results_ds"))
        return results

    def _set_vllm_ports(self, seed: Optional[int] = None):
        """
        The main process searches for self.distributed_state.num_processes's free ports.
        and then broadcasts the ports to all processes.
        """
        if self.distributed_state.process_index == 0:
            ports = find_n_free_ports(
                self.distributed_state.num_processes, generator=self._port_generator_rng
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
        vllm_server_lazy = self.vllm_server_lazy
        vllm_gpu_memory_utilization = self.vllm_gpu_memory_utilization
        self._set_vllm_ports(seed=seed)
        vllm_port = self._vllm_port
        if vllm_gpu_memory_utilization == "auto":
            # Compute the GPU utilization based on amount of remaining memory
            allocated_mem_mb = get_gpu_memory()[process_index]
            total_mem_mb = (
                torch.cuda.get_device_properties(process_index).total_memory / 1024**2
            )

            remaining_mem_mb = (
                total_mem_mb - allocated_mem_mb
            ) * 0.9  # Allow for 10% tolerance
            vllm_gpu_memory_utilization = round(remaining_mem_mb / total_mem_mb, 2)

            logger.info(
                f"GPU #{process_index} Auto-computed vLLM GPU memory utilization: {vllm_gpu_memory_utilization}. "
                f"Currently Allocated: {allocated_mem_mb} MB, "
                f"Total: {total_mem_mb} MB, "
                f"Remaining: {remaining_mem_mb} MB."
            )

        def _init() -> Tuple[VLLMServer, Dict[str, Any]]:
            vllm_log_path = results_dir / "vllm_server.log"

            logger.info(
                f"Rank #{process_index} starting vLLM: "
                f"model={hf_ckpt_path_or_model}   port={vllm_port}   seed={seed}"
            )
            t0 = time.time()
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
            self._cloud_log(
                {
                    "timing/episode_generation/vllm_start": time.time() - t0,
                }
            )

            return vllm_server, {
                "api_base": server_url,
                "model": hf_ckpt_path_or_model,
            }

        return _init
    
    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        if isinstance(episode_obj, dict):
            return episode_obj

        return asdict(episode_obj)
