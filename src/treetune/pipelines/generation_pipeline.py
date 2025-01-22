import copy
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate.utils import release_memory
from datasets import Dataset
from deepspeed.runtime.utils import see_memory_usage
from transformers import Pipeline

from treetune.common import Lazy
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import find_n_free_ports
from treetune.common.vllm_server import VLLMServer, compute_vllm_stats
from treetune.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from treetune.logging_utils import get_logger
from treetune.pipelines import Pipeline

logger = get_logger(__name__)

@Pipeline.register("generation_pipeline")
class GenerationPipeline(Pipeline):
    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        model_name_or_path: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.inference_strategy_lazy = inference_strategy
        self.model_name_or_path = model_name_or_path

    def __call__(self, input_dataset: Dataset, cache: bool = False) -> Dataset:
        
        see_memory_usage("Before generating episodes", force=True)
        t0 = time.time()
        infer_results = self._run_inference_with_vllm(input_dataset, self._iteration)
        self._metrics["timing/episode_generation/inference"] = time.time() - t0

        logger.info(f"Process {self.distributed_state.process_index} finished inference.")
        see_memory_usage("After generating episodes", force=True)

        self.distributed_state.wait_for_everyone()
        
        return infer_results
    
    def set_model(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path

    def _run_inference_with_vllm(self, input_dataset: Dataset) -> Tuple[Dataset, Dict[str, float]]:
        """
        Initializes the vLLM server for this process, runs inference, handles cleanup,
        then computes and logs vLLM stats.
        """
        # Create the iteration directory
        temp_dir = self.root_dir / f"iteration__{self._iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Derive a seed for reproducibility
        process_index = self.distributed_state.process_index
        seed = self.seed + self._iteration * self.distributed_state.num_processes + process_index

        # Directory for results from this rank
        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        infer_results = self._run_inference(
            input_dataset=input_dataset,
            results_dir=results_dir,
            seed=seed
        )

        return infer_results

    def _run_inference(
        self,
        input_dataset: Dataset,
        results_dir: Path,
        seed: int
    ) -> Dataset:
        """
        Runs the configured inference strategy against the dataset shard, 
        saves the results, and stops the vLLM server.
        """
        infer_result_path = results_dir / "results_ds"
        vllm_server_config = self.vllm_server_handler.get_or_create_vllm_server_with_model(self.model_name_or_path)
        
        # Inject the vLLM server info into the inference strategy
        inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
        inference_strategy_lazy._params["guidance_llm"].update(vllm_server_config)
        inference_strategy = inference_strategy_lazy.construct(
            result_dir=results_dir,
            seed=seed,
            cloud_logger=None,
            log_level=(
                logging.WARNING
                if not self.distributed_state.is_local_main_process
                else None
            ),
        )
        
        # Generate and save inference results
        results = inference_strategy.generate(input_dataset)
        results.save_to_disk(str(infer_result_path))
        logger.info(f"Rank {self.distributed_state.process_index} finished inference.")

        # Reload final inference results from disk
        return Dataset.load_from_disk(str(infer_result_path))

    def _set_vllm_ports(self, seed: Optional[int] = None) -> None:
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
        model_name_or_path: str,
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
                f"model={model_name_or_path}, port={vllm_port}, seed={seed}"
            )
            start_t = time.time()

            # Construct the vLLM server
            vllm_server = vllm_server_lazy.construct(
                seed=seed,
                port=vllm_port,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
            )
            server_url = vllm_server.start_server(
                hf_ckpt_path_or_model=model_name_or_path,
                gpu_idx=process_index,
                wait_for_response=True,
                log_path=vllm_log_path,
                timeout=800,
            )

            # Log how long it took to start vLLM
            self._cloud_log({"timing/episode_generation/vllm_start": time.time() - start_t})
            return vllm_server, {
                "api_base": server_url,
                "model": model_name_or_path,
            }

        return _init
