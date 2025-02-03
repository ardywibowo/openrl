import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate.utils import release_memory

from treetune.common.component import Component
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.lazy import Lazy
from treetune.common.logging_utils import get_logger
from treetune.common.py_utils import find_n_free_ports

from .base_server import InferenceServer

logger = get_logger(__name__)

class InferenceServerHandler(Component):
    def __init__(
        self,
        inference_server: Lazy[InferenceServer],
        gpu_memory_utilization: Union[float, str] = 0.9,
        min_available_gpu_memory_mb: Optional[int] = None,
        wait_until_memory_release: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.inference_server_lazy = inference_server
        self.gpu_memory_utilization = gpu_memory_utilization
        self.min_available_gpu_memory_mb = min_available_gpu_memory_mb
        self.wait_until_memory_release = wait_until_memory_release
        
        self._port_rng = random.Random(self.seed)
        self._port: Optional[int] = None
        self._server: Optional[InferenceServer] = None
        self._server_configs: Optional[Dict[str, Any]] = None
        self._cleanup_fn: Optional[Callable] = None
    
    def get_or_create_server_with_model(self, model_name_or_path: str, results_dir: Path) -> Dict[str, Any]:
        """
        Returns configs that contain the API URL and model name of the inference server.
        """
        
        if (
            self._server is not None and 
            self._server_configs is not None and 
            self._server_configs.get("model", "") == model_name_or_path
        ):
            return self._server_configs
        
        if self._server is not None:
            self.kill_server()
        
        release_memory()
        
        init_fn, cleanup_fn = self.get_server_init_and_cleanup_fn(
            results_dir=results_dir,
            model_name_or_path=model_name_or_path,
        )
        self._server, self._server_configs = init_fn()
        self._cleanup_fn = cleanup_fn
        
        return self._server_configs
    
    def kill_server(self):
        if self._server is None:
            return
        
        self._server.stop_server()
        # del self._server
        # del self._server_configs
        
        self._server = None
        self._server_configs = None
        
        self._cleanup_fn()
        release_memory()
        logger.info(f"Rank {self.distributed_state.process_index} stopped inference server.")
        
        self.distributed_state.wait_for_everyone()
    
    def get_server_init_and_cleanup_fn(
        self,
        results_dir: Path,
        model_name_or_path: str,
    ) -> Callable[[], Tuple[InferenceServer, Dict[str, Any]]]:
        this_process_device = self.distributed_state.device
        if self.min_available_gpu_memory_mb is not None:
            total_mem_mb = (
                torch.cuda.get_device_properties(this_process_device.index).total_memory
                / 1024**2
            )
            used_threshold_mb = total_mem_mb - self.min_available_gpu_memory_mb
            logger.info(
                f"Need at least {self.min_available_gpu_memory_mb}. "
                f"Waiting for GPU{this_process_device.index} used memory to be below {used_threshold_mb} MB. "
                f"Total GPU memory: {total_mem_mb} MB."
            )
            wait_for_memory_release(
                this_process_device.index,
                threshold_mb=used_threshold_mb,
            )
        
        inference_server_lazy = self.inference_server_lazy
        gpu_memory_utilization = self.gpu_memory_utilization
        self._set_server_ports()
        server_port = self._port
        if gpu_memory_utilization == "auto":
            # Compute the GPU utilization based on amount of remaining memory
            allocated_mem_mb = get_gpu_memory()[self.distributed_state.process_index]
            total_mem_mb = (
                torch.cuda.get_device_properties(self.distributed_state.process_index).total_memory / 1024**2
            )

            remaining_mem_mb = (
                total_mem_mb - allocated_mem_mb
            ) * 0.9  # Allow for 10% tolerance
            gpu_memory_utilization = round(remaining_mem_mb / total_mem_mb, 2)

            logger.info(
                f"GPU #{self.distributed_state.process_index} Auto-computed inference server GPU memory utilization: {gpu_memory_utilization}. "
                f"Currently Allocated: {allocated_mem_mb} MB, "
                f"Total: {total_mem_mb} MB, "
                f"Remaining: {remaining_mem_mb} MB."
            )

        def _server_init_fn() -> Tuple[InferenceServer, Dict[str, Any]]:
            server_log_path = results_dir / "server.log"

            logger.info(
                f"Rank #{self.distributed_state.process_index} starting inference server: "
                f"model={model_name_or_path}   port={server_port}   seed={self.get_process_seed()}"
            )
            t0 = time.time()
            inference_server = inference_server_lazy.construct(
                seed=self.get_process_seed(),
                port=server_port,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            server_url = inference_server.start_server(
                hf_ckpt_path_or_model=model_name_or_path,
                gpu_idx=self.distributed_state.process_index,
                wait_for_response=True,
                log_path=server_log_path,
                timeout=800,
            )
            self._cloud_log(
                {
                    "timing/episode_generation/inference_server_start": time.time() - t0,
                }
            )
            
            server_configs = {
                "api_base": server_url,
                "model": model_name_or_path,
            }
            return inference_server, server_configs
        
        device = self.distributed_state.device
        release_memory()
        gpu_memory_before_mb = get_gpu_memory()[device.index]
        def _server_cleanup_fn():
            if self.wait_until_memory_release:
                threshold_mb = gpu_memory_before_mb * 1.1  # 10% tolerance
                wait_for_memory_release(device.index, threshold_mb=threshold_mb)

        return _server_init_fn, _server_cleanup_fn

    def compute_server_stats(self, results_dir: Path) -> Dict[str, Any]:
        if self.distributed_state.is_main_process:
            try:
                log_path = results_dir / "inference_server.log"
                stats = self._server.compute_stats(log_path)
            except Exception as e:
                logger.error(f"Error while computing inference server stats: {e}")
                stats = {}

            if "avg_generation_throughput" in stats:
                stats["total_approx_generation_throughput"] = (
                    stats["avg_generation_throughput"] 
                    * self.distributed_state.num_processes
                )

            stats = {f"inference_server_stats/{k}": round(v, 2) for k, v in stats.items()}
            logger.info(f"Inference Server Stats: {stats}")
            self._metrics = stats
        
        self.distributed_state.wait_for_everyone()

    def _set_server_ports(self) -> None:
        """
        On the main process, find free ports for all ranks. 
        Broadcast them so each rank knows which port to use.
        """
        if self.distributed_state.process_index == 0:
            ports = find_n_free_ports(
                self.distributed_state.num_processes,
                generator=self._port_rng
            )
            logger.info(f"Found free ports: {ports}")
        else:
            ports = [0] * self.distributed_state.num_processes

        from accelerate.utils import broadcast_object_list
        ports = broadcast_object_list(ports, from_process=0)
        release_memory()
        self._port = ports[self.distributed_state.process_index]
        logger.info(
            f"Rank {self.distributed_state.process_index} using inference server port {self._port}"
        )
