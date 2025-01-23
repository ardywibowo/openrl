import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate.utils import release_memory

from .component import Component
from .gpu_utils import get_gpu_memory, wait_for_memory_release
from .lazy import Lazy
from .logging_utils import get_logger
from .py_utils import find_n_free_ports
from .vllm_server import VLLMServer, compute_vllm_stats

logger = get_logger(__name__)

class VLLMServerHandler(Component):
    def __init__(
        self,
        vllm_server: Lazy[VLLMServer],
        gpu_memory_utilization: Union[float, str] = 0.9,
        min_available_gpu_memory_mb: Optional[int] = None,
        wait_until_memory_release: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vllm_server_lazy = vllm_server
        self.gpu_memory_utilization = gpu_memory_utilization
        self.min_available_gpu_memory_mb = min_available_gpu_memory_mb
        self.wait_until_memory_release = wait_until_memory_release
        
        self._port_generator_rng = random.Random(self.seed)
        self._vllm_port: Optional[int] = None
        self._vllm_server: Optional[VLLMServer] = None
        self._vllm_server_configs: Optional[Dict[str, Any]] = None
        self._vllm_cleanup_fn: Optional[Callable] = None
    
    def get_or_create_vllm_server_with_model(self, model_name_or_path: str, results_dir: Path) -> Dict[str, Any]:
        """
        Returns configs that contain the API URL and model name of the VLLM Server.
        """
        
        if (
            self._vllm_server is not None and 
            self._vllm_server_configs is not None and 
            self._vllm_server_configs.get("model", "") == model_name_or_path
        ):
            return self._vllm_server_configs
        
        if self._vllm_server is not None:
            self.kill_server()
        
        release_memory()
        
        init_fn, cleanup_fn = self.get_vllm_init_and_cleanup_fn(
            results_dir=results_dir,
            model_name_or_path=model_name_or_path,
        )
        self._vllm_server, self._vllm_server_configs = init_fn()
        self._vllm_cleanup_fn = cleanup_fn
        
        return self._vllm_server_configs
    
    def kill_server(self):
        if self._vllm_server is None:
            return
        
        self._vllm_server.stop_server()
        del self._vllm_server
        del self._vllm_server_configs
        self._vllm_cleanup_fn()
        release_memory()
        logger.info(f"Rank {self.distributed_state.process_index} stopped vLLM server.")
    
    def get_vllm_init_and_cleanup_fn(
        self,
        results_dir: Path,
        model_name_or_path: str,
    ) -> Callable[[], Tuple[VLLMServer, Dict[str, Any]]]:
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
        
        vllm_server_lazy = self.vllm_server_lazy
        gpu_memory_utilization = self.gpu_memory_utilization
        self._set_vllm_ports()
        vllm_port = self._vllm_port
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
                f"GPU #{self.distributed_state.process_index} Auto-computed vLLM GPU memory utilization: {gpu_memory_utilization}. "
                f"Currently Allocated: {allocated_mem_mb} MB, "
                f"Total: {total_mem_mb} MB, "
                f"Remaining: {remaining_mem_mb} MB."
            )

        def _vllm_init_fn() -> Tuple[VLLMServer, Dict[str, Any]]:
            vllm_log_path = results_dir / "vllm_server.log"

            logger.info(
                f"Rank #{self.distributed_state.process_index} starting vLLM: "
                f"model={model_name_or_path}   port={vllm_port}   seed={self.get_process_seed()}"
            )
            t0 = time.time()
            vllm_server = vllm_server_lazy.construct(
                seed=self.get_process_seed(),
                port=vllm_port,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            server_url = vllm_server.start_server(
                hf_ckpt_path_or_model=model_name_or_path,
                gpu_idx=self.distributed_state.process_index,
                wait_for_response=True,
                log_path=vllm_log_path,
                timeout=800,
            )
            self._cloud_log(
                {
                    "timing/episode_generation/vllm_start": time.time() - t0,
                }
            )
            
            vllm_configs = {
                "api_base": server_url,
                "model": model_name_or_path,
            }
            return vllm_server, vllm_configs
        
        device = self.distributed_state.device
        release_memory()
        gpu_memory_before_mb = get_gpu_memory()[device.index]
        def _vllm_cleanup_fn():
            if self.wait_until_memory_release:
                threshold_mb = gpu_memory_before_mb * 1.1  # 10% tolerance
                wait_for_memory_release(device.index, threshold_mb=threshold_mb)

        return _vllm_init_fn, _vllm_cleanup_fn

    def _compute_and_log_vllm_stats(self) -> Dict[str, Any]:
        if self.distributed_state.is_main_process:
            try:
                log_path = self.root_dir / "vllm_server.log"
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
            self._metrics = vllm_stats
        
        self.distributed_state.wait_for_everyone()

    def _set_vllm_ports(self) -> None:
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
