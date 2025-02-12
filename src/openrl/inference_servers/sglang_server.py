import time
import requests
from pathlib import Path
from typing import Optional, Union

import sglang as sgl
import torch
from sglang.utils import terminate_process, wait_for_server

from openrl.common.logging_utils import get_logger

from .base_server import InferenceServer
from .utils import execute_shell_command

logger = get_logger(__name__)

@InferenceServer.register("sglang")
class SGLangServer(InferenceServer):
    def __init__(
        self,
        port: Optional[int] = 30000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.port = port
        self.server_process = None

    def start_server(
        self,
        hf_ckpt_path_or_model: Union[str, Path],
        log_path: Optional[Path] = None,
    ) -> str:
        if self.server_process is not None and self.server_process.poll() is None:
            raise RuntimeError("Server is already running")
        
        assert self.port is not None
        
        server_url = f"http://localhost:{self.port}"
        if self.is_main_process():
            num_devices = torch.cuda.device_count()
            command = (
                f"python -m sglang.launch_server" 
                f" --model-path {hf_ckpt_path_or_model}"
                f" --port {self.port}" 
                f" --host 0.0.0.0"
                f" --dp-size {num_devices}"
            )
            
            log_path = log_path / "server.log"
            if log_path is not None:
                with log_path.open("w") as f:
                    self.server_process = execute_shell_command(command, f)
            else:
                self.server_process = execute_shell_command(command)
            
            wait_for_server(server_url)
        
        self.distributed_state.wait_for_everyone()
        
        endpoint = sgl.RuntimeEndpoint(server_url)
        sgl.set_default_backend(endpoint)
        
        self.distributed_state.wait_for_everyone()
        return server_url

    def stop_server(self):
        if self.is_main_process():
            if self.server_process is not None:
                terminate_process(self.server_process)
        
        self.distributed_state.wait_for_everyone()