import os
import random
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import psutil
import requests

from treetune.common.logging_utils import get_logger
from treetune.common.notebook_utils import get_repo_dir

from .base_server import InferenceServer
from .utils import (ensure_executable, find_and_kill_process, get_free_port,
                    is_port_in_use_error)

logger = get_logger(__name__)

@InferenceServer.register("sglang")
class SGLangServer(InferenceServer):
    def __init__(
        self,
        script_path: Optional[Path] = None,
        server_running_check_url: str = "v1/models",
        port: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if script_path is None:
            script_path = (
                get_repo_dir() / "scripts" / "start_sglang_server_named_params.sh"
            )
        ensure_executable(script_path)
        
        self.script_path = script_path
        self.port = port
        self.server_running_check_url = server_running_check_url
        self.process: Optional[subprocess.Popen] = None

    def _wait_for_server(
        self,
        launch_func: Callable[[], None],
        timeout: int,
        log_path: Optional[Path] = None,
    ) -> bool:
        """Wait for the server to start by polling the check URL."""
        start_time = time.time()
        while True:
            try:
                response = requests.get(
                    f"http://localhost:{self.port}/{self.server_running_check_url}",
                    proxies={"http": None, "https": None},  # Disabling proxies
                )
                if response.status_code == 200:
                    logger.info("Server is up and responding.")
                    return True
            except requests.ConnectionError:
                # Server is not up yet
                pass
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"An exception occurred while checking the server status: {e}"
                )
                return False

            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for the server to start.")
                return False

            time.sleep(1)

            # Check if the process is still running
            if self.process.poll() is not None:
                logger.error("SGLang process has exited. Restarting...")
                if log_path is not None:
                    with log_path.open("r") as f:
                        server_log = f.read()
                        logger.error(f"SGLang Server log:\n{server_log}")

                        if is_port_in_use_error(server_log):
                            # Get a random number as the port and try again
                            self.port = random.randint(1024, 65533)
                            logger.error(
                                f"Port is already in use. Trying to restart "
                                f"the server using port {self.port}"
                            )

                # Try to restart the server
                launch_func()

    def start_server(
        self,
        hf_ckpt_path_or_model: Union[str, Path],
        log_path: Optional[Path] = None,
        gpu_idx: Optional[int] = None,
        wait_for_response: bool = True,
        timeout: int = 600,
    ) -> str:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Server is already running")

        if self.port is None:
            self.port = get_free_port()

        def launch_func():
            self._launch_process(gpu_idx, hf_ckpt_path_or_model, log_path)

        find_and_kill_process(self.port)
        launch_func()
        logger.info(f"Server started with PID {self.process.pid} on port {self.port}")

        if wait_for_response:
            if not self._wait_for_server(
                launch_func=launch_func, timeout=timeout, log_path=log_path
            ):
                self.stop_server()
                if log_path is not None:
                    with log_path.open("r") as f:
                        logger.error(f"SGLang Server log:\n{f.read()}")
                raise RuntimeError("Server did not start within the expected time.")

        server_url = f"http://localhost:{self.port}/v1"
        return server_url

    def _launch_process(self, gpu_idx, hf_ckpt_path_or_model, log_path):
        # The command arguments:
        command = (
            f"{self.script_path}"
            f" --model {hf_ckpt_path_or_model}"
            f" --port {self.port}"
            f" --random-seed {self.seed}"
            f" --cpu-offload-gb {self.swap_space}"
            f" --mem-fraction-static {self.gpu_memory_utilization}"
        )
        if gpu_idx is not None:
            command += f" --gpu-idx {gpu_idx}"
        
        command = shlex.split(command)
        # Redirect both stdout and stderr to the log file if specified
        if log_path is not None:
            with log_path.open("w") as f:
                self.process = subprocess.Popen(command, stdout=f, stderr=f)
        else:
            self.process = subprocess.Popen(command)

    def stop_server(self):
        if self.process is None or self.process.poll() is not None:
            logger.info("Server is not running or already stopped.")
            return

        pid = self.process.pid
        logger.info(f"Stopping server with PID {pid}.")
        
        try:
            # Terminate the main process
            self.process.terminate()
            self.process.wait(timeout=5)
        except psutil.NoSuchProcess:
            logger.warning(f"Process PID: {pid} already terminated.")
        except subprocess.TimeoutExpired:
            logger.error(f"Main process PID: {pid} did not terminate, killing it.")
            self.process.kill()
        except Exception as e:
            logger.error(f"Error stopping process PID: {pid}: {e}")
