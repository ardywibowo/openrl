import os
import socket
from pathlib import Path
from typing import Union

import psutil

from treetune.common.logging_utils import get_logger

logger = get_logger(__name__)

def get_free_port() -> int:
    """Find a free port by binding to port 0 and then releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def ensure_executable(script_path: Union[str, Path]):
    """Make sure the server script is executable."""
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, os.stat(script_path).st_mode | 0o111)

def find_and_kill_process(port: int):
    for proc in psutil.process_iter(["pid", "name", "connections"]):
        try:
            connections = proc.info["connections"]
            if connections is None:
                continue

            for conn in connections:
                if conn.laddr.port == port:
                    # If the port matches, print process info and kill the process
                    logger.info(
                        f"Found process {proc.info['name']} with PID {proc.info['pid']} using port {port}"
                    )
                    os.kill(proc.info["pid"], 9)
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def is_port_in_use_error(server_log: str) -> bool:
    server_log = server_log.lower()
    return (
        "error while attempting to bind on address" in server_log
        and "address already in use" in server_log
    )
