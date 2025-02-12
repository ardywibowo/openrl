import os
import socket
import subprocess
from pathlib import Path
from typing import Union

import psutil

from openrl.common.logging_utils import get_logger

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
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            connections = proc.net_connections()
            for conn in connections:
                if conn.laddr.port == port:
                    logger.info(f"Killing process {proc.info['name']} (PID {proc.info['pid']}) using port {port}")
                    os.kill(proc.info["pid"], 9)
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


def is_port_in_use_error(server_log: str) -> bool:
    server_log = server_log.lower()
    return (
        "error while attempting to bind on address" in server_log
        and "address already in use" in server_log
    )

def execute_shell_command(command: str, log_file = None) -> subprocess.Popen:
    """
    Execute a shell command and return the process handle

    Args:
        command: Shell command as a string (can include \\ line continuations)
    Returns:
        subprocess.Popen: Process handle
    """
    # Replace \ newline with space and split
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    
    if log_file is not None:
        return subprocess.Popen(parts, text=True, stdout=log_file, stderr=log_file)
    else:
        return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)