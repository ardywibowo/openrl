import random
import time
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate import PartialState
from accelerate.utils import release_memory
from wandb.sdk.wandb_run import Run

from treetune.common import Lazy
from treetune.common.gpu_utils import get_gpu_memory, wait_for_memory_release
from treetune.common.py_utils import find_n_free_ports
from treetune.common.vllm_server import VLLMServer, compute_vllm_stats
from treetune.logging_utils import get_logger
from . import FromParams

logger = get_logger(__name__)

class Component:
    def __init__(
        self,
        seed: int,
        distributed_state: PartialState, 
        cloud_logger: Optional[Run] = None,
        logger: Logger = None,
        root_dir: Optional[Path] = None,
    ):
        super().__setattr__('_components', {})  # Initialize the component registry
        
        self.seed = seed
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
        self.logger = logger
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self._iteration = 0
        self._metrics = {}
        self._root_dir = self.root_dir / f"iteration__{self._iteration:04d}"

    def is_main_process(self) -> bool:
        return self.distributed_state.is_main_process

    def _cloud_log(self, *args, **kwargs):
        if self.is_main_process() and self.cloud_logger is not None:
            self.cloud_logger.log(*args, **kwargs)
    
    def _log_on_main(self, logger, msg, level="info"):
        if self.is_main_process() and logger is not None:
            getattr(logger, level)(msg)

    def __setattr__(self, name, value):
        if isinstance(value, Component):  # Automatically register subcomponents
            self._components[name] = value
        super().__setattr__(name, value)  # Set attribute normally

    def named_components(self, memo=None, prefix=""):
        if memo is None:
            memo = set()
        
        # Prevent duplicate components
        if self in memo:
            return
        memo.add(self)

        # Yield current component
        yield prefix, self

        # Yield child components
        for name, component in self._components.items():
            if component is None:
                continue
            subcomponent_prefix = f"{prefix}.{name}" if prefix else name
            yield from component.named_components(memo, subcomponent_prefix)

    def apply(self, fn):
        """
        Recursively applies a function `fn` to the current module and all its submodules.
        """
        fn(self)
        for module in self._modules.values():
            if module is not None:
                module.apply(fn)
    
    def init(self, iteration: int):
        self._iteration = iteration
        self._metrics = {}
        for component in self._components.values():
            if component is not None:
                component.init(iteration)
