from pathlib import Path
from typing import Optional

from accelerate import PartialState
from wandb.sdk.wandb_run import Run

from .logging_utils import get_logger
from .registrable import Registrable
from .tokenizer import Tokenizer

logger = get_logger(__name__)

class Component(Registrable):
    def __init__(
        self,
        seed: Optional[int] = None,
        distributed_state: Optional[PartialState] = None, 
        cloud_logger: Optional[Run] = None,
        root_dir: Optional[Path] = None,
        tokenizer: Optional[Tokenizer] = None,
        debug_mode: Optional[bool] = False
    ):
        super().__setattr__('_components', {})  # Initialize the component registry
        
        self.seed = seed
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self._iteration = 0
        self._metrics = {}
        self._root_dir = self.root_dir / f"iteration__{self._iteration:04d}"
        
        self._log_on_main(logger, self)

    def get_process_seed(self) -> int:
        return self.seed + self._iteration * self.distributed_state.num_processes + self.distributed_state.process_index

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
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

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
        """
        Initialize the component and all its subcomponents with the given iteration.
        """
        self._iteration = iteration
        self._metrics = {}

        # Update root directory for the current iteration
        self._root_dir = self.root_dir / f"iteration__{self._iteration:04d}"

        # Recursively initialize all subcomponents
        for _, component in self.named_components():
            if component is not self:
                component.init(iteration)

    def gather_metrics(self) -> dict:
        """
        Gather and combine _metrics from the current Component and all subcomponents.
        """
        combined_metrics = {}

        # Gather metrics from the current component
        combined_metrics.update(self._metrics)

        # Gather metrics from all subcomponents
        for _, component in self.named_components():
            if component is not self:
                combined_metrics.update(component._metrics)

        return combined_metrics
