from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from accelerate import PartialState
from wandb.sdk.wandb_run import Run

from treetune.common import Registrable
from treetune.episodes import Episode


class RewardFunction(Registrable):
    def __init__(
        self, 
        seed: int,
        distributed_state: PartialState, 
        cloud_logger: Optional[Run] = None,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.seed = seed
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
        self.root_dir = root_dir
    
    def get_unfinished_response_penalty(self) -> float:
        raise NotImplementedError

    def __call__(
        self, query: str, response: str, dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        raise NotImplementedError

    def is_unfinished_response(
        self, response: str, dataset_instance: Dict[str, Any]
    ) -> bool:
        raise NotImplementedError
    
    def is_main_process(self) -> bool:
        return self.distributed_state.is_main_process
    
    def batch_compute_rewards(
        self, 
        episodes_without_rewards: List[Episode],
        iteration: Optional[int] = None
    ) -> List[Episode]:
        raise NotImplementedError
    
    def _cloud_log(self, *args, **kwargs):
        if self.is_main_process() and self.cloud_logger is not None:
            self.cloud_logger.log(*args, **kwargs)
