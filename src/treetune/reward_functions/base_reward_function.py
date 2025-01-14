from typing import Any, Dict, List, Tuple, Optional

from accelerate import PartialState
from wandb.sdk.wandb_run import Run

from treetune.common import Registrable
from treetune.episodes import Episode


class RewardFunction(Registrable):
    def __init__(
        self, 
        distributed_state: PartialState, 
        cloud_logger: Optional[Run] = None
    ) -> None:
        self.distributed_state = distributed_state
        self.cloud_logger = cloud_logger
    
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
    ) -> Tuple[List[Episode], Dict[str, Any]]:
        raise NotImplementedError
