from typing import Any, Dict, Tuple

from openrl.common.logging_utils import get_logger
from openrl.reward_functions import RewardFunction

logger = get_logger(__name__)


@RewardFunction.register("simple_math_reward_function")
class SimpleMATHRewardFunction(RewardFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, 
        response: Dict[str, Any], 
        dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        pred_answer = response['response']
        gold_answer = dataset_instance["answer"]
        if pred_answer == gold_answer:
            return 1.0
        elif gold_answer in pred_answer:
            return 0.5
        else:
            return 0.0