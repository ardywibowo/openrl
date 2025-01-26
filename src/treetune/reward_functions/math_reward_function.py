from typing import Any, Dict, List, Optional, Tuple

from treetune.common.logging_utils import get_logger
from treetune.episodes import Episode
from treetune.reward_functions import RewardFunction
from treetune.tasks import GSM8K, Task
from treetune.tasks.math import MATH

logger = get_logger(__name__)


@RewardFunction.register("math_reward_function")
class MATHRewardFunction(RewardFunction):
    def __init__(
        self,
        math_task: Task,
        penalize_unfinished_response: bool = False,
        unfinished_response_penalty: float = -1.0,
        timeout: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(math_task, (MATH, GSM8K))
        self.math_task = math_task
        self.penalize_unfinished_response = penalize_unfinished_response
        self.unfinished_response_penalty = unfinished_response_penalty
        self.timeout = timeout

    def __call__(
        self, 
        query: str,
        response: str, 
        dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        pred_answer = self.math_task.extract_predicted_answer_from_text(
            response, dataset_instance["problem"]
        )
        is_unfinished_response = pred_answer is None
        if is_unfinished_response and self.penalize_unfinished_response:
            return float(self.get_unfinished_response_penalty()), is_unfinished_response

        gold_answer = dataset_instance["answer"]
        reward = self.math_task.grade_answer(
            given_answer=pred_answer,
            ground_truth=gold_answer,
            item=dataset_instance,
            timeout=self.timeout,
        )
        
        return float(reward), is_unfinished_response
    
    def batch_compute_rewards(
        self,
        episodes_without_rewards: List[Episode], 
        instances: List[Dict[str, Any]], 
        paths: List[Dict[str, Any]]
    ) -> Tuple[List[Episode], Dict[str, Any]]:
        episodes_with_rewards = []
        for episode, instance, path in zip(episodes_without_rewards, instances, paths):
            query_text = path["node_chain"][0]["text"]
            full_text = path["node_chain"][-1]["full_text"]
            response_text = full_text[len(query_text):]
            
            reward = self.compute_reward(response_text, instance)
            episode.reward = reward
            episodes_with_rewards.append(episode)
        
        metrics = {}
        if self.penalize_unfinished_response:
            metrics = self.record_num_unfinished_responses(episodes_with_rewards)
        
        return episodes_with_rewards, metrics
    
    def record_num_unfinished_responses(self, episodes: int):
        num_unfinished = sum(
            1 for e in episodes if e.scores == self.get_unfinished_response_penalty()
        )
        num_total = len(episodes)
        return {
            "episodes_metric/is_unfinished_response": (
                num_unfinished / num_total
            )
        }
    
    def get_unfinished_response_penalty(self) -> float:
        return self.unfinished_response_penalty
