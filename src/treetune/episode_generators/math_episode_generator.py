import json
import uuid
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
from datasets import Dataset

from treetune.common.logging_utils import get_logger
from treetune.episode_generators import (EpisodeGenerator,
                                         EpisodeGeneratorWithRewardFunction)
from treetune.episodes import Episode

logger = get_logger(__name__)

@EpisodeGenerator.register("math_episode_generator")
class MathEpisodeGenerator(EpisodeGeneratorWithRewardFunction):
    def __init__(
        self,
        reasoning_step_delimiter: Optional[str] = None,
        answer_prefix: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.reasoning_step_delimiter = reasoning_step_delimiter
        self.answer_prefix = answer_prefix

        try:
            self._bleu_metric = evaluate.load(
                "bleu",
                experiment_id=uuid.uuid4().hex,
            )
        except Exception as e:
            logger.error(f"Failed to load BLEU metric: {e}")
            self._bleu_metric = None

        assert hasattr(
            self.task, "split_solution_into_intermediate_steps"
        ), f"Task {self.task} does not have a method `split_solution_into_intermediate_steps`."

    def compute_number_of_reasoning_steps(self, response_text: str) -> int:
        # noinspection PyUnresolvedReferences
        indices = self.task.split_solution_into_intermediate_steps(response_text)
        return len(indices) - 1

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes = []
        metrics = {}
        for instance in inference_results:
            tree = json.loads(instance["_treetune__reasoning_tree"])
            paths = self.extract_paths_from_tree(tree)
            all_rewards = []
            all_responses = []
            for path in paths:
                # noinspection DuplicatedCode
                assert len(path["node_chain"]) == 2, "Does not support multi-hop paths."

                finish_reason = path["node_chain"][-1]["finish_reason"]
                query_text = path["node_chain"][0]["text"]
                full_text = path["node_chain"][-1]["full_text"]
                response_text = full_text[len(query_text) :]

                try:
                    num_reasoning_steps = self.compute_number_of_reasoning_steps(
                        response_text
                    )
                    metrics.setdefault("num_reasoning_steps", []).append(
                        num_reasoning_steps
                    )
                    metrics.setdefault("parse_failed", []).append(False)
                except Exception as e:
                    logger.error(f"Parsing reasoning steps failed {e}")
                    logger.error(f"Response: `{response_text}`")
                    metrics.setdefault("parse_failed", []).append(True)

                if finish_reason != "length":
                    # Generation stopped because the model hit <eos>
                    reward, is_unfinished_response = self.reward_function(
                        query_text, response_text, instance
                    )
                else:
                    # Generation stopped because the model hit the `max_tokens` limit
                    reward = self.reward_function.get_unfinished_response_penalty()
                    is_unfinished_response = True

                try:
                    query_token_ids, response_token_ids = (
                        self._tokenize_query_and_response(
                            query_text,
                            response_text,
                            # Only append EOS token if the response is complete
                            allow_append_eos=not is_unfinished_response,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to tokenize query and response for instance {instance['__uuid__']}: {e}"
                    )
                    logger.error(f"Query: {query_text}")
                    logger.error(f"Response: {response_text}")
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                all_responses.append(response_text)

                if self.max_sequence_length is not None:
                    seq_len = len(query_token_ids) + len(response_token_ids)
                    if seq_len > self.max_sequence_length:
                        logger.warning(
                            f"Sequence length {seq_len} is greater than "
                            f"max sequence length {self.max_sequence_length}."
                        )

                        # Truncate the response
                        response_token_ids = response_token_ids[
                            : self.max_sequence_length - len(query_token_ids)
                        ]
                        reward = self.reward_function.get_unfinished_response_penalty()
                        is_unfinished_response = True

                if len(response_token_ids) == 0:
                    logger.warning(
                        f"Response token ids are empty for instance {instance['__uuid__']}"
                    )
                    metrics.setdefault("empty_response", []).append(True)
                    continue

                metrics.setdefault("empty_response", []).append(False)
                metrics.setdefault("is_unfinished_response", []).append(
                    is_unfinished_response
                )

                episode = Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    scores=float(reward),
                )

                episodes.append(episode)
                all_rewards.append(float(reward))

            if len(all_rewards) > 0:
                once_hit = any([r == 1.0 for r in all_rewards])
                metrics.setdefault("once_hit", []).append(float(once_hit))

            if len(all_responses) > 1:
                metrics.setdefault("num_unique_responses", []).append(
                    len(set(all_responses))
                )
                if self._bleu_metric is not None:
                    bleu = self._avg_bleu_of_pairs_of_response(all_responses)
                    metrics.setdefault("trajectory_bleu", []).append(bleu)

        if "is_unfinished_response" in metrics:
            metrics["is_unfinished_response"] = sum(
                metrics["is_unfinished_response"]
            ) / len(metrics["is_unfinished_response"])

        if "empty_response" in metrics:
            metrics["empty_response"] = sum(metrics["empty_response"]) / len(
                metrics["empty_response"]
            )

        if "num_reasoning_steps" in metrics:
            num_reasoning_steps = np.array(metrics.pop("num_reasoning_steps"))
            metrics["num_reasoning_steps/dist"] = num_reasoning_steps
            metrics["num_reasoning_steps/mean"] = np.mean(num_reasoning_steps)

        if "parse_failed" in metrics:
            metrics["parse_failed"] = sum(metrics["parse_failed"]) / len(
                metrics["parse_failed"]
            )

        if "once_hit" in metrics:
            metrics["once_hit"] = sum(metrics["once_hit"]) / len(metrics["once_hit"])

        if "trajectory_bleu" in metrics:
            metrics["trajectory_bleu"] = sum(metrics["trajectory_bleu"]) / len(
                metrics["trajectory_bleu"]
            )

        if len(metrics) > 0:
            logs = {f"episodes_metric/{k}": v for k, v in metrics.items()}
            self._cloud_log({**logs, "train/global_iteration": iteration})

        return episodes

    # noinspection DuplicatedCode
    def _avg_bleu_of_pairs_of_response(self, response: List[str]) -> float:
        preds = []
        refs = []
        for i in range(len(response)):
            for j in range(i + 1, len(response)):
                sen_1 = response[i]
                sen_2 = response[j]
                preds.append(sen_1)
                refs.append(sen_2)
        bleu_full_stats = self._bleu_metric.compute(predictions=preds, references=refs)
        bleu = bleu_full_stats["bleu"]
        return bleu
