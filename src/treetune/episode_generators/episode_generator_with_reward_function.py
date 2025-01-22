import random
import json
from typing import List, Union, Dict, Any, Tuple, Optional

from datasets import Dataset

from treetune.common import Registrable, Lazy
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.episodes import Episode
from treetune.episode_generators.on_policy_episode_generator import (
    OnPolicyEpisodeGenerator,
)
from treetune.episode_generators.tree_episode_generator import TreeEpisodeUtils
from treetune.reward_functions import RewardFunction
from treetune.logging_utils import get_logger

logger = get_logger(__name__)

@EpisodeGenerator.register("episode_generator_with_reward_function")
class EpisodeGeneratorWithRewardFunction(OnPolicyEpisodeGenerator, TreeEpisodeUtils):
    def __init__(
        self,
        reward_function: Lazy[RewardFunction],
        append_bos_to_query: Union[str, bool] = "auto",
        append_eos_to_response: Union[str, bool] = "auto",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_function = reward_function.construct(
            seed=self.seed,
            distributed_state=self.distributed_state,
            cloud_logger=self.cloud_logger,
            root_dir= self.root_dir / "reward_function",
            tokenizer=self.tokenizer
        )
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        episodes_without_rewards = []
        for instance in inference_results:
            episodes = self._convert_to_episodes(instance)
            episodes_without_rewards.extend(episodes)
        
        episodes = self.reward_function.batch_compute_rewards(
            episodes_without_rewards, iteration)

        return episodes

    def _convert_to_episodes(self, instance: Dict[str, Any]) -> List[Episode]:
        tree = json.loads(instance["_treetune__reasoning_tree"])
        paths = self.extract_paths_from_tree(tree)
        
        return [self._convert_path_to_episode(instance, path) for path in paths]

    def _convert_path_to_episode(
        self, instance: Dict[str, Any], path: Dict[str, Any]
    ) -> List[Episode]:
        assert len(path["node_chain"]) == 2
        
        query_text = path["node_chain"][0]["text"]
        full_text = path["node_chain"][-1]["full_text"]
        response_text = full_text[len(query_text) :]
        
        finish_reason = path["node_chain"][-1]["finish_reason"]
        is_chopped = finish_reason == "length"

        try:
            query_token_ids, response_token_ids = self._tokenize_query_and_response(
                query_text, response_text, not is_chopped
            )
        except Exception as e:
            logger.error(
                f"Failed to tokenize query and response for instance {instance['_treetune__idx']}"
            )
            logger.error(f"Query: {query_text}")
            logger.error(f"Response: {response_text}")
            return []

        episode = Episode(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query_text=query_text,
            response_text=response_text,
            scores=None,
        )
        return episode

    def _tokenize_query_and_response(
        self, query: str, response: str, allow_append_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        # This a legacy method that is not used anymore. It is kept here for reference.
        return self._tokenize_trajectory(
            {"query_text": query, "response_text": response},
            is_unfinished_response=not allow_append_eos,
            return_offsets=False,
        )

    def _tokenize_trajectory(
        self,
        trajectory: Dict[str, Any],
        is_unfinished_response: bool = False,
        return_offsets: bool = False,
        safety_check_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[List[int], List[int]],
        Tuple[List[int], List[int], List[Tuple[int, int]]],
    ]:
        safety_check_kwargs = safety_check_kwargs or {}
        query_text = trajectory["query_text"]
        response_text = trajectory["response_text"]

        episode_text = f"{query_text}{response_text}"
        episode_encoding = self.tokenizer(
            episode_text,
            add_special_tokens=False,  # We will add BOS and EOS tokens at the end
            return_offsets_mapping=True,
        )

        token_ids = episode_encoding["input_ids"]
        offsets = episode_encoding["offset_mapping"]

        response_start_index = next(
            i for i, (start, end) in enumerate(offsets) if start >= len(query_text)
        )
        query_token_ids = token_ids[:response_start_index]
        response_token_ids = token_ids[response_start_index:]

        self._safety_check_tokenization(
            query_token_ids=query_token_ids,
            response_token_ids=response_token_ids,
            query=query_text,
            response=response_text,
            episode_text=episode_text,
            **safety_check_kwargs,
        )

        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids

        if not is_unfinished_response and self._should_append_eos_to_response():
            response_token_ids = response_token_ids + [self.tokenizer.eos_token_id]

        if return_offsets:
            return query_token_ids, response_token_ids, offsets
        else:
            return query_token_ids, response_token_ids

    def _safety_check_tokenization(
        self,
        query_token_ids: List[str],
        response_token_ids: List[str],
        query: str,
        response: str,
        episode_text: str,
        check_query_reconstruction: bool = True,
        check_response_reconstruction: bool = True,
    ):
        decoding_kwargs = {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        decoded_instance = self.tokenizer.decode(
            query_token_ids + response_token_ids, **decoding_kwargs
        )
        assert decoded_instance == episode_text, (
            f"Decoded instance does not match original instance.\n"
            f"Original instance: {episode_text}\n"
            f"Decoded instance: {decoded_instance}"
        )
        
        if check_query_reconstruction:
            decoded_query = self.tokenizer.decode(query_token_ids, **decoding_kwargs)
            assert decoded_query == query, (
                f"Decoded query does not match original query.\n"
                f"Original query: {query}\n"
                f"Decoded query: {decoded_query}"
            )
        
        if check_response_reconstruction:
            decoded_response = self.tokenizer.decode(
                response_token_ids, **decoding_kwargs
            )
            assert decoded_response == response, (
                f"Decoded response does not match original response.\n"
                f"Original response: {response}\n"
                f"Decoded response: {decoded_response}"
            )
