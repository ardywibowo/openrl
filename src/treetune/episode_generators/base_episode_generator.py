import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import wandb
from accelerate import PartialState
from datasets import Dataset
from wandb.sdk.wandb_run import Run

from treetune.common import Component, Lazy, Registrable, Tokenizer
from treetune.common.logging_utils import get_logger
from treetune.episodes import Episode
from treetune.inference_strategies import InferenceStrategy

logger = get_logger(__name__)

class EpisodeGeneratorStrategy(Registrable):
    def __call__(self, paths):
        raise NotImplementedError


class EpisodeGenerator(Component):
    can_precompute_episodes: bool = False
    support_distributed: bool = False

    def __init__(
        self,
        num_episodes_per_iteration: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_episodes_per_iteration = num_episodes_per_iteration

    def generate(
        self, iteration: Optional[int] = None
    ) -> Union[List[Episode], Dataset]:
        raise NotImplementedError

    def precompute_episodes(self):
        raise NotImplementedError

    def set_models(self, models_weakref) -> None:
        pass

    def set_trainer(self, trainer_weakref) -> None:
        pass

    def log_episodes(
        self,
        episodes: Union[List[Episode], Dataset],
        iteration_idx: int,
        num_examples: int = 100,
        num_examples_for_wandb: int = 128,
        seed: int = 42,
        log_to_cloud: bool = True,
    ):
        if not self.is_main_process():
            return

        table = wandb.Table(
            columns=[
                "idx",
                "query",
                "response",
                "query_tokens",
                "response_tokens",
                "advantages",
                "reward",
                "instance_length",
            ]
        )

        logger.info(f"Logging {num_examples} examples:")
        rng = random.Random(seed)

        num_console_logs = min(num_examples, len(episodes))
        num_wandb_logs = min(num_examples_for_wandb, len(episodes))
        indices = rng.sample(range(len(episodes)), num_wandb_logs)

        for idx in indices:
            episode = episodes[idx]
            if not isinstance(episode, dict):
                episode = asdict(episode)

            query_token_ids = episode["query_token_ids"]
            response_token_ids = episode["response_token_ids"]
            reward = episode["reward"]

            query_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in query_token_ids
            ]
            query = self.tokenizer.decode(query_token_ids)

            response_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in response_token_ids
            ]
            response = self.tokenizer.decode(response_token_ids)

            advantages = episode.get("advantages")
            instance_length = len(query_token_ids) + len(response_token_ids)

            table.add_data(
                idx,
                query,
                response,
                ", ".join(query_tokens),
                ", ".join(response_tokens),
                ", ".join(
                    [str(a) for a in advantages] if advantages is not None else []
                ),
                reward,
                instance_length,
            )

            if len(table.data) >= num_console_logs:
                continue

            logger.info(f"Example {idx}")
            for k, v in episode.items():
                logger.info(f"{k}: `{v}`")
            logger.info(f"Query: `{query}`")
            logger.info(f"Response: `{response}`")
            logger.info(f"Instance Length: {instance_length}")
            logger.info(f"Reward = Scores: {reward}")

            if advantages is not None:
                # Log aligned advantages with response tokens
                logger.info("Advantages:")
                for i, (adv, tok) in enumerate(zip(advantages, response_tokens)):
                    logger.info(f"{str(i).zfill(4)}: {tok:<20} -> {adv}")

            logger.info("-" * 100)

        if log_to_cloud and self.cloud_logger is not None:
            self.cloud_logger.log({f"episodes_{iteration_idx:04}": table})

    def _should_append_bos_to_query(self) -> bool:
        """
        Determine whether to append BOS to the query based on the tokenizer
        """
        if self.append_bos_to_query != "auto":
            return self.append_bos_to_query

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.bos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append BOS for tokenizer {self.tokenizer.name_or_path}"
            )

    def _should_append_eos_to_response(self) -> bool:
        """
        Determine whether to append EOS to the response based on the tokenizer
        """
        if self.append_eos_to_response != "auto":
            return self.append_eos_to_response

        if "llama" in self.tokenizer.name_or_path.lower():
            assert self.tokenizer.eos_token_id is not None
            return True
        else:
            raise ValueError(
                f"Cannot automatically determine whether to append EOS for tokenizer {self.tokenizer.name_or_path}"
            )



@EpisodeGenerator.register("empty")
class EmptyEpisodeGenerator(EpisodeGenerator):
    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        include_advantages: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.tokenizer is not None
        self.include_advantages = include_advantages

    def generate(self) -> List[Episode]:
        # For now randomly create episodes

        episodes = []

        rng = random.Random(42)

        for i in range(self.num_episodes_per_iteration):
            query_len = rng.randint(5, 15)
            response_len = rng.randint(5, 15)
            query_token_ids = [rng.randint(1, 100) for _ in range(query_len)]
            response_token_ids = [rng.randint(1, 100) for _ in range(response_len)]
            reward = rng.random() * 10
            if self.include_advantages:
                advantages = [rng.random() for _ in range(response_len)]
                advantages = [1.0 for _ in range(response_len)]
            else:
                advantages = None
            episodes.append(
                Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    query_text=self.tokenizer.decode(query_token_ids),
                    response_text=self.tokenizer.decode(response_token_ids),
                    reward=reward,
                    advantages=advantages,
                )
            )

        return episodes


@EpisodeGenerator.register("debug_file")
class DebugFileEpisodeGenerator(EpisodeGenerator):
    def __init__(
        self,
        file_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.debug_data = json.load(open(file_path, "r"))

    def generate(self) -> List[Episode]:
        # For now randomly create episodes

        episodes = []

        all_queries = self.debug_data["query"]
        all_responses = self.debug_data["response"]
        all_rewards = self.debug_data["reward"]

        for i in range(self.num_episodes_per_iteration):
            query_token_ids = all_queries[i]
            response_token_ids = all_responses[i]
            reward = all_rewards[i]
            episodes.append(
                Episode(
                    query_token_ids=query_token_ids,
                    response_token_ids=response_token_ids,
                    query_text=self.tokenizer.decode(query_token_ids),
                    response_text=self.tokenizer.decode(response_token_ids),
                    reward=reward,
                )
            )

        return episodes
