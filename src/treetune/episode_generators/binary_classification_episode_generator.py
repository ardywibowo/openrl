import random
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict

import wandb
from datasets import Dataset
from tqdm import tqdm

from treetune.common.py_utils import format_string
from treetune.episode_generators.base_episode_generator import (
    EpisodeGenerator,
    Episode,
)
from treetune.logging_utils import get_logger
from treetune.tasks.base_task import Task

logger = get_logger(__name__)

@dataclass
class BinaryClassificationEpisode:
    query_token_ids: List[int]
    target_prob: float

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert self.target_prob is not None


@EpisodeGenerator.register("binary_classification")
class BinaryClassificationEpisodeGenerator(EpisodeGenerator):
    """
    A static episode generator that just converts task examples into episodes.
    """

    can_precompute_episodes = True

    def __init__(
        self,
        query_template: str,
        task: Task,
        append_bos_to_query: Union[str, bool] = "auto",
        append_eos_to_response: Union[str, bool] = "auto",
        task_split: str = "train",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_template = query_template
        self.append_bos_to_query = append_bos_to_query
        self.append_eos_to_response = append_eos_to_response
        self.task = task
        self.episode_cache = None

        if not self.is_main_process():
            return

        self._ds = self.task.get_datasets(split=task_split)

        # Get the dataset fields that are used in the templates
        query_format_keys = []
        for column in self._ds.column_names:
            if f"{{{column}}}" in self.query_template:
                query_format_keys.append(column)

        self.query_format_keys = query_format_keys

        logger.info(f"Number of examples in dataset: {len(self._ds)}")
        logger.info(f"query_format_keys: {self.query_format_keys}.")

        if append_bos_to_query == "auto":
            logger.info(
                f"append_bos_to_query is set to 'auto', which is {self._should_append_bos_to_query()}"
            )
        if append_eos_to_response == "auto":
            logger.info(
                f"append_eos_to_response is set to 'auto', which is {self._should_append_eos_to_response()}"
            )

        self.has_warned_about_decoding_mismatch = False

    def generate(self) -> Dataset:
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"

        if self.episode_cache is None:
            logger.warning(
                "`generate` is called before populating the cache. "
                "This isn't probably the intended use case of this class. "
                "But we will precompute the episodes now."
            )
            self.precompute_episodes()

        episodes = [] + self.episode_cache
        logger.info(f"Number of episodes in cache before generation: {len(episodes)}")

        random.shuffle(episodes)
        # update cache with left over episodes
        if len(episodes) > self.num_episodes_per_iteration:
            extra_episodes = episodes[self.num_episodes_per_iteration :]
            episodes = episodes[: self.num_episodes_per_iteration]
            self.episode_cache = extra_episodes
            logger.info(
                f"Number of episodes in cache after generation: {len(self.episode_cache)}"
            )

        # Won't need the dataset anymore
        # del self._ds
        # del self.episode_cache

        episodes = Dataset.from_dict(
            {k: [getattr(e, k) for e in episodes] for k in episodes[0].__dict__.keys()}
        )

        import gc

        gc.collect()

        return episodes

    def precompute_episodes(self):
        assert (
            self.is_main_process()
        ), "This method should only be called on the main process"

        self.episode_cache = []
        for example in tqdm(
            self._ds, desc="Precomputing episodes", total=len(self._ds)
        ):
            episode = self._convert_example_to_episode(example)
            self.episode_cache.append(episode)

    def _convert_example_to_episode(self, example: Dict[str, Any]) -> Episode:
        query_format_kwargs = {
            key: example[key] for key in self.query_format_keys if key in example
        }
        query = format_string(self.query_template, **query_format_kwargs)
        query_token_ids = self._tokenize_query(query)
        
        target_prob = example["target_prob"]
        
        return BinaryClassificationEpisode(
            query_token_ids=query_token_ids,
            target_prob=target_prob
        )

    def _tokenize_query(self, query: str) -> Tuple[List[int], List[int]]:
        instance_text = f"{query}"
        instance_encoding = self.tokenizer(
            instance_text,
            add_special_tokens=False,  # We already added BOS and EOS tokens at the end
        )
        
        token_ids = instance_encoding["input_ids"]
        
        # Split the token IDs into query and response parts
        query_token_ids = token_ids
        
        # Check that the decoded text matches the original text
        if not self.has_warned_about_decoding_mismatch:
            decoded_query = self.tokenizer.decode(
                query_token_ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if decoded_query != query:
                logger.warning(
                    f"Decoded query does not match original query.\n"
                    f"Original query: {query}\n"
                    f"Decoded query: {decoded_query}"
                )
            
            self.has_warned_about_decoding_mismatch = True
        
        # We manually add BOS and EOS tokens to the query and response
        # just to be very explicit about them. `add_special_tokens=True` may not
        # always add BOS and EOS tokens.
        if self._should_append_bos_to_query():
            query_token_ids = [self.tokenizer.bos_token_id] + query_token_ids
        
        if self._should_append_eos_to_response():
            query_token_ids = query_token_ids + [self.tokenizer.eos_token_id]
        
        return query_token_ids

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
                "query_tokens",
                "target_prob",
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
            target_prob = episode["target_prob"]
            
            query_tokens = [
                (
                    self.tokenizer.convert_ids_to_tokens(tok_id)
                    if tok_id >= 0
                    else str(tok_id)
                )
                for tok_id in query_token_ids
            ]
            query = self.tokenizer.decode(query_token_ids)
            
            instance_length = len(query_token_ids)
            
            table.add_data(
                idx,
                query,
                ", ".join(query_tokens),
                target_prob,
                instance_length,
            )
            
            if len(table.data) >= num_console_logs:
                continue
            
            logger.info(f"Example {idx}")
            for k, v in episode.items():
                logger.info(f"{k}: `{v}`")
            logger.info(f"Query: `{query}`")
            logger.info(f"Instance Length: {instance_length}")
            logger.info(f"Target Prob: {target_prob}")
            logger.info("-" * 100)
        
        if log_to_cloud and self.cloud_logger is not None:
            self.cloud_logger.log({f"episodes_{iteration_idx:04}": table})
