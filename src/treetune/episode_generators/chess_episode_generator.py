import copy
import logging
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets

from treetune.common import Lazy
from treetune.common.logging_utils import get_logger
from treetune.episode_generators.base_episode_generator import EpisodeGenerator
from treetune.episodes import Episode
from treetune.inference_servers import InferenceServerHandler
from treetune.inference_strategies.base_inference_strategy import \
    InferenceStrategy
from treetune.tasks.base_task import Task

logger = get_logger(__name__)


class ChessEpisodeGenerator(EpisodeGenerator):
    can_precompute_episodes: bool = False
    support_distributed: bool = True

    def __init__(
        self,
        inference_strategy: Lazy[InferenceStrategy],
        inference_server_handler: Lazy[InferenceServerHandler],
        task: Task,
        initial_model_name_or_path: str,
        total_num_iterations: Optional[int] = None,
        temp_dir_root: Optional[str] = None,
        save_generations_every_n_iteration: Optional[int] = None,
        **kwargs,
    ):
        """
        The base class for episode generators that generate episodes by sampling from the model.
        It supports distributed environments.
        """
        super().__init__(**kwargs)
        self._logger = logger

        self.inference_strategy_lazy = inference_strategy
        self.inference_server_handler = inference_server_handler.construct(**kwargs)
        self.task = task
        self.initial_model_name_or_path = initial_model_name_or_path
        self.total_num_iterations = total_num_iterations
        self.save_generations_every_n_iteration = save_generations_every_n_iteration

        if temp_dir_root is None:
            self.temp_dir_root = self.root_dir / "temp_episodes"
            self._log_on_main(logger, f"Using default temp_dir_root: {self.temp_dir_root}")
        else:
            self.temp_dir_root = Path(temp_dir_root)
        self.temp_dir_root.mkdir(parents=True, exist_ok=True)

    def generate(
        self, iteration: Optional[int] = None, latest_policy_path: Optional[Path] = None
    ):
        """
        Generate episodes by sampling from the model.
        """
        release_memory()

        from deepspeed.runtime.utils import see_memory_usage

        see_memory_usage("Before generating episodes", force=True)

        if iteration is None:
            self._log_on_main(
                logger,
                "Iteration is None. Using 0 as the iteration.", level="warning"
            )
            iteration = 0
        
        self.init(iteration)

        process_index = self.distributed_state.process_index


        # self._log_on_main(
        #     logger,
        #     f"Dataset Size(portion={self.dataset_portion}): {len(dataset)}"
        # )
        # self._log_on_main(
        #     logger,
        #     f"Dataset Examples: "
        #     f"{json.dumps([dataset[i] for i in range(min(2, len(dataset)))], indent=2, sort_keys=True)}"
        # )

        temp_dir = self.temp_dir_root / f"iteration__{iteration:04d}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        results_dir = temp_dir / "infer_results" / f"process_{process_index:02d}"
        results_dir.mkdir(parents=True, exist_ok=True)

        if latest_policy_path is None:
            hf_ckpt_path_or_model = self.initial_model_name_or_path
        else:
            hf_ckpt_path_or_model = str(latest_policy_path)
        
        t0 = time.time()
        infer_results = self._run_inference(
            dataset_shard=dataset,
            model_name_or_path=hf_ckpt_path_or_model,
            results_root_dir=results_dir
        )
        self.distributed_state.wait_for_everyone()
        self._metrics["timing/episode_generation/inference"] = time.time() - t0

        logger.info(f"Process {process_index} finished inference.")

        # Generate episodes from inference results. Each process generates its own episodes.
        t0 = time.time()
        episodes_lst = [
            self._convert_to_dict(e)
            for e in self._generate_episodes(infer_results, iteration)
        ]
        episodes_ds_shard = Dataset.from_list(episodes_lst)
        episodes_ds_shard.save_to_disk(
            temp_dir / f"episodes" / f"shard_{process_index:02d}"
        )
        del episodes_ds_shard
        release_memory()
        self._metrics["timing/episode_generation/inferResult_to_episodes"] = time.time() - t0

        # Concatenate all episodes shards
        self.distributed_state.wait_for_everyone()
        if self.is_main_process():
            shard_paths = list((temp_dir / f"episodes").glob("shard_*"))
            shard_paths.sort(key=lambda x: int(x.name.split("shard_")[-1]))

            merged = concatenate_datasets(
                [Dataset.load_from_disk(str(p)) for p in shard_paths]
            )
            if self.num_episodes_per_iteration is None:
                pass
            elif len(merged) > self.num_episodes_per_iteration:
                merged = merged.shuffle(seed=self.seed + iteration)
                merged = merged.select(range(self.num_episodes_per_iteration))
            elif len(merged) < self.num_episodes_per_iteration:
                if self.fill_missing_episodes:
                    # Fill the missing episodes by repeating the existing ones
                    logger.warning(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration}). "
                        f"Repeating the existing episodes."
                    )
                    num_repeats = self.num_episodes_per_iteration // len(merged) + 1
                    merged = concatenate_datasets([merged] * num_repeats)
                    merged = merged.shuffle(seed=self.seed + iteration)
                    merged = merged.select(range(self.num_episodes_per_iteration))
                    logs = {f"episodes_metric/fill_missing_episodes": num_repeats}
                    
                    self._metrics.update(logs)
                else:
                    raise ValueError(
                        f"Number of episodes generated ({len(merged)}) is less than "
                        f"num_episodes_per_iteration ({self.num_episodes_per_iteration})"
                    )

            merged.save_to_disk(temp_dir / "episodes" / "merged")
            del merged
            release_memory()

        self.distributed_state.wait_for_everyone()
        episodes = Dataset.load_from_disk(str(temp_dir / "episodes" / "merged"))

        see_memory_usage("After generating episodes", force=True)

        self._save_generations_to_cloud(temp_dir, iteration)
        self._clean_up_temp_dir(temp_dir)
        self.distributed_state.wait_for_everyone()
        
        metrics = self.gather_metrics()
        if len(metrics) > 0:
            metrics["train/global_iteration"] = iteration
            self._cloud_log(metrics)
        
        self.distributed_state.wait_for_everyone()
        return episodes

    def _run_inference(
        self,
        dataset_shard: Dataset,
        model_name_or_path: str,
        results_root_dir: Path,
    ):
        """
        Potentially start an inference server and run inference to generate results needed for episode generation.

        Args:
            dataset_shard (Dataset):
                The shard of the prompt dataset to run inference on.
            model_name_or_path (str):
                The model to use for inference.
            results_root_dir (Path):
                The directory to save the results to (this is unique for each process).
        """
        infer_result_path = results_root_dir / "results_ds"
        guidance_llm_kwargs = self.inference_server_handler.get_or_create_server_with_model(
            model_name_or_path, results_root_dir)

        # Initialize the inference strategy with the inference server URL
        inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
        inference_strategy_lazy._params["guidance_llm"].update(guidance_llm_kwargs)
        inference_strategy = inference_strategy_lazy.construct(
            result_dir=results_root_dir,
            seed=self.get_process_seed(),
            cloud_logger=None,
            log_level=(
                logging.WARNING
                if not self.distributed_state.is_local_main_process
                else None
            ),
        )

        results = inference_strategy.generate(dataset_shard)
        results.save_to_disk(str(infer_result_path))
        
        logger.info(f"Rank {self.distributed_state.process_index} finished inference.")
        del results
        
        self.inference_server_handler.kill_server()
        self.inference_server_handler.compute_server_stats(results_root_dir)
        
        results = Dataset.load_from_disk(str(results_root_dir / "results_ds"))
        return results

    def _generate_episodes(
        self, inference_results: Dataset, iteration: int
    ) -> List[Union[Dict[str, Any], Episode]]:
        raise NotImplementedError

    def _save_generations_to_cloud(self, generations_dir: Path, iteration: int):
        if self.cloud_logger is None or not self.is_main_process():
            return

        if self.save_generations_every_n_iteration is None:
            # Saving generations is disabled
            return

        if iteration != 0 and iteration % self.save_generations_every_n_iteration != 0:
            # We only save generations every n iterations and the first iteration
            return

        temp_dir = Path(tempfile.mkdtemp())

        generations = temp_dir / f"iteration__{iteration:04d}.zip"
        shutil.make_archive(
            str(generations.with_suffix("")),
            format="zip",
            root_dir=generations_dir,
        )
        self.cloud_logger.save(str(generations.absolute()), policy="now")

    def _filter_init_dataset(self, dataset: Dataset) -> Dataset:
        if self.max_question_length is None:
            return dataset

        tokenizer = self.tokenizer
        question_template = self.question_template
        max_question_length = self.max_question_length
        question_format_keys = []
        for column in dataset.column_names:
            if f"{{{column}}}" in self.question_template:
                question_format_keys.append(column)

        if len(question_format_keys) == 0:
            raise ValueError(
                "No columns found in the question template. "
                "Please add the column names in the question template."
            )

        def filter_out_long_questions(example):
            format_kwargs = {key: example[key] for key in question_format_keys}
            prompt = question_template.format(**format_kwargs)
            tokens = tokenizer(prompt).input_ids
            return len(tokens) <= max_question_length

        length_before = len(dataset)
        dataset = dataset.filter(
            filter_out_long_questions, num_proc=4, desc="Filtering long questions"
        )
        self._log_on_main(
            logger,
            f"Filtered out {length_before - len(dataset)} long questions from {length_before} questions."
        )
        return dataset

    def _clean_up_temp_dir(self, temp_dir: Path) -> None:
        if not self.is_main_process():
            return

        try:
            # Remove all input_dataset__* directories
            for p in temp_dir.glob("input_dataset__*"):
                shutil.rmtree(p, ignore_errors=True)

            # Remove all episodes shards
            for p in (temp_dir / "episodes").glob("shard_*"):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.error(f"Error while cleaning up temp dir: {e}")

    def _convert_to_dict(self, episode_obj) -> Dict[str, Any]:
        if isinstance(episode_obj, dict):
            return episode_obj

        return asdict(episode_obj)
