import json
from typing import Optional

from accelerate.utils import release_memory
from datasets import Dataset, concatenate_datasets
from transformers import Pipeline

from treetune.logging_utils import get_logger
from treetune.pipelines import Pipeline

logger = get_logger(__name__)

@Pipeline.register("dataset_pipeline")
class DatasetPipeline(Pipeline):
    def __init__(
        self,
        shuffle_on_each_iteration: bool = True,
        shuffle_before_portion: bool = True,
        portion: Optional[float] = None,
        num_samples_per_iteration: Optional[int] = None,
        sample_with_replacement: bool = True,
        initial_size: Optional[int] = None,
        total_num_iterations: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.dataset_portion = portion
        self.num_samples_per_iteration = num_samples_per_iteration
        self.shuffle_before_portion = shuffle_before_portion
        self.shuffle_on_each_iteration = shuffle_on_each_iteration
        self.sample_with_replacement = sample_with_replacement
        self.dataset_initial_size = initial_size
        self.total_num_iterations = total_num_iterations
        
        if (
            self.dataset_portion is not None
            and self.num_samples_per_iteration is not None
        ):
            raise ValueError(
                "Only one of `dataset_portion` and `dataset_num_samples_per_iteration` can be set."
            )
        if (
            self.dataset_portion is None
            and self.num_samples_per_iteration is None
        ):
            self.dataset_portion = 1.0
    
    def __call__(self, input_dataset: Dataset, cache: bool = False) -> Dataset:
        dataset = input_dataset
        if self.num_samples_per_iteration is not None:
            num_samples = self.num_samples_per_iteration
        else:
            num_samples = int(self.initial_ds_after_filter_size * self.dataset_portion)
            self._log_on_main(
                logger,
                f"Using {num_samples} samples for each iteration based on the dataset portion.")
        
        assert num_samples <= len(dataset)
        if not self.sample_with_replacement:
            # Split the dataset into portions and select one portion based on the iteration
            samples_per_iteration = (
                self.num_samples_per_iteration
                if self.num_samples_per_iteration is not None
                else int(self.initial_ds_after_filter_size * self.dataset_portion)
            )
            start_idx = samples_per_iteration * self._iteration
            end_idx = samples_per_iteration * self._iteration + num_samples
            dataset = dataset.select(range(start_idx, end_idx))
        else:
            # Shuffle the dataset so that the same dataset is not used in every iteration
            do_shuffle = self.shuffle_on_each_iteration or self.shuffle_before_portion
            if do_shuffle:
                dataset = dataset.shuffle(seed=self.seed + self._iteration)

            dataset = dataset.select(range(num_samples))

        self._log_on_main(
            logger,
            f"Dataset Size(portion={self.dataset_portion}): {len(dataset)}"
        )
        self._log_on_main(
            logger,
            f"Dataset Examples: "
            f"{json.dumps([dataset[i] for i in range(min(2, len(dataset)))], indent=2, sort_keys=True)}"
        )
        
        if cache:
            process_index = self.distributed_state.process_index
            temp_dir = self.root_dir / f"iteration__{self._iteration:04d}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save to disk so that it's memory efficient. Note that this is done on all processes.
            # to avoid any issues with distributed environment and funkiness of HF Datasets.
            inp_ds_path = temp_dir / f"input_dataset__{process_index}"
            dataset.save_to_disk(inp_ds_path)
            del dataset

            # The same dataset is loaded on all processes
            dataset = Dataset.load_from_disk(str(inp_ds_path))

            # Shard the dataset based on the number of processes
            dataset = dataset.shard(
                num_shards=self.distributed_state.num_processes,
                index=process_index,
                contiguous=True,
            )
        
        return dataset
    
    def process_init_dataset(self, input_dataset: Dataset) -> Dataset:
        self.initial_ds_after_filter_size = len(input_dataset)
        
        output_dataset = input_dataset
        if self.dataset_initial_size is not None:
            output_dataset = output_dataset.shuffle(self.seed).select(
                range(self.dataset_initial_size)
            )
            self._log_on_main(
                logger,
                f"Dataset Size after initial selection: {len(output_dataset)}"
            )

        if not self.sample_with_replacement:
            # Create the dataset once on all processes
            self._log_on_main(
                logger,
                f"Creating and caching dataset for once on all processes."
            )
            if self.total_num_iterations is None:
                if self.shuffle_on_each_iteration:
                    output_dataset = output_dataset.shuffle(seed=self.seed)
            else:
                # Compute the number of dataset repeats needed to cover all iterations
                dataset_size = len(output_dataset)
                samples_per_iteration = (
                    self.num_samples_per_iteration
                    if self.num_samples_per_iteration is not None
                    else int(dataset_size * self.dataset_portion)
                )

                num_repeats = (
                    self.total_num_iterations * samples_per_iteration // dataset_size
                )
                num_repeats += 1
                if num_repeats > 1:
                    self._log_on_main(
                        logger,
                        f"Repeating the dataset {num_repeats} times to cover all iterations."
                    )
                    if self.distributed_state.is_main_process:
                        new_ds = concatenate_datasets(
                            [
                                output_dataset.shuffle(seed=self.seed + i)
                                for i in range(num_repeats)
                            ]
                        )
                        new_ds.save_to_disk(self.root_dir / "cached_dataset")
                        del new_ds
                        release_memory()
                    output_dataset = Dataset.load_from_disk(
                        str(self.root_dir / "cached_dataset")
                    )
                else:
                    if self.shuffle_on_each_iteration:
                        output_dataset = output_dataset.shuffle(seed=self.seed)
        
        return output_dataset
