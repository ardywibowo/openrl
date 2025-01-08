from datasets import DatasetDict

from treetune import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


@Task.register("reward_bench", exist_ok=True)
class RewardBench(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_dataset(self) -> DatasetDict:
        ds = super().build_dataset()
        
        train_dataset = ds['filtered']
        
        map_fn = self._get_preprocess_fn()
        train_dataset = train_dataset.map(
            map_fn,
            num_proc=4,
            batched=True, 
            remove_columns=train_dataset.column_names,
            desc="Preprocessing examples"
        )
        
        ds['train'] = train_dataset
        
        return ds

    def _get_preprocess_fn(self):
        def _preprocess_example(batch):
            return {
                "query": batch["prompt"],
                "chosen": batch["chosen"],
                "rejected": batch["rejected"],
            }
        
        return _preprocess_example
