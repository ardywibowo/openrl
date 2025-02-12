from pathlib import Path

import sglang as sgl
from datasets import Dataset

from openrl.common import Component


class InferenceStrategy(Component):
    def __init__(self, log_level=None, **kwargs):
        """
        root_dir: to store the middle results to enable resuming
        """
        super().__init__(**kwargs)
        
        self.log_level = log_level

    def generate(self, dataset: Dataset) -> Dataset:
        """
        Generate a new dataset based on the given dataset
        Params:
            dataset: The dataset to generate from

        Returns:
            A new dataset, which is the input dataset with new columns.
            New columns are prefixed with "_openrl__{column_name}".
            The following columns should be always added:
            - _openrl__candidate_answers: List[str] - The candidate answers
        """
        raise NotImplementedError()
