from pathlib import Path

import sglang as sgl
from datasets import Dataset

from openrl.common import Registrable


class InferenceStrategy(Registrable):
    def __init__(self, server_url: str, result_dir: Path, cloud_logger=None, log_level=None):
        """
        result_dir: to store the middle results to enable resuming
        """
        self.server_url = server_url
        self.result_dir = result_dir
        self.cloud_logger = cloud_logger
        self.log_level = log_level
        
        sgl.set_default_backend(sgl.RuntimeEndpoint(server_url))

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
