import sglang as sgl
from datasets import Dataset
from tqdm import tqdm

from openrl.common import logging_utils
from openrl.inference_strategies.base_inference_strategy import InferenceStrategy

logger = logging_utils.get_logger(__name__)

@InferenceStrategy.register("basic", exist_ok=True)
class BasicInferenceStrategy(InferenceStrategy):
    def __init__(
        self,
        question_template: str,
        question_field: str = "question",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question_template = question_template
        
        self.question_field = question_field
        if self.log_level is not None:
            logger.setLevel(self.log_level)

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
        question_format_keys = []
        for column in dataset.column_names:
            if f"{{{column}}}" in self.question_template:
                question_format_keys.append(column)
        logger.info(f"Question format keys: {question_format_keys}")
        assert self.question_field in question_format_keys, (
            f"Question field '{self.question_field}' must be in the question template. "
            f"Available format keys: {question_format_keys}"
        )
        
        @sgl.function
        def resp(s, query):
            s += query
            s += sgl.gen("response", stop="\n\n")
        
        responses = []
        for data_instance in tqdm(
            dataset,
            desc="Creating concurrent asyncio tasks for tree construction...",
        ):
            format_kwargs = {key: data_instance[key] for key in question_format_keys}
            initial_prompt = self.question_template.format(**format_kwargs)
            
            response = resp.run(initial_prompt)
            responses.append(response)
        
        dataset = dataset.add_column(
            "response", [response["response"] for response in responses]
        )
        dataset = dataset.add_column(
            "logprobs", 
            [response.get_meta_info("response")["output_token_logprobs"] for response in responses]
        )
        dataset = dataset.add_column(
            "num_tokens",
            [response.get_meta_info("response")["completion_tokens"] for response in responses]
        )
        
        return dataset