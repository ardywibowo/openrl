from concurrent.futures import ThreadPoolExecutor

import sglang as sgl
from datasets import Dataset
from tqdm import tqdm

from openrl.common import logging_utils, JsonDict
from openrl.inference_strategies.base_inference_strategy import \
    InferenceStrategy

logger = logging_utils.get_logger(__name__)

@InferenceStrategy.register("reasoning", exist_ok=True)
class ReasoningInferenceStrategy(InferenceStrategy):
    def __init__(
        self,
        question_template: str,
        sampling_parameters: JsonDict,
        question_field: str = "question",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.question_template = question_template
        self.sampling_parameters = sampling_parameters
        
        self.question_field = question_field
        if self.log_level is not None:
            logger.setLevel(self.log_level)
    
    def generate(self, dataset: Dataset) -> Dataset:
        """
        Generate a new dataset based on the given dataset.
        New columns are prefixed with "_openrl__{column_name}".
        The following columns should be always added:
        - _openrl__candidate_answers: List[str] - The candidate answers
        """
        # Determine which columns are used in the question template.
        question_format_keys = [
            column for column in dataset.column_names
            if f"{{{column}}}" in self.question_template
        ]
        logger.info(f"Question format keys: {question_format_keys}")
        
        assert self.question_field in question_format_keys, (
            f"Question field '{self.question_field}' must be in the question template. "
            f"Available format keys: {question_format_keys}"
        )
        
        sampling_parameters = self.sampling_parameters
        @sgl.function
        def resp(s, query):
            s += query
            s += sgl.gen("response", **sampling_parameters)
        
        # Helper function to process one data instance.
        def process_instance(data_instance):
            format_kwargs = {key: data_instance[key] for key in question_format_keys}
            initial_prompt = self.question_template.format(**format_kwargs)
            
            # Run the SGL function and return its result.
            return resp.run(initial_prompt)
        
        # Set the number of worker threads (adjust as needed).
        # Process the dataset concurrently.
        NUM_WORKERS = 32
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # executor.map preserves the order of the dataset.
            responses = list(tqdm(
                executor.map(process_instance, dataset),
                total=len(dataset),
                desc="Processing dataset concurrently..."
            ))
        
        # Add new columns based on the responses.
        dataset = dataset.add_column(
            "response", [response["response"] for response in responses]
        )
        dataset = dataset.add_column(
            "logprobs", [
                response.get_meta_info("response")["output_token_logprobs"]
                for response in responses
            ]
        )
        dataset = dataset.add_column(
            "num_tokens", [
                response.get_meta_info("response")["completion_tokens"]
                for response in responses
            ]
        )
        
        return dataset