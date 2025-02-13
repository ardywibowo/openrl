import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import sglang as sgl
from datasets import Dataset
from tqdm import tqdm

from openrl.common import JsonDict, logging_utils
from openrl.inference_strategies.base_inference_strategy import \
    InferenceStrategy

logger = logging_utils.get_logger(__name__)

@InferenceStrategy.register("reasoning", exist_ok=True)
class ReasoningInferenceStrategy(InferenceStrategy):
    def __init__(
        self,
        system_prompt: str,
        question_template: str,
        num_samples: int,
        sampling_parameters: JsonDict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.question_template = question_template
        self.num_samples = num_samples
        self.sampling_parameters = sampling_parameters
        
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
        
        assert len(question_format_keys) > 0, "No question format keys found."
        
        sampling_parameters = self.sampling_parameters
        @sgl.function
        def resp(s, query):
            s += sgl.system(self.system_prompt)
            s += sgl.user(query)
            s += sgl.assistant(
                sgl.gen("response", **sampling_parameters)
            )

        def run_single_response(initial_prompt):
            return resp.run(initial_prompt)
        
        # Prepare a structure to collect responses grouped by instance.
        responses_by_instance = {idx: [None] * self.num_samples for idx in range(len(dataset))}
        
        # We'll create one task per sample across all instances.
        tasks = []  # List of tuples: (instance_index, sample_index, future)
        num_workers = 8  # Adjust as needed.

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit a separate task for each sample generation.
            for idx, data_instance in enumerate(dataset):
                # Build the prompt for this instance.
                format_kwargs = {key: data_instance[key] for key in question_format_keys}
                initial_prompt = self.question_template.format(**format_kwargs)
                for sample in range(self.num_samples):
                    future = executor.submit(run_single_response, initial_prompt)
                    tasks.append((idx, sample, future))
            
            # Create a mapping from each future to its (instance index, sample index)
            future_to_task = {future: (idx, sample) for (idx, sample, future) in tasks}
            
            # As each task completes, assign the result to its proper place.
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task),
                            desc="Generating responses concurrently..."):
                idx, sample = future_to_task[future]
                responses_by_instance[idx][sample] = future.result()

        # Now, group the responses back into a list that mirrors the dataset order.
        responses_grouped = [responses_by_instance[i] for i in range(len(dataset))]
        
        def extract_answer(output_text):
            answer_match = re.search(
                r"<answer>(.*?)</answer>",
                output_text,
                flags=re.DOTALL,
            )
            answer = answer_match.group(1) if answer_match else None
            return answer
        
        # Add the new columns to the dataset.
        dataset = dataset.add_column(
            "response_group", 
            [[response["response"] for response in responses] for responses in responses_grouped]
        )
        dataset = dataset.add_column(
            "full_text_group", 
            [[response.text() for response in responses] for responses in responses_grouped]
        )
        dataset = dataset.add_column(
            "logprobs_group", 
            [[response.get_meta_info("response")["output_token_logprobs"] for response in responses]
            for responses in responses_grouped]
        )
        dataset = dataset.add_column(
            "num_tokens_group",
            [[response.get_meta_info("response")["completion_tokens"] for response in responses]
            for responses in responses_grouped]
        )
        dataset = dataset.add_column(
            "answer_group",
            [[extract_answer(response["response"]) for response in responses] for responses in responses_grouped]
        )
        dataset = dataset.add_column(
            "finish_reason_group",
            [
                [
                    response.get_meta_info("chain_of_thought")["finish_reason"]['type'] 
                    for response in responses
                ]
                for responses in responses_grouped
            ]
        )
        
        return dataset