from typing import Any, Dict

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from treetune.common import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


@Task.register("answer_extraction", exist_ok=True)
class AnswerExtraction(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_dataset(self) -> DatasetDict:
        ds = super().build_dataset()
        train_ds = ds['train'].to_pandas()
        
        trivia_qa_dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="train")
        trivia_qa_dataset = trivia_qa_dataset.to_pandas()
        
        def process_tqa_row(row):
            # Get answers from the list of answer
            answer_list = row['answer']['aliases']
            top_answer = row['answer']['value']
            row['ground_truth_answer_list'] = answer_list
            row['ground_truth_top_answer'] = top_answer
            return row
        
        train_ds = train_ds.merge(trivia_qa_dataset, on="question_id", how="left")
        train_ds = train_ds.apply(process_tqa_row, axis=1)
        
        train_ds = Dataset.from_pandas(train_ds)
        
        map_fn = self._get_preprocess_fn()
        train_ds = train_ds.map(
            map_fn,
            num_proc=4,
            desc="Preprocessing examples",
        )
        
        ds['train'] = train_ds
        return ds

    def _get_preprocess_fn(self):
        def _preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
            ground_truth_str = example["lc_sft_ground_truth_and_extracted_answers"]
            forecasted_probs_str = example["lc_sft_forecasted_probs"]
            question = example["question"]
            generated_paragraph = example["lc_sft_generated_paragraph"]

            # 1. Check for NaN
            #    Because HF datasets are arrow-based, you might not be able to use pd.isna directly.
            #    But let's assume you can. If not, a simple check for None or empty strings might suffice.
            if pd.isna(ground_truth_str) or pd.isna(forecasted_probs_str):
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,  # Mark row as invalid
                }

            # 2. eval() to lists
            #    (Be sure these are indeed strings containing lists; if they're JSON strings, you might want to do json.loads)
            try:
                ground_truth_and_extracted_answers = eval(ground_truth_str)
                forecasted_probs = eval(forecasted_probs_str)
            except:
                # If parsing fails, return None placeholders
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,
                }

            # 3. Check list lengths and validity
            if not isinstance(ground_truth_and_extracted_answers, list) or not isinstance(forecasted_probs, list):
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,
                }
            if len(ground_truth_and_extracted_answers) != len(forecasted_probs):
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,
                }

            # 4. Check for NaNs inside the lists
            if any(pd.isna(x) for x in ground_truth_and_extracted_answers) or any(pd.isna(x) for x in forecasted_probs):
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,
                }

            # 5. If no extracted answers at all, skip
            if len(ground_truth_and_extracted_answers) == 0:
                return {
                    "question": None,
                    "generated_paragraph": None,
                    "target": None,
                }

            # 6. Construct the final target
            if len(ground_truth_and_extracted_answers) == 1:
                # Single item means no extracted answers
                target = "No Answer"
            else:
                # [1:] because index 0 is the "ground_truth" part
                extracted_answers = ground_truth_and_extracted_answers[1:]
                # Remove duplicates while preserving order
                extracted_answers = list(dict.fromkeys(extracted_answers))
                # Convert to strings, then join
                extracted_answers = list(map(str, extracted_answers))
                target = "; ".join(extracted_answers)

            # 7. Return the new fields
            return {
                "question": question,
                "generated_paragraph": generated_paragraph,
                "target": target,
            }

        return _preprocess_example
