from collections import defaultdict
from typing import Any, Dict

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from treetune.common import logging_utils
from treetune.tasks import Task

logger = logging_utils.get_logger(__name__)


@Task.register("probability_forecasting", exist_ok=True)
class ProbabilityForecasting(Task):
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
            batched=True, 
            remove_columns=train_ds.column_names,
            desc="Preprocessing examples"
        )
        
        ds['train'] = train_ds
        return ds

    def _get_preprocess_fn(self):
        def _preprocess_example(batch):
            """
            This function will:
            - eval() the 'lc_sft_ground_truth_and_extracted_answers' and 'lc_sft_forecasted_probs'
            - skip invalid rows (NaN, mismatched lengths, out-of-bounds rewards, etc.)
            - 'unroll' each row into multiple rows, appending to arrays
            Returns a dictionary of lists that the HF Dataset will combine into columns.
            """

            # Prepare output arrays
            questions = []
            ground_truth_top_answers = []
            target_probs = []
            generated_paragraphs = []

            # Because 'batched=True', each field in 'batch' is a list of values
            # We'll iterate over them in parallel
            for (question,
                ground_truth_str,
                forecasted_probs_str,
                generated_paragraph) in zip(
                    batch["question"],
                    batch["lc_sft_ground_truth_and_extracted_answers"],
                    batch["lc_sft_forecasted_probs"],
                    batch["lc_sft_generated_paragraph"]
                ):

                # --- 1) Check for NaN/None ---
                # Arrow-based columns might be None or string "nan".
                # If you're using Pandas inside HF, pd.isna is sometimes okay:
                if pd.isna(ground_truth_str) or pd.isna(forecasted_probs_str):
                    # Skip this entire example
                    continue

                # --- 2) Parse with eval (or json.loads if it's valid JSON) ---
                try:
                    ground_truth_list = eval(ground_truth_str)
                    forecasted_probs_list = eval(forecasted_probs_str)
                except Exception:
                    # If parsing fails, skip
                    continue

                # --- 3) Check types and lengths ---
                if not isinstance(ground_truth_list, list) or not isinstance(forecasted_probs_list, list):
                    continue
                if len(ground_truth_list) == 0 or len(forecasted_probs_list) == 0:
                    continue
                if len(ground_truth_list) != len(forecasted_probs_list):
                    continue

                # --- 4) Loop over pairs (answer_choice, reward) ---
                for ans, target_prob in zip(ground_truth_list, forecasted_probs_list):
                    if pd.isna(ans) or pd.isna(target_prob):
                        continue
                    
                    # Make sure reward is in [0, 1]
                    if not (0 <= target_prob <= 1):
                        continue

                    # Append to output arrays
                    questions.append(question)
                    ground_truth_top_answers.append(ans)
                    target_probs.append(target_prob)
                    generated_paragraphs.append(generated_paragraph)

            # Return dict of lists
            return {
                "question": questions,
                "ground_truth_top_answer": ground_truth_top_answers,
                "target_prob": target_probs,
                "generated_paragraph": generated_paragraphs,
            }

        return _preprocess_example
