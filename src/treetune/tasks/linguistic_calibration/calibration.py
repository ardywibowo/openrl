import copy
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict

from treetune import logging_utils
from treetune.tasks import Task
from treetune.tokenization_utils import Tokenizer

logger = logging_utils.get_logger(__name__)


@Task.register("linguistic_calibration", exist_ok=True)
class LinguisticCalibration(Task):
    def __init__(
        self,
        prepend_in_context_few_shot: bool,
        few_shot_dataset_path: Optional[str] = None,
        use_gold_steps_for_few_shot: bool = False,
        num_few_shot_examples: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        ensure_fit_in_context_size: bool = False,
        max_few_shot_dataset_size: Optional[int] = None,
        context_size: Optional[int] = None,
        max_generation_tokens: Optional[int] = None,
        answer_prefix: Optional[str] = "\n\n# Answer\n",
        answer_extractor: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prepend_in_context_few_shot = prepend_in_context_few_shot
        self.num_few_shot_examples = num_few_shot_examples
        self.ensure_fit_in_context_size = ensure_fit_in_context_size
        self.tokenizer = tokenizer
        self.use_gold_steps_for_few_shot = use_gold_steps_for_few_shot
        self.few_shot_dataset = None
        self.context_size = context_size
        self.max_generation_tokens = max_generation_tokens
        self.max_few_shot_dataset_size = max_few_shot_dataset_size
        self.answer_prefix = answer_prefix

        # If few-shot examples are to be included, load the dataset from the provided path.
        if self.prepend_in_context_few_shot:

            if self.ensure_fit_in_context_size:
                assert self.context_size is not None, "Context size must be provided."

            if self.max_few_shot_dataset_size is not None:
                self.few_shot_dataset = self.few_shot_dataset.shuffle(seed=42).select(
                    range(self.max_few_shot_dataset_size)
                )

            if (
                self.few_shot_dataset is not None
                and "gold_solution_steps" in self.few_shot_dataset.column_names
            ):

                def append_gold_solution_steps_str(
                    example: Dict[str, Any]
                ) -> Dict[str, Any]:
                    sol_steps = example["gold_solution_steps"]
                    sol_steps_str = "\n".join(sol_steps)
                    return {"gold_solution_steps_str": sol_steps_str}

                self.few_shot_dataset = self.few_shot_dataset.map(
                    append_gold_solution_steps_str,
                    num_proc=4,
                    desc="Appending gold solution steps",
                )

            if self.few_shot_dataset is not None and any(
                [
                    self.max_few_shot_problem_length,
                    self.max_few_shot_solution_length,
                    self.max_few_shot_num_steps,
                ]
            ):
                assert tokenizer is not None, "Tokenizer must be provided."
                self._preprocess_few_shot_dataset()

    def _preprocess_few_shot_dataset(self):
        tokenizer = self.tokenizer
        use_gold_steps_for_few_shot = self.use_gold_steps_for_few_shot

        def keep_shorter_than_max_length(example: Dict[str, Any]) -> bool:
            problem_length = len(tokenizer.encode(example["problem"]))
            if "gold_solution_steps" in example and use_gold_steps_for_few_shot:
                solution = example["gold_solution_steps"]
                num_steps = len(solution)
            elif isinstance(example["solution"], list):
                solution = "\n".join(example["solution"])
                num_steps = len(example["solution"])
            else:
                solution = example["solution"]
                num_steps = None

            solution_length = len(tokenizer.encode(solution))

            is_short_enough = True
            if self.max_few_shot_problem_length is not None:
                is_short_enough &= problem_length <= self.max_few_shot_problem_length

            if self.max_few_shot_solution_length is not None:
                is_short_enough &= solution_length <= self.max_few_shot_solution_length

            if self.max_few_shot_num_steps is not None and num_steps is not None:
                is_short_enough &= num_steps <= self.max_few_shot_num_steps

            return is_short_enough

        ds_len_before = len(self.few_shot_dataset)
        self.few_shot_dataset = self.few_shot_dataset.filter(
            keep_shorter_than_max_length,
            num_proc=4,
            desc="Filtering few-shot examples",
        )
        logger.info(
            f"Filtered few-shot examples from {ds_len_before} to {len(self.few_shot_dataset)} examples"
        )

    def build_dataset(self) -> DatasetDict:
        datasets = super().build_dataset()

        def append_gold_solution_steps_str(example: Dict[str, Any]) -> Dict[str, Any]:
            sol_steps = example["gold_solution_steps"]
            sol_steps_str = "\n".join(sol_steps)
            return {"gold_solution_steps_str": sol_steps_str}

        if 'train' in datasets and "gold_solution_steps" in datasets["train"].column_names:
            datasets = datasets.map(
                append_gold_solution_steps_str,
                num_proc=4,
                desc="Appending gold solution steps",
            )
        
        map_fn = self._get_preprocess_fn()
        datasets = datasets.map(
            map_fn,
            num_proc=4,
            desc="Preprocessing examples",
        )
        return datasets

    def _get_preprocess_fn(self):
        tokenizer = self.tokenizer
        few_shot_dataset = self.few_shot_dataset
        num_few_shot_examples = self.num_few_shot_examples
        prepend_in_context_few_shot = self.prepend_in_context_few_shot
        use_gold_steps_for_few_shot = self.use_gold_steps_for_few_shot
        ensure_fit_in_context_size = self.ensure_fit_in_context_size
        context_size = self.context_size
        max_generation_tokens = self.max_generation_tokens
        answer_prefix = self.answer_prefix

        def get_solution_text(example, answer=None):
            solution = example["solution"]
            if "gold_solution_steps_str" in example and use_gold_steps_for_few_shot:
                # MATH solutions split into steps using a heuristic.
                solution = example["gold_solution_steps_str"]
            elif isinstance(solution, list):
                # OpenAI PRM format
                return "\n".join(solution)  # Already contains final answer.

            if answer is not None and answer_prefix is not None:
                # Append the answer to the solution text.
                return solution + answer_prefix + answer
            return solution

        def generate_fewshot_context(rng, problem_text, delimiter):
            random_indices = rng.choice(
                len(few_shot_dataset),
                num_few_shot_examples + 1,  # Extra to avoid self-inclusion.
                replace=False,
            )

            # Filter out the current problem from the few-shot examples.
            few_shot_examples = [
                few_shot_dataset[i]
                for i in random_indices.tolist()
                if few_shot_dataset[i]["problem"] != problem_text
            ][:num_few_shot_examples]

            # Format few-shot examples as strings.
            few_shot_examples_strs = []
            for ex in few_shot_examples:
                fs_problem = ex["problem"]
                fs_solution_str = get_solution_text(ex, answer=ex.get("answer"))
                fs_str = f"Problem:\n{fs_problem}\n\nSolution:\n{fs_solution_str}"
                few_shot_examples_strs.append(fs_str)

            few_shot_context = delimiter.join(few_shot_examples_strs)
            return few_shot_context

        def _preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
            problem_text = example["problem"]

            few_shot_delimiter = "\n\n\n"

            max_retries = 10 if ensure_fit_in_context_size else 1

            output = {}
            if prepend_in_context_few_shot:
                # Generate a seed based on the example's index for reproducibility.
                init_seed = example["_treetune__idx"]

                num_tries = 0
                while num_tries < max_retries:
                    rng = np.random.RandomState(init_seed + num_tries)
                    few_shot_ctx = generate_fewshot_context(
                        rng, problem_text, few_shot_delimiter
                    )
                    query = (
                        few_shot_ctx
                        + few_shot_delimiter
                        + f"Problem:\n{problem_text}\n\nSolution:\n"
                    )
                    prompt_tokens = tokenizer.encode(query)
                    if (len(prompt_tokens) + max_generation_tokens) <= context_size:
                        break
                    logger.warning(
                        f"Could not fit the prompt in the context size. Retrying..."
                    )
                    num_tries += 1

                if ensure_fit_in_context_size and num_tries == max_retries:
                    logger.warning(
                        f"Could not fit the few-shot context in the context size for problem: {problem_text}"
                    )
                    # Just discard the first few tokens
                    extra_tokens_length = (
                        len(prompt_tokens)
                        + self.max_generation_tokens
                        - self.context_size
                    )
                    prompt_tokens = prompt_tokens[extra_tokens_length:]
                    query = tokenizer.decode(prompt_tokens)

                output["_few_shot_context"] = few_shot_ctx
            else:
                query = problem_text

            output["query"] = query

            return output

        return _preprocess_example

    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        """
        We assume that the solution is in the format:
        Solution:
        <reasoning_step_1>
        <reasoning_step_2>
        ...
        <reasoning_step_n>
        # Answer

        <answer>
        """
        splits = text.split("# Answer\n")

        # Be conservative and return None if the format is not as expected.
        if len(splits) != 2:
            return None

        return splits[1].strip()

    def grade_answer(
        self,
        *,
        given_answer: str = None,
        ground_truth: str = None,
        item: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> bool:
        def grade_fn():
            return grade_answer(given_answer=given_answer, ground_truth=ground_truth)

        if timeout is None:
            return grade_fn()

        from call_function_with_timeout import SetTimeout

        func = SetTimeout(grade_fn, timeout=timeout)
        is_done, is_timeout, error_message, results = func()
        if is_timeout:
            logger.warning(
                f"Grading function timed out for problem:\n{item['problem']}\n and solution:\n{given_answer}"
            )
            return False

        return results

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        assert len(predictions) == len(references)
        assert len(predictions) > 0, "No predictions provided."

        once_hit_acc = []
        correct_frac = []
        majority_vote_acc = []
        unique_answer_count = []
        none_answer_extracted = []
        for solution_candidates, ref in zip(predictions, references):
            gold_answer = ref["answer"]
            problem = ref["problem"]

            assert len(solution_candidates) > 0
            answer_candidates = [
                self.extract_predicted_answer_from_text(sol, problem=problem)
                for sol in solution_candidates
            ]
            none_answer_extracted.append(
                sum([1 for ans in answer_candidates if ans is None])
                / len(answer_candidates)
            )

            grading_results = [
                self.grade_answer(given_answer=ans, ground_truth=gold_answer, item=ref)
                for ans in answer_candidates
            ]
            once_hit_acc.append(float(any(grading_results)))
            correct_frac.append(sum(grading_results) / len(grading_results))

            answer_candidates = [
                tuple(ans) if isinstance(ans, list) else ans
                for ans in answer_candidates
            ]

            majority_answer, _ = Counter(answer_candidates).most_common(n=1)[0]
            assert len(answer_candidates) == len(grading_results)
            majority_answer_index = answer_candidates.index(majority_answer)
            majority_answer_is_correct = grading_results[majority_answer_index]
            majority_vote_acc.append(majority_answer_is_correct)

            unique_answer_count.append(len(set(answer_candidates)))

        once_hit = sum(once_hit_acc) / len(once_hit_acc)
        correct_frac = sum(correct_frac) / len(correct_frac)

        return {
            "once_hit": once_hit,
            "exact_match": once_hit,  # for backwards compatibility
            "correct_frac": correct_frac,
            "exact_match_frac": correct_frac,  # for backwards compatibility
            "majority_vote_acc": sum(majority_vote_acc) / len(majority_vote_acc),
            "unique_answer_count": sum(unique_answer_count) / len(unique_answer_count),
            "none_answer_extracted_frac_per_problem": (
                sum(none_answer_extracted) / len(none_answer_extracted)
            ),
        }
