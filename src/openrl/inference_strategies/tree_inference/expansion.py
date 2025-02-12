import os
import random
import re
import uuid
from typing import Callable, List, Optional

import asyncio
import evaluate
import numpy as np
import sglang as sgl

from openrl.common import JsonDict, Registrable, Tokenizer, logging_utils
from openrl.common.py_utils import format_string
from openrl.inference_strategies.tree_inference import Node
from openrl.inference_strategies.tree_inference.branch_factor_strategy import \
    BranchFactorStrategy

logger = logging_utils.get_logger(__name__)

class NodeExpander(Registrable):
    def __init__(
        self, branch_factor_strategy: BranchFactorStrategy, seed: Optional[int] = None
    ):
        self.branch_factor_strategy = branch_factor_strategy
        self.seed = seed

        self._sem_program = None

    def set_program_semaphore(self, sem_program):
        self._sem_program = sem_program

    def set_seed(self, seed):
        self.seed = seed

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        raise NotImplementedError()


@NodeExpander.register("iid")
class IIDExpander(NodeExpander):
    def __init__(
        self, node_text_template: str, sampling_parameters: JsonDict, **kwargs
    ):
        super().__init__(**kwargs)

        if "return_logprob" not in sampling_parameters:
            sampling_parameters["return_logprob"] = False
        else:
            assert sampling_parameters["return_logprob"] in [False, True], "return_logprob must be False or True"

        self.sampling_parameters = sampling_parameters

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

    async def _sample_node(self, prefix: str, depth: int) -> Node:
        
        sampling_parameters = self.sampling_parameters
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.gen("chain_of_thought", **sampling_parameters)
        
        result = resp.run(prefix=prefix)
        
        # assert self._sem_program is not None
        # async with self._sem_program:
        #     result = resp.run(prefix=prefix)
        
        chain_of_thought = result["chain_of_thought"]
        if 'matched' in result.get_meta_info("chain_of_thought")['finish_reason']:
            stop_text = result.get_meta_info("chain_of_thought")['finish_reason']['matched']
        else:
            stop_text = None
        
        node_text = self.node_text_template.format(chain_of_thought=chain_of_thought)
        
        node = {
            "text": node_text,
            "depth": depth,
            "full_text": result.text,
            "stop_text": stop_text,
        }
        
        if (
            "output_token_logprobs" in result.get_meta_info("chain_of_thought")
            and self.sampling_parameters["return_logprob"]
        ):
            logprobs: List[float] = [r[0] for r in result.get_meta_info("answer")["output_token_logprobs"]]
            assert isinstance(logprobs, list)
            node_num_tokens = len(logprobs)
            node_logprobs = sum(logprobs)
            
            node["sum_logprobs"] = node_logprobs
            node["num_tokens"] = node_num_tokens
        
        return node

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        tasks = []
        branch_factor = self.branch_factor_strategy(current_node)
        for i in range(branch_factor):
            task = asyncio.create_task(self._sample_node(prefix, depth + 1))
            tasks.append(task)

        nodes = []
        for task in tasks:
            node = await task
            nodes.append(node)

        return nodes

@NodeExpander.register("efficient_iid")
class EfficientIIDExpander(NodeExpander):
    def __init__(
        self,
        node_text_template: str,
        sampling_parameters: JsonDict,
        num_expansion_rounds: int = 1,
        model_context_size: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if "return_logprob" in sampling_parameters:
            sampling_parameters["return_logprob"] = False
            logger.warning(
                "return_logprob will be set to False as it's not supported. Please remove return_logprob from sampling_parameters."
            )
        else:
            sampling_parameters["return_logprob"] = 0

        if "num_samples" in sampling_parameters:
            sampling_parameters.pop("num_samples")
            logger.warning(
                "num_samples will be set by the branch_factor_strategy. "
                "Please remove num_samples from sampling_parameters."
            )

        if "stop_regex" in sampling_parameters:
            sampling_parameters.pop("stop_regex")
            logger.warning("stop_regex is not supported. Please use `stop`")

        self.num_expansion_rounds = num_expansion_rounds
        self.sampling_parameters = sampling_parameters

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"

        self.model_context_size = model_context_size
        self.tokenizer = tokenizer
        if self.model_context_size is not None:
            assert self.tokenizer is not None, "tokenizer must be provided"

    async def _sample_node(
        self, prefix: str, depth: int, branch_factor: int, seed: Optional[int] = None
    ) -> List[Node]:
        sampling_parameters = self.sampling_parameters.copy()

        need_to_compute_max_tokens = (
            "max_tokens" in self.sampling_parameters
            and self.model_context_size is not None
        )
        if need_to_compute_max_tokens:
            new_max_tokens = self._compute_max_tokens(
                prefix, sampling_parameters.get("max_tokens")
            )
            if new_max_tokens != sampling_parameters.get("max_tokens"):
                logger.warning(
                    f"Overriding max_tokens: {sampling_parameters.get('max_tokens')} -> {new_max_tokens}"
                )
            assert new_max_tokens > 0, f"new_max_tokens: {new_max_tokens}"
            sampling_parameters["max_tokens"] = new_max_tokens
        
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.gen("chain_of_thought", **sampling_parameters)
        
        results = resp.run_batch([{"prefix": prefix} for _ in range(branch_factor)])
        
        # assert self._sem_program is not None
        # async with self._sem_program:
        #     results = resp.run_batch([{"prefix": prefix} for _ in range(branch_factor)])
        
        nodes = []
        for result in results:
            chain_of_thought = result["chain_of_thought"]
            finish_reason = result.get_meta_info("chain_of_thought")["finish_reason"]['type']
            full_text = result.text()
            
            node_text = self.node_text_template.format(
                chain_of_thought=chain_of_thought
            )

            node = {
                "text": node_text,
                "depth": depth,
                "full_text": full_text,
                "stop_text": None,
                "finish_reason": finish_reason,
            }
            nodes.append(node)
        
        if branch_factor > 1:
            assert len(nodes) == branch_factor
        
        return nodes

    def _compute_max_tokens(
        self, prefix: str, prompt_max_token: Optional[int] = None
    ) -> int:
        num_prefix_tokens = len(self.tokenizer.tokenize(prefix))
        return min(
            self.model_context_size - num_prefix_tokens,
            prompt_max_token if prompt_max_token is not None else float("inf"),
        )

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        branch_factor = self.branch_factor_strategy(current_node)
        tasks = []
        for i in range(self.num_expansion_rounds):
            seed = self.seed
            if seed is not None:
                seed += i
            task = asyncio.create_task(
                self._sample_node(prefix, depth + 1, branch_factor, seed=seed)
            )
            tasks.append(task)

        all_nodes = []
        for task in tasks:
            nodes = await task
            all_nodes.extend(nodes)

        return all_nodes


@NodeExpander.register("confidence_interval_aware_efficient_iid")
class ConfidenceIntervalAwareEfficientIIDExpander(EfficientIIDExpander):
    def __init__(
        self,
        acceptable_ci_length_threshold: float,
        max_num_rollouts: int,
        num_new_rollouts: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.acceptable_ci_length_threshold = acceptable_ci_length_threshold
        self.max_num_rollouts = max_num_rollouts
        self.num_new_rollouts = num_new_rollouts
        self.rollout_eval_callback: Optional[Callable[..., float]] = None

        from openrl.inference_strategies.tree_inference.answer_extraction import \
            AnswerExtractor

        self.answer_extractor: Optional[AnswerExtractor] = None

    def set_rollout_eval_callback(self, callback: Callable[..., float]) -> None:
        self.rollout_eval_callback = callback

    def set_answer_extractor(self, answer_extractor) -> None:
        self.answer_extractor = answer_extractor

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        assert "_request_object" in current_node
        request_obj = current_node["_request_object"]

        branch_factor = self.branch_factor_strategy(current_node)

        num_nodes_per_round = branch_factor
        initial_num_nodes = branch_factor * self.num_expansion_rounds

        async def _sample_new_nodes(num_nodes: int) -> List[Node]:
            i = 0
            sampled = 0
            tasks = []
            while sampled < num_nodes:
                to_sample = min(num_nodes_per_round, num_nodes - sampled)

                task = asyncio.create_task(
                    self._sample_node(prefix, depth + 1, to_sample)
                )
                tasks.append(task)

                sampled += to_sample
                i += 1

            nodes_lst = []
            for task in tasks:
                nodes = await task
                nodes_lst.extend(nodes)

            return nodes_lst

        async def _compute_rewards(nodes_lst: List[Node]) -> List[float]:
            rewards = []
            for node in nodes_lst:
                answer = await self.answer_extractor.extract_from_node(node)
                reward = self.rollout_eval_callback(
                    query=prefix,
                    rollout=answer,
                    finish_reason=node["finish_reason"],
                    request_object=request_obj,
                )
                rewards.append(reward)
            return rewards

        all_nodes = await _sample_new_nodes(initial_num_nodes)
        all_rewards = await _compute_rewards(all_nodes)

        ci_length = self._compute_confidence_interval_length(all_rewards)

        while ci_length > self.acceptable_ci_length_threshold:
            if len(all_nodes) >= self.max_num_rollouts:
                break
            logger.info(
                f"Resampling {self.num_new_rollouts} more rollouts (curr #rolls: {len(all_nodes)}): "
                f"ci_length: {ci_length}, threshold: {self.acceptable_ci_length_threshold}"
            )
            new_nodes = await _sample_new_nodes(self.num_new_rollouts)
            new_rewards = await _compute_rewards(new_nodes)

            all_nodes += new_nodes
            all_rewards += new_rewards

            ci_length = self._compute_confidence_interval_length(all_rewards)

        current_node["ci_length"] = ci_length

        return all_nodes

    def _compute_confidence_interval_length(
        self,
        rewards: List[float],
        t_score: float = 1.96,
    ) -> float:
        estimate = np.mean(rewards)
        ci = t_score * np.sqrt(estimate * (1 - estimate) / len(rewards))
        return 2 * ci


@NodeExpander.register("efficient_iid_for_tree")
class EfficientIIDExpanderForTree(EfficientIIDExpander):
    def __init__(
        self,
        max_branching_depth: int,
        intermediate_stop_sequence: str,
        full_response_stop_sequence: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_branching_depth = max_branching_depth

        if "stop" in self.sampling_parameters:
            program_kwarg_stop = self.sampling_parameters.pop("stop")
            logger.warning(
                f"stop will be overridden by intermediate_stop_sequence and final_stop_sequence. "
                f"Provided stop: {program_kwarg_stop}"
            )

        self.intermediate_stop_sequence = intermediate_stop_sequence
        self.final_stop_sequence = full_response_stop_sequence

    async def _sample_node(
        self,
        prefix: str,
        depth: int,
        branch_factor: int,
        use_intermediate_stop: bool = False,
    ) -> List[Node]:
        sampling_parameters = self.sampling_parameters.copy()
        sampling_parameters["stop"] = (
            self.intermediate_stop_sequence
            if use_intermediate_stop
            else self.final_stop_sequence
        )
        
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.gen("chain_of_thought", **sampling_parameters)
        
        results = resp.run_batch([{"prefix": prefix} for _ in range(branch_factor)])
        
        # assert self._sem_program is not None
        # async with self._sem_program:
        #     results = resp.run_batch([{"prefix": prefix} for _ in range(branch_factor)])
        
        nodes = []
        for result in results:
            chain_of_thought = result["chain_of_thought"]
            finish_reason = result.get_meta_info("chain_of_thought")["finish_reason"]['type']
            full_text = result.text()
            
            node_text = self.node_text_template.format(
                chain_of_thought=chain_of_thought
            )

            node = {
                "text": node_text,
                "depth": depth,
                "full_text": full_text,
                "stop_text": (
                    None
                    if not use_intermediate_stop
                    else self.intermediate_stop_sequence
                ),
                "finish_reason": finish_reason,
            }
            nodes.append(node)
        
        if branch_factor > 1:
            assert len(nodes) == branch_factor
        
        return nodes

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        branch_factor = self.branch_factor_strategy(current_node)
        if depth > self.max_branching_depth:
            assert branch_factor == 1, (
                f"branch_factor must be 1, but "
                f"got {branch_factor}, depth: {depth}, max_branching_depth: {self.max_branching_depth}"
            )

        use_intermediate_stop = depth <= self.max_branching_depth

        tasks = []
        for _ in range(self.num_expansion_rounds):
            task = asyncio.create_task(
                self._sample_node(
                    prefix,
                    depth + 1,
                    branch_factor,
                    use_intermediate_stop=use_intermediate_stop,
                )
            )
            tasks.append(task)

        all_nodes = []
        for task in tasks:
            nodes = await task
            all_nodes.extend(nodes)

        return all_nodes


@NodeExpander.register("high_low_temperature_iid")
class HighLowTemperatureIIDExpander(IIDExpander):
    def __init__(
        self, 
        node_text_template: str, 
        high_temp_params: JsonDict, 
        low_temp_params: JsonDict, 
        **kwargs
    ):
        super(IIDExpander, self).__init__(**kwargs)
        self.high_temp_params = high_temp_params
        self.low_temp_params = low_temp_params

        self.node_text_template = node_text_template
        assert (
            "{chain_of_thought_1}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"
        assert (
            "{chain_of_thought_2}" in self.node_text_template
        ), "Node_text_template must contain '{chain_of_thought}'"
    
    async def _sample_node(self, prefix: str, depth: int) -> Node:
        
        high_temp_params = self.high_temp_params
        low_temp_params = self.low_temp_params
        
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.gen("chain_of_thought_1", **high_temp_params)
            s += sgl.gen("chain_of_thought_2", **low_temp_params)
        
        result = resp.run(prefix=prefix)
        
        # assert self._sem_program is not None
        # async with self._sem_program:
        #     result = resp.run(prefix=prefix)
        
        chain_of_thought_1 = result["chain_of_thought_1"]
        chain_of_thought_2 = result["chain_of_thought_2"]
        if 'matched' in result.get_meta_info("chain_of_thought_2")['finish_reason']:
            stop_text = result.get_meta_info("chain_of_thought_2")['finish_reason']['matched']
        else:
            stop_text = None

        node_text = self.node_text_template.format(
            chain_of_thought_1=chain_of_thought_1,
            chain_of_thought_2=chain_of_thought_2
        )

        node = {
            "text": node_text,
            "depth": depth,
            "full_text": result.text(),
            "stop_text": stop_text,
        }

        return node


@NodeExpander.register("bleu_rejection_sampling_iid")
class BleuRejectionSamplingIIDExpander(IIDExpander):
    def __init__(
        self,
        node_text_template: str,
        sampling_parameters: JsonDict,
        bleu_acc_threshold: float,
        max_try: int,
        **kwargs,
    ):
        super().__init__(node_text_template, sampling_parameters, **kwargs)
        # Use a random experiment id
        self.bleu = evaluate.load("bleu", experiment_id=str(uuid.uuid4()))
        self.bleu_acc_threshold = bleu_acc_threshold
        self.max_try = max_try

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        acc_nodes = []
        try_counter = 0
        branch_factor = self.branch_factor_strategy(current_node)
        while len(acc_nodes) < branch_factor and try_counter < self.max_try:
            nodes = await super().expand(current_node, prefix, depth)
            try_counter += 1

            for node in nodes:
                if len(acc_nodes) == branch_factor:
                    break

                if len(acc_nodes) == 0:
                    acc_nodes.append(node)
                    continue

                if try_counter == self.max_try:
                    logger.info(
                        f"max_try reached, achieved {len(acc_nodes)} nodes, will append remaining nodes"
                    )
                    acc_nodes.append(node)
                    continue

                acc_texts = [acc_node["text"] for acc_node in acc_nodes]
                avg_bleu = self.avg_bleu_of_those_with_this(
                    those_texts=acc_texts, this_text=node["text"]
                )
                if avg_bleu <= self.bleu_acc_threshold:
                    acc_nodes.append(node)

        return acc_nodes

    def avg_bleu_of_those_with_this(self, *, those_texts, this_text):
        preds = []
        refs = []
        for i in range(len(those_texts)):
            that_text = those_texts[i]
            preds.append(that_text)
            refs.append(this_text)
        bleu_full_stats = self.bleu.compute(predictions=preds, references=refs)
        bleu = bleu_full_stats["bleu"]
        return bleu


@NodeExpander.register("iid_with_different_system_message")
class IIDWithDifferentSystemMessageExpander(IIDExpander):
    def __init__(self, system_messages: List[str], sys_msg_regex: str, **kwargs):
        super().__init__(**kwargs)
        self.system_messages = system_messages
        self.sys_msg_regex = re.compile(sys_msg_regex)
        seed = int(os.environ.get("APP_SEED", "42"))
        self.rng = random.Random(seed)

    async def expand(self, current_node: Node, prefix: str, depth: int) -> List[Node]:
        tasks = []
        branch_factor = self.branch_factor_strategy(current_node)
        assert len(self.system_messages) >= branch_factor

        # Sample system messages
        system_messages = self.rng.sample(self.system_messages, branch_factor)
        for new_sys_msg in system_messages:
            # Replace system message in the prefix
            old_sys_msg = self.sys_msg_regex.search(prefix).group(1)
            prefix = prefix.replace(old_sys_msg, new_sys_msg)

            # Sample node with the new prefix
            task = asyncio.create_task(self._sample_node(prefix, depth + 1))
            tasks.append(task)

        nodes = []
        for task in tasks:
            node = await task
            nodes.append(node)

        return nodes