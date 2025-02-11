from typing import Optional

import sglang as sgl

from openrl.common import JsonDict, Registrable, logging_utils
from openrl.common.py_utils import format_string
from openrl.inference_strategies.tree_inference import Node

logger = logging_utils.get_logger(__name__)


class AnswerExtractor(Registrable):
    def __init__(self, server_url: str, seed: Optional[int] = None, **kwargs):
        self.server_url = server_url
        self.seed = seed
        
        sgl.set_default_backend(sgl.RuntimeEndpoint(server_url))

    def set_seed(self, seed: int):
        self.seed = seed

    def extract_from_node(self, node: Node) -> str:
        return self.extract(node["full_text"])

    def extract(self, full_text: str) -> str:
        raise NotImplementedError()


@AnswerExtractor.register("next_chat_turn")
class NextTurnAnswerExtractor(AnswerExtractor):
    def __init__(self, sampling_parameters: JsonDict):
        super().__init__()
        self.sampling_parameters = sampling_parameters

    def extract(self, full_text: str) -> str:
        
        sampling_parameters = self.sampling_parameters
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.gen("final_answer", **sampling_parameters)
        
        result = resp.run(prefix=full_text)
        final_answer = result["final_answer"]

        return final_answer


@AnswerExtractor.register("next_chat_turn_code")
class NextTurnCodeAnswerExtractor(NextTurnAnswerExtractor):
    def extract(self, full_text: str) -> str:
        final_answer = super().extract(full_text)
        final_answer = "```\ndef " + final_answer
        return final_answer


@AnswerExtractor.register("next_chat_turn_ABCD_choices")
class NextTurnABCDAnswerExtractor(AnswerExtractor):
    def __init__(self, sampling_parameters: JsonDict):
        super().__init__()
        self.sampling_parameters = sampling_parameters

    def extract(self, full_text: str) -> str:
        sampling_parameters = self.sampling_parameters
        @sgl.function
        def resp(s, prefix):
            s += prefix
            s += sgl.select("final_answer", ["A", "B", "C", "D"], **sampling_parameters)
        
        result = resp.run(prefix=full_text)
        final_answer = result["final_answer"]
        
        # make sure just one answer is in the final answer
        count = 0
        for ch in ["A", "B", "C", "D"]:
            if ch in final_answer:
                count += 1
        if count != 1:
            return "no-answer"
        
        for ch in ["A", "B", "C", "D"]:
            if ch in final_answer:
                return ch


@AnswerExtractor.register("identity")
class IdentityAnswerExtractor(AnswerExtractor):
    def __init__(self, node_key_name: str = "full_text", **kwargs):
        super().__init__(**kwargs)
        self.node_key_name = node_key_name

    def extract_from_node(self, node: Node) -> str:
        return node[self.node_key_name]


@AnswerExtractor.register("identity_with_solution_prefix")
class IdentityWithSolutionPrefix(IdentityAnswerExtractor):
    def __init__(
        self, solution_prefix: str, end_of_turn_token: Optional[str] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.solution_prefix = solution_prefix
        self.end_of_turn_token = end_of_turn_token

    def extract_from_node(self, node: Node) -> str:
        answer = super().extract_from_node(node)
        parts = answer.split(self.solution_prefix)
        assert (
            len(parts) >= 2
        ), f"Expected '{self.solution_prefix}' in answer. Got:\n{answer}"
        # Remove the first part
        parts = parts[1:]
        out = self.solution_prefix.join(parts)

        if self.end_of_turn_token is not None:
            finish_reason = node["finish_reason"]
            if finish_reason != "length":
                out += self.end_of_turn_token
                node["full_text"] += self.end_of_turn_token

        return out
