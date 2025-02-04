from enum import Enum
from typing import Any, Dict

from .component import Component
from .from_params import ConfigurationError, FromParams
from .lazy import Lazy
from .params import Params
from .registrable import Registrable
from .vllm_server_handler import VLLMServerHandler
from .tokenizer import Tokenizer, DIPreTrainedTokenizer

assert FromParams
assert Lazy
assert Params
assert Registrable
assert ConfigurationError
assert Component
assert VLLMServerHandler
assert Tokenizer
assert DIPreTrainedTokenizer

JsonDict = Dict[str, Any]

class ExperimentStage(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2
    PREDICTION = 3

    @staticmethod
    def from_split(split: str) -> "ExperimentStage":
        stage = {
            "valid": ExperimentStage.VALIDATION,
            "validation": ExperimentStage.VALIDATION,
            "test": ExperimentStage.TEST,
            "train": ExperimentStage.TRAINING,
            "predict": ExperimentStage.PREDICTION,
        }[split]
        return stage

from . import logging_utils

logging_utils.enable_default_handler()
# logging_utils.enable_propagation()
logging_utils.enable_explicit_format()
