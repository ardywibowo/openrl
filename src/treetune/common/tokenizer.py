from typing import Any, Dict, Optional

from transformers import (AutoTokenizer, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)

from .registrable import Registrable


class Tokenizer(PreTrainedTokenizerBase, Registrable):
    pass

class DIPreTrainedTokenizer(Tokenizer):
    @classmethod
    def from_di(
        cls, hf_model_name: str, pretrained_args: Optional[Dict[str, Any]] = None, **kwargs
    ) -> PreTrainedTokenizerFast:
        if pretrained_args is None:
            pretrained_args = {}

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, use_fast=True, **pretrained_args
        )
        return tokenizer


Tokenizer.register("pretrained", constructor="from_di", exist_ok=True)(
    DIPreTrainedTokenizer
)
