from dataclasses import dataclass
from typing import List

@dataclass
class DPOPositiveEpisode:
    query_token_ids: List[int]
    accept_response_token_ids: List[int]
    reject_response_token_ids: List[int]

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.accept_response_token_ids) > 0
        assert len(self.reject_response_token_ids) > 0
