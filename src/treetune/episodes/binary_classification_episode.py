from dataclasses import dataclass
from typing import List

@dataclass
class BinaryClassificationEpisode:
    query_token_ids: List[int]
    target_prob: float

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert self.target_prob is not None
