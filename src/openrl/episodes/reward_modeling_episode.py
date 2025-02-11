from dataclasses import dataclass
from typing import List

@dataclass
class RewardModelingEpisode:
    query_token_ids: List[int]
    chosen_token_ids: List[int]
    rejected_token_ids: List[int]

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.chosen_token_ids) > 0
        assert len(self.rejected_token_ids) > 0
