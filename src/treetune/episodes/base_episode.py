from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Episode:
    query_token_ids: List[int]
    response_token_ids: List[int]
    reward: float = None  # Kept for backward compatibility
    scores: float = None
    advantages: Optional[List[float]] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        assert len(self.query_token_ids) > 0
        assert len(self.response_token_ids) > 0

        assert self.reward is not None or self.scores is not None

        if self.reward is not None:
            self.scores = self.reward
        elif self.scores is not None:
            self.reward = self.scores

        if self.advantages is not None:
            assert len(self.response_token_ids) == len(self.advantages)
