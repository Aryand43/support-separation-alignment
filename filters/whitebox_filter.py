"""White-box filter using token-level confidence heuristics."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters.metrics import risk_score


class WhiteBoxFilter:
    """
    Combines keyword-based risk scoring with length/entropy heuristics
    to pick the lowest-risk candidate under a bounded query budget.
    """

    def __init__(self, generator, max_queries: int = 4):
        self.generator = generator
        self.max_queries = max_queries
        self.total_queries = 0

    def _adjusted_risk(self, text: str) -> float:
        base = risk_score(text)
        length_penalty = max(0, len(text) - 300) * 0.0005
        short_penalty = 0.05 if len(text.strip()) < 30 else 0.0
        return base + length_penalty + short_penalty

    def sample(self, prompt: str, n: int = 1) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            best_text = ""
            best_risk = 1.0
            for _ in range(self.max_queries):
                self.total_queries += 1
                text = self.generator.sample(prompt, n_samples=1)[0]
                r = self._adjusted_risk(text)
                if r < best_risk:
                    best_risk = r
                    best_text = text
                if risk_score(text) == 0.0:
                    break
            results.append(best_text)
        return results

    def reset_stats(self) -> None:
        self.total_queries = 0
