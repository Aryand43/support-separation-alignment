"""Black-box bounded filter wrapping a real LLM via sampling."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import risk_score


class BoundedBlackBoxFilter:
    """
    For each prompt, draws up to *max_queries* candidate outputs from the
    base LLM and returns the one with the lowest proxy risk score.
    """

    def __init__(self, generator, max_queries: int = 4):
        self.generator = generator
        self.max_queries = max_queries
        self.total_queries = 0

    def sample(self, prompt: str, n: int = 1) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            best_text = ""
            best_risk = 1.0
            for _ in range(self.max_queries):
                self.total_queries += 1
                text = self.generator.sample(prompt, n_samples=1)[0]
                r = risk_score(text)
                if r < best_risk:
                    best_risk = r
                    best_text = text
                if r == 0.0:
                    break
            results.append(best_text)
        return results

    def reset_stats(self) -> None:
        self.total_queries = 0
