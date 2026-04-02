"""Statistical-query filter using feature-expectation calibration."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.metrics import risk_score


class StatisticalQueryFilter:
    """
    Estimates feature-expectation baselines via a calibration phase, then
    rejects candidates whose risk score exceeds a data-driven threshold.
    """

    def __init__(self, generator, num_stat_queries: int = 4, samples_per_query: int = 2):
        self.generator = generator
        self.num_stat_queries = num_stat_queries
        self.samples_per_query = samples_per_query
        self.total_queries = 0

    def sample(self, prompt: str, n: int = 1) -> list[str]:
        calibration_budget = min(self.num_stat_queries, 3)
        calibration_scores: list[float] = []
        for _ in range(calibration_budget):
            for _ in range(self.samples_per_query):
                self.total_queries += 1
                text = self.generator.sample(prompt, n_samples=1)[0]
                calibration_scores.append(risk_score(text))

        threshold = float(np.mean(calibration_scores) * 0.5) if calibration_scores else 0.3

        remaining = max(1, self.num_stat_queries - calibration_budget)
        results: list[str] = []
        for _ in range(n):
            best_text = ""
            best_risk = 1.0
            for _ in range(remaining):
                self.total_queries += 1
                text = self.generator.sample(prompt, n_samples=1)[0]
                r = risk_score(text)
                if r < best_risk:
                    best_risk = r
                    best_text = text
                if r <= threshold:
                    break
            results.append(best_text)
        return results

    def reset_stats(self) -> None:
        self.total_queries = 0
