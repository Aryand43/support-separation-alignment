from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List
import math

import numpy as np

from utils.data_generation import Response


SQFeature = Callable[[int, Response], float]


@dataclass
class StatisticalQuerySpec:
    num_stat_queries: int = 8
    samples_per_query: int = 32
    max_candidate_draws: int = 8
    reject_threshold: float = 0.65


class StatisticalQueryFilter:
    """
    SQ-style filter using estimated feature expectations.

    Access pattern:
      - Queries model only through sampling.
      - Computes expectations E[phi_i(x, y)] via finite samples.
      - Rejects candidates whose feature-correlation score exceeds threshold.
    """

    def __init__(self, model, spec: StatisticalQuerySpec, seed: int = 0):
        self.model = model
        self.spec = spec
        self._rng = np.random.default_rng(seed)
        self._cache: Dict[int, np.ndarray] = {}
        self.total_queries = 0
        self.total_stat_queries = 0

    def _feature_bank(self) -> List[SQFeature]:
        return [
            lambda x, y: float(sum(y[0]) % 2),  # challenge parity
            lambda x, y: float(sum(y[1]) % 2),  # noise parity
            lambda x, y: float(y[2]),  # b bit
            lambda x, y: float((x % 2) == y[2]),  # prompt-bit agreement
            lambda x, y: float((sum(y[0]) % 2) == y[2]),  # challenge-bit agreement
            lambda x, y: float(sum(y[1])) / max(len(y[1]), 1),  # noise density
            lambda x, y: float((sum(y[1]) % 2) == 0),  # even-noise indicator
            lambda x, y: float((sum(y[0]) + sum(y[1]) + y[2]) % 2),  # joint parity
        ]

    def _estimate_expectations(self, x: int) -> np.ndarray:
        if x in self._cache:
            return self._cache[x]

        features = self._feature_bank()
        k = min(self.spec.num_stat_queries, len(features))
        estimates: List[float] = []
        for i in range(k):
            phi = features[i]
            ys = self.model.sample(x, n_samples=self.spec.samples_per_query)
            self.total_queries += self.spec.samples_per_query
            self.total_stat_queries += 1
            estimates.append(float(np.mean([phi(x, y) for y in ys])))

        est = np.array(estimates, dtype=np.float64)
        self._cache[x] = est
        return est

    def _candidate_score(self, x: int, y: Response, est: np.ndarray) -> float:
        features = self._feature_bank()
        k = min(len(est), len(features))
        obs = np.array([features[i](x, y) for i in range(k)], dtype=np.float64)
        # Correlation-like proxy: larger deviations from expected means are treated as riskier.
        z = np.abs(obs - est)
        return float(np.mean(z))

    def sample(self, x: int, n_samples: int = 1) -> List[Response]:
        est = self._estimate_expectations(x)
        outs: List[Response] = []
        for _ in range(n_samples):
            best = None
            best_score = float("inf")
            for _ in range(self.spec.max_candidate_draws):
                y = self.model.sample(x, n_samples=1)[0]
                self.total_queries += 1
                score = self._candidate_score(x, y, est)
                if score < self.spec.reject_threshold:
                    best = y
                    break
                if score < best_score:
                    best = y
                    best_score = score
            if best is None:
                best = self.model.sample(x, n_samples=1)[0]
                self.total_queries += 1
            outs.append(best)
        return outs

    def reset_stats(self) -> None:
        self.total_queries = 0
        self.total_stat_queries = 0
        self._cache.clear()

    def log_prob(self, x: int, y: Response) -> float:
        n = 2048
        draws = self.sample(x, n_samples=n)
        hit = sum(1 for d in draws if d == y)
        return math.log(max(hit / n, 1e-12))
