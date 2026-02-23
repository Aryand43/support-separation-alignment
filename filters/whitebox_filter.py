from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

import numpy as np

from utils.data_generation import Response


@dataclass
class WhiteBoxSpec:
    max_queries: int = 8
    capacity: int = 16
    entropy_weight: float = 0.6
    confidence_weight: float = 0.4


class WhiteBoxFilter:
    """
    White-box filter without trapdoor:
      - can inspect response-space probabilities/logits,
      - cannot access hidden PRF key or harmful oracle.
    """

    def __init__(self, model, spec: WhiteBoxSpec, seed: int = 0):
        self.model = model
        self.spec = spec
        self._rng = np.random.default_rng(seed)
        self.total_queries = 0
        self._support = self.model.response_space() if hasattr(self.model, "response_space") else []

    def _proxy_risk(self, x: int, y: Response, p_y: float) -> float:
        c, r, b = y
        parity_agree = 1.0 - abs((sum(c) % 2) - b)
        prompt_agree = 1.0 - abs((x % 2) - b)
        even_noise = 1.0 if (sum(r) % 2) == 0 else 0.0
        confidence = -math.log(max(p_y, 1e-12))
        # Uses rich model likelihood information but only observable structure.
        return (
            self.spec.entropy_weight * confidence
            + self.spec.confidence_weight * (0.4 * parity_agree + 0.4 * prompt_agree + 0.2 * even_noise)
        )

    def sample(self, x: int, n_samples: int = 1) -> List[Response]:
        outs: List[Response] = []
        probs = self.model.probabilities(x) if hasattr(self.model, "probabilities") else None
        if probs is None or not self._support:
            return self.model.sample(x, n_samples=n_samples)

        for _ in range(n_samples):
            # Candidate set from top-likelihood outputs (white-box access).
            top_k = min(len(self._support), max(4, 2 * self.spec.capacity))
            idx_sorted = np.argsort(-probs)[:top_k]
            best_y = None
            best_score = float("inf")

            q = min(self.spec.max_queries, len(idx_sorted))
            for i in idx_sorted[:q]:
                self.total_queries += 1
                y = self._support[int(i)]
                score = self._proxy_risk(x, y, float(probs[int(i)]))
                if score < best_score:
                    best_score = score
                    best_y = y

            if best_y is None:
                best_y = self.model.sample(x, n_samples=1)[0]
            outs.append(best_y)
        return outs

    def reset_stats(self) -> None:
        self.total_queries = 0

    def log_prob(self, x: int, y: Response) -> float:
        n = 2048
        draws = self.sample(x, n_samples=n)
        hit = sum(1 for d in draws if d == y)
        return math.log(max(hit / n, 1e-12))
