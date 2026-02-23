from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math

import numpy as np

from utils.data_generation import Response


@dataclass
class FilterSpec:
    max_queries: int = 8
    capacity: int = 16
    accept_threshold: float = 0.45
    bootstrap_samples: int = 128


class BoundedBlackBoxFilter:
    """
    Black-box bounded filter.

    Constraints:
      - Accesses generator only via sample/log_prob APIs.
      - Uses at most max_queries proposals per final sample.
      - No trapdoor or direct parameter access.
    """

    def __init__(self, model, spec: FilterSpec, seed: int = 0):
        self.model = model
        self.spec = spec
        self._rng = np.random.default_rng(seed)
        self._linear: Dict[int, np.ndarray] = {}
        self.total_queries = 0

    def _features(self, x: int, y: Response) -> np.ndarray:
        c, r, b = y
        return np.array(
            [
                1.0,
                float(sum(c)) / max(len(c), 1),
                float(sum(r)) / max(len(r), 1),
                float(sum(c) % 2),
                float(sum(r) % 2),
                float(b),
                float(x % 2),
                float((x + sum(c) + sum(r) + b) % 2),
            ],
            dtype=np.float64,
        )

    def _weak_label(self, x: int, y: Response) -> float:
        """
        Weak proxy available to bounded filters.
        It is intentionally imperfect and does not reveal trapdoor harm exactly.
        """
        c, r, b = y
        parity_gap = abs((sum(c) % 2) - b)
        prompt_gap = abs((x % 2) - b)
        even_noise = 1.0 if (sum(r) % 2) == 0 else 0.0
        score = 0.30 * (1.0 - parity_gap) + 0.20 * (1.0 - prompt_gap) + 0.50 * even_noise
        return 1.0 if score > 0.5 else 0.0

    def _fit_if_needed(self, x: int) -> None:
        if x in self._linear:
            return

        n = max(self.spec.bootstrap_samples, self.spec.capacity * 16)
        ys = self.model.sample(x, n_samples=n)
        x_mat = np.stack([self._features(x, y) for y in ys], axis=0)
        labels = np.array([self._weak_label(x, y) for y in ys], dtype=np.float64)

        w = self._rng.normal(loc=0.0, scale=0.1, size=x_mat.shape[1])
        lr = 0.1 / max(1, self.spec.capacity)
        steps = 20 + self.spec.capacity
        for _ in range(steps):
            logits = x_mat @ w
            probs = 1.0 / (1.0 + np.exp(-logits))
            grad = (x_mat.T @ (probs - labels)) / len(labels)
            w -= lr * grad
        self._linear[x] = w

    def risk_score(self, x: int, y: Response) -> float:
        self._fit_if_needed(x)
        w = self._linear[x]
        z = float(self._features(x, y) @ w)
        return 1.0 / (1.0 + math.exp(-z))

    def sample(self, x: int, n_samples: int = 1) -> List[Response]:
        outs: List[Response] = []
        for _ in range(n_samples):
            best = None
            best_score = 1e9
            for _q in range(self.spec.max_queries):
                self.total_queries += 1
                y = self.model.sample(x, n_samples=1)[0]
                score = self.risk_score(x, y)
                if score < self.spec.accept_threshold:
                    best = y
                    break
                if score < best_score:
                    best = y
                    best_score = score
            outs.append(best if best is not None else self.model.sample(x, n_samples=1)[0])
        return outs

    def reset_stats(self) -> None:
        self.total_queries = 0

    def log_prob(self, x: int, y: Response) -> float:
        # Monte Carlo approximation; experiments primarily use sampling estimators.
        n = 2048
        draws = self.sample(x, n_samples=n)
        hit = sum(1 for d in draws if d == y)
        p = max(hit / n, 1e-12)
        return math.log(p)
