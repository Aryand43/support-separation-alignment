from __future__ import annotations

from typing import Callable, List
import math

import numpy as np

from utils.data_generation import Response


HarmfulOracle = Callable[[int, Response], bool]


class IdealEliminator:
    """
    Oracle eliminator:
        P*(y|x) ∝ P_theta(y|x) * 1[h(y)=0]
    """

    def __init__(self, base_model, harmful_oracle: HarmfulOracle, seed: int = 0):
        self.base_model = base_model
        self.harmful_oracle = harmful_oracle
        self._rng = np.random.default_rng(seed)
        self._support = self.base_model.response_space()
        self._index_map = {y: i for i, y in enumerate(self._support)}
        self._prob_cache: dict[int, np.ndarray] = {}

    def probabilities(self, x: int) -> np.ndarray:
        if x in self._prob_cache:
            return self._prob_cache[x]
        if hasattr(self.base_model, "probabilities"):
            base_probs = np.array(self.base_model.probabilities(x), dtype=np.float64)
        else:
            base_probs = np.array([math.exp(self.base_model.log_prob(x, y)) for y in self._support], dtype=np.float64)
        mask = np.array([0.0 if self.harmful_oracle(x, y) else 1.0 for y in self._support], dtype=np.float64)
        filtered = base_probs * mask
        z = np.sum(filtered)
        if z <= 0:
            # If all support is harmful, revert to base distribution (defensive).
            probs = base_probs / np.sum(base_probs)
            self._prob_cache[x] = probs
            return probs
        probs = filtered / z
        self._prob_cache[x] = probs
        return probs

    def sample(self, x: int, n_samples: int = 1) -> List[Response]:
        probs = self.probabilities(x)
        idx = self._rng.choice(len(self._support), size=n_samples, p=probs)
        return [self._support[int(i)] for i in np.atleast_1d(idx)]

    def log_prob(self, x: int, y: Response) -> float:
        i = self._index_map.get(y)
        if i is None:
            return float("-inf")
        p = float(self.probabilities(x)[i])
        if p <= 0.0:
            return float("-inf")
        return math.log(p)
