from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import math

import numpy as np

from utils.data_generation import Response


RewardModel = Callable[[int, Response], float]


@dataclass
class AlignmentSpec:
    beta: float = 1.0


def default_reward_model(x: int, y: Response) -> float:
    """
    Reward based only on observable syntax-level proxies.
    It intentionally cannot access trapdoor harmfulness.
    """
    c, r, b = y
    c_parity = sum(c) % 2
    r_even = 1.0 if (sum(r) % 2) == 0 else 0.0
    prompt_gap = abs((x % 2) - b)
    parity_gap = abs(c_parity - b)
    # Penalize even-noise responses (correlated with harmfulness) using only visible structure.
    return -1.2 * r_even - 0.45 * prompt_gap - 0.20 * parity_gap


class RLHFAlignedModel:
    """
    Support-preserving alignment via Gibbs reweighting:
        P_tilde(y|x) ∝ P_theta(y|x) * exp(beta * r_phi(x, y))

    Because exp(.) > 0, this cannot introduce hard zeros where P_theta > 0.
    """

    def __init__(
        self,
        base_model,
        spec: AlignmentSpec,
        reward_model: RewardModel | None = None,
        seed: int = 0,
    ):
        self.base_model = base_model
        self.spec = spec
        self.reward_model = reward_model or default_reward_model
        self._rng = np.random.default_rng(seed)
        self._support = self.base_model.response_space()
        self._index_map = {y: i for i, y in enumerate(self._support)}
        self._prob_cache: dict[int, np.ndarray] = {}

    def _unnormalized(self, x: int) -> np.ndarray:
        if hasattr(self.base_model, "probabilities"):
            base_probs = np.array(self.base_model.probabilities(x), dtype=np.float64)
        else:
            base_probs = np.array([math.exp(self.base_model.log_prob(x, y)) for y in self._support], dtype=np.float64)
        rewards = np.array([self.reward_model(x, y) for y in self._support], dtype=np.float64)
        weights = np.exp(self.spec.beta * rewards)
        return base_probs * weights

    def response_space(self) -> List[Response]:
        return list(self._support)

    def probabilities(self, x: int) -> np.ndarray:
        if x in self._prob_cache:
            return self._prob_cache[x]
        u = self._unnormalized(x)
        z = np.sum(u)
        if z <= 0.0:
            # Defensive fallback; should not happen due to strict positivity.
            return np.full_like(u, 1.0 / len(u))
        probs = u / z
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
        return math.log(max(p, 1e-300))
