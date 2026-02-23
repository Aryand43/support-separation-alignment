from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math

import numpy as np

from utils.data_generation import (
    Response,
    enumerate_responses,
    response_to_feature_vector,
)


@dataclass
class ToyGeneratorSpec:
    challenge_bits: int
    noise_bits: int
    hidden_dim: int = 32
    temperature: float = 1.0
    harmful_bias: float = 1.6
    nonharmful_bias: float = 0.2
    seed: int = 7


class ToyBaseGenerator:
    """
    Toy base generator over y = (c, r, b), where:
      c: challenge bits
      r: noise bits
      b: pseudo-PRF prediction bit

    Harmfulness is trapdoor-defined through a hidden keyed MLP.
    """

    def __init__(self, spec: ToyGeneratorSpec):
        self.spec = spec
        self._rng = np.random.default_rng(spec.seed)
        self._response_space = enumerate_responses(spec.challenge_bits, spec.noise_bits)
        self._index_map = {y: i for i, y in enumerate(self._response_space)}
        self._prob_cache: dict[int, np.ndarray] = {}
        g = self._rng.normal

        in_dim = 1 + spec.challenge_bits
        self._prf_w1 = g(0.0, 0.6, size=(in_dim, spec.hidden_dim))
        self._prf_b1 = g(0.0, 0.2, size=(spec.hidden_dim,))
        self._prf_w2 = g(0.0, 0.6, size=(spec.hidden_dim, 1))
        self._prf_b2 = g(0.0, 0.2, size=(1,))

        feat_dim = len(response_to_feature_vector(0, self._response_space[0]))
        self._score_w1 = g(0.0, 0.4, size=(feat_dim, spec.hidden_dim))
        self._score_b1 = g(0.0, 0.1, size=(spec.hidden_dim,))
        self._score_w2 = g(0.0, 0.4, size=(spec.hidden_dim, 1))
        self._score_b2 = g(0.0, 0.1, size=(1,))

    def response_space(self) -> List[Response]:
        return list(self._response_space)

    def _prf_bit(self, x: int, c_bits: Sequence[int]) -> int:
        c_arr = np.array(c_bits, dtype=np.float32)
        x_arr = np.array([float(x % 2)], dtype=np.float32)
        inp = np.concatenate([x_arr, c_arr], axis=0)
        h = np.tanh(inp @ self._prf_w1 + self._prf_b1)
        out = float((h @ self._prf_w2 + self._prf_b2).item())
        return int(out > 0.0)

    def is_harmful_with_trapdoor(self, x: int, y: Response) -> bool:
        c, r, b = y
        prf_match = int(b) == self._prf_bit(x, c)
        even_noise_parity = (sum(r) % 2) == 0
        return bool(prf_match and even_noise_parity)

    def _base_score(self, x: int, y: Response) -> float:
        c, r, b = y
        feat = np.array(response_to_feature_vector(x, y), dtype=np.float64)
        h = np.maximum(feat @ self._score_w1 + self._score_b1, 0.0)
        learned = float((h @ self._score_w2 + self._score_b2).item())
        prf_match = b == self._prf_bit(x, c)
        harmful = prf_match and ((sum(r) % 2) == 0)
        if harmful:
            trapdoor_bonus = self.spec.harmful_bias
        elif prf_match:
            trapdoor_bonus = 0.8
        else:
            trapdoor_bonus = self.spec.nonharmful_bias
        return (learned + trapdoor_bonus) / max(self.spec.temperature, 1e-8)

    def _all_scores(self, x: int) -> np.ndarray:
        return np.array([self._base_score(x, y) for y in self._response_space], dtype=np.float64)

    def probabilities(self, x: int) -> np.ndarray:
        if x in self._prob_cache:
            return self._prob_cache[x]
        scores = self._all_scores(x)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        self._prob_cache[x] = probs
        return probs

    def sample(self, x: int, n_samples: int = 1) -> List[Response]:
        probs = self.probabilities(x)
        idx = self._rng.choice(len(self._response_space), size=n_samples, p=probs)
        return [self._response_space[int(i)] for i in np.atleast_1d(idx)]

    def log_prob(self, x: int, y: Response) -> float:
        i = self._index_map.get(y)
        if i is None:
            return float("-inf")
        p = float(self.probabilities(x)[i])
        return math.log(max(p, 1e-300))

    def train(self, _: int = 0) -> None:
        """
        Placeholder to mirror experiment pipelines.
        The model is intentionally random/frozen to represent a fixed generator.
        """
        return None
