from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Tuple


Prompt = int
Response = Tuple[Tuple[int, ...], Tuple[int, ...], int]


class ConditionalDistribution(Protocol):
    def sample(self, x: Prompt, n_samples: int = 1) -> list[Response]:
        ...

    def log_prob(self, x: Prompt, y: Response) -> float:
        ...


HarmfulPredicate = Callable[[Prompt, Response], bool]
AlignmentOperator = Callable[[ConditionalDistribution], ConditionalDistribution]


@dataclass(frozen=True)
class FormalObjects:
    """
    Formal primitives used in this repository.

    X: prompt space
    Y: response space
    h: harmful predicate h(y) (implemented here as h(x, y) for convenience)
    P_theta: base conditional distribution
    A: support-preserving alignment operator
    P_star: ideal eliminator/oracle
    """

    prompt_space: Iterable[Prompt]
    response_space: Iterable[Response]
    harmful_predicate: HarmfulPredicate


def support_preservation_axiom(
    p_theta: ConditionalDistribution,
    aligned: ConditionalDistribution,
    prompt_space: Iterable[Prompt],
    response_space: Iterable[Response],
) -> bool:
    """
    Checks the support-preservation condition:
        If P_theta(y|x) > 0 then (A P_theta)(y|x) > 0.

    The check is numeric and only meaningful on finite spaces.
    """

    for x in prompt_space:
        for y in response_space:
            base_positive = p_theta.log_prob(x, y) > float("-inf")
            aligned_positive = aligned.log_prob(x, y) > float("-inf")
            if base_positive and not aligned_positive:
                return False
    return True


ASSUMPTIONS = {
    "A1": "P_theta has finite support over Y for each x and assigns strictly positive mass over its support.",
    "A2": "Alignment operator A is support-preserving (no hard zeroing on points in supp(P_theta(.|x))).",
    "A3": "Ideal eliminator P_star can access hidden trapdoor signal and sets harmful mass to zero.",
    "A4": "Bounded filters are polynomial-time and only query the generator through black-box sampling.",
    "A5": "One-way functions exist, yielding PRF-style families that are computationally hard to predict without a key.",
}
