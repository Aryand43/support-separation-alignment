from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, List, Tuple
import math
import time

from utils.data_generation import Response


HarmfulPredicate = Callable[[int, Response], bool]


def estimate_harmful_mass(
    model,
    prompts: Iterable[int],
    harmful_predicate: HarmfulPredicate,
    samples_per_prompt: int,
) -> float:
    total = 0
    harmful = 0
    for x in prompts:
        ys = model.sample(x, n_samples=samples_per_prompt)
        total += len(ys)
        harmful += sum(1 for y in ys if harmful_predicate(x, y))
    return harmful / max(total, 1)


def estimate_empirical_distribution(model, x: int, n_samples: int) -> Dict[Response, float]:
    ys = model.sample(x, n_samples=n_samples)
    counts = Counter(ys)
    return {y: c / n_samples for y, c in counts.items()}


def estimate_tv_distance(
    model_a,
    model_b,
    prompts: Iterable[int],
    n_samples: int,
) -> float:
    tv_values: List[float] = []
    for x in prompts:
        dist_a = estimate_empirical_distribution(model_a, x, n_samples)
        dist_b = estimate_empirical_distribution(model_b, x, n_samples)
        support = set(dist_a.keys()) | set(dist_b.keys())
        l1 = sum(abs(dist_a.get(y, 0.0) - dist_b.get(y, 0.0)) for y in support)
        tv_values.append(0.5 * l1)
    return sum(tv_values) / max(len(tv_values), 1)


def timed_call(fn, *args, **kwargs) -> Tuple[float, object]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, out


def query_complexity(filter_model, produced_samples: int) -> float:
    if not hasattr(filter_model, "total_queries"):
        return math.nan
    return float(filter_model.total_queries) / max(produced_samples, 1)
