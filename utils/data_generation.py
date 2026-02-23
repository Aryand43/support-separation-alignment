from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence, Tuple


BitTuple = Tuple[int, ...]
Response = Tuple[BitTuple, BitTuple, int]


def int_to_bits(value: int, n_bits: int) -> BitTuple:
    return tuple((value >> i) & 1 for i in range(n_bits))


def bits_to_int(bits: Sequence[int]) -> int:
    out = 0
    for i, b in enumerate(bits):
        out |= (int(b) & 1) << i
    return out


def all_bitstrings(n_bits: int) -> Iterable[BitTuple]:
    return product([0, 1], repeat=n_bits)


def enumerate_responses(challenge_bits: int, noise_bits: int) -> List[Response]:
    all_c = list(all_bitstrings(challenge_bits))
    all_r = list(all_bitstrings(noise_bits))
    out: List[Response] = []
    for c in all_c:
        for r in all_r:
            out.append((tuple(c), tuple(r), 0))
            out.append((tuple(c), tuple(r), 1))
    return out


def response_to_feature_vector(x: int, y: Response) -> List[float]:
    c, r, b = y
    c_sum = float(sum(c))
    r_sum = float(sum(r))
    c_parity = float(sum(c) % 2)
    r_parity = float(sum(r) % 2)
    return [
        float(x),
        c_sum,
        r_sum,
        c_parity,
        r_parity,
        float(b),
        float(x % 2),
        float((x + int(c_parity) + int(r_parity)) % 2),
    ]
