"""
Minimal demo: Random Filter vs Trapdoor Oracle → shows the Harm Floor.

Think of it like this:
  - The "Trapdoor Oracle" knows the SECRET password to tell good from bad outputs.
  - The "Random Filter" is just guessing — it flips a coin to decide what to keep.
  - The GAP between them is the "Harm Floor": the leftover bad stuff that
    no normal filter can remove without knowing the secret.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.base_generator import ToyBaseGenerator, ToyGeneratorSpec
from models.rlhf_alignment import RLHFAlignedModel, AlignmentSpec
from filters.trapdoor_filter import TrapdoorFilter
from utils.data_generation import Response


class RandomFilter:
    """Coin-flip filter: randomly accepts or rejects each sample."""

    def __init__(self, model, max_retries: int = 8, seed: int = 0):
        self.model = model
        self.max_retries = max_retries
        self._rng = np.random.default_rng(seed)

    def sample(self, x: int, n_samples: int = 1) -> list[Response]:
        results: list[Response] = []
        for _ in range(n_samples):
            accepted = None
            for _ in range(self.max_retries):
                y = self.model.sample(x, n_samples=1)[0]
                if self._rng.random() > 0.5:
                    accepted = y
                    break
            if accepted is None:
                accepted = self.model.sample(x, n_samples=1)[0]
            results.append(accepted)
        return results


def measure_harmful_mass(model, prompts, harmful_fn, samples_per_prompt):
    total = 0
    harmful = 0
    for x in prompts:
        ys = model.sample(x, n_samples=samples_per_prompt)
        total += len(ys)
        harmful += sum(1 for y in ys if harmful_fn(x, y))
    return harmful / max(total, 1)


def main():
    security_params = [4, 6, 8]
    n_prompts = 16
    samples_per_prompt = 256
    seed = 42

    rows = []

    for n in security_params:
        print(f"\n=== Security parameter n={n} ===")

        gen = ToyBaseGenerator(ToyGeneratorSpec(
            challenge_bits=n, noise_bits=4, seed=seed + n,
        ))

        aligned = RLHFAlignedModel(
            base_model=gen,
            spec=AlignmentSpec(beta=1.0),
            seed=seed + n + 100,
        )

        random_filt = RandomFilter(model=aligned, max_retries=8, seed=seed + n + 200)
        oracle_filt = TrapdoorFilter(base_generator=gen, model=aligned, seed=seed + n + 300)

        rng = np.random.default_rng(seed + n)
        prompts = [int(rng.integers(0, 2 ** n)) for _ in range(n_prompts)]

        harm_base = measure_harmful_mass(gen, prompts, gen.is_harmful_with_trapdoor, samples_per_prompt)
        harm_random = measure_harmful_mass(random_filt, prompts, gen.is_harmful_with_trapdoor, samples_per_prompt)
        harm_oracle = measure_harmful_mass(oracle_filt, prompts, gen.is_harmful_with_trapdoor, samples_per_prompt)

        print(f"  Base model        : {harm_base:.4f}")
        print(f"  Random filter     : {harm_random:.4f}")
        print(f"  Trapdoor oracle   : {harm_oracle:.4f}")
        print(f"  Harm Floor (gap)  : {harm_random - harm_oracle:.4f}")

        rows.append({"n": n, "condition": "base_model", "harmful_mass": harm_base})
        rows.append({"n": n, "condition": "random_filter", "harmful_mass": harm_random})
        rows.append({"n": n, "condition": "trapdoor_oracle", "harmful_mass": harm_oracle})

    df = pd.DataFrame(rows)

    out_dir = Path("outputs")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "harm_floor_demo.csv", index=False)

    plt.figure(figsize=(8, 5))
    labels = {"base_model": "Base Model", "random_filter": "Random Filter", "trapdoor_oracle": "Trapdoor Oracle"}
    colors = {"base_model": "#e74c3c", "random_filter": "#3498db", "trapdoor_oracle": "#2ecc71"}
    for cond in ["base_model", "random_filter", "trapdoor_oracle"]:
        sub = df[df["condition"] == cond]
        plt.plot(sub["n"], sub["harmful_mass"], marker="o", linewidth=2,
                 label=labels[cond], color=colors[cond])

    plt.fill_between(
        df[df["condition"] == "random_filter"]["n"],
        df[df["condition"] == "random_filter"]["harmful_mass"],
        df[df["condition"] == "trapdoor_oracle"]["harmful_mass"],
        alpha=0.2, color="#95a5a6", label="Harm Floor (gap)",
    )

    plt.xlabel("Security Parameter n")
    plt.ylabel("Harmful Mass")
    plt.title("Harm Floor: Random Filter vs Trapdoor Oracle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "harm_floor_comparison.png", dpi=200)
    plt.close()

    print(f"\nCSV  -> {out_dir / 'harm_floor_demo.csv'}")
    print(f"Plot -> {plots_dir / 'harm_floor_comparison.png'}")


if __name__ == "__main__":
    main()
