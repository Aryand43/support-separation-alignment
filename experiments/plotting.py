from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def generate_plots(df: pd.DataFrame, out_dir: str) -> None:
    _setup_style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Harmful mass vs filter query budget
    plt.figure(figsize=(8, 5))
    sub = df[df["condition"] == "bounded_filter"]
    sns.lineplot(data=sub, x="filter_budget", y="harmful_mass", hue="beta", marker="o")
    plt.title("Harmful Mass vs Filter Query Budget")
    plt.xlabel("Filter Query Budget T")
    plt.ylabel("Estimated Harmful Mass")
    plt.tight_layout()
    plt.savefig(out / "harmful_mass_vs_budget.png", dpi=220)
    plt.close()

    # Harmful mass vs filter capacity
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sub, x="filter_capacity", y="harmful_mass", hue="beta", marker="o")
    plt.title("Harmful Mass vs Filter Capacity")
    plt.xlabel("Filter Capacity")
    plt.ylabel("Estimated Harmful Mass")
    plt.tight_layout()
    plt.savefig(out / "harmful_mass_vs_capacity.png", dpi=220)
    plt.close()

    # TV distance vs runtime
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sub, x="runtime_sec", y="tv_to_ideal", hue="beta", style="filter_budget", s=90)
    plt.title("TV Distance to Ideal vs Runtime")
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Estimated TV Distance to Ideal")
    plt.tight_layout()
    plt.savefig(out / "tv_vs_runtime.png", dpi=220)
    plt.close()

    # Alignment strength vs residual harmful mass
    plt.figure(figsize=(8, 5))
    base_aligned = df[df["condition"] == "rlhf_aligned"]
    sns.lineplot(data=base_aligned, x="beta", y="harmful_mass", marker="o")
    plt.title("Alignment Strength vs Residual Harmful Mass")
    plt.xlabel("Alignment Strength beta")
    plt.ylabel("Estimated Harmful Mass")
    plt.tight_layout()
    plt.savefig(out / "beta_vs_residual_harmful_mass.png", dpi=220)
    plt.close()
