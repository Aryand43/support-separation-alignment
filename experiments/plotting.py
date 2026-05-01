"""
Plotting for LLM evaluation results.

Usage:
    python experiments/plotting.py --metrics outputs/llm_results/llm_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _has_attack_type(df: pd.DataFrame) -> bool:
    return "attack_type" in df.columns and df["attack_type"].notna().any()


def generate_llm_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("[WARN] No successful results to plot.")
        return

    summary = (
        ok.groupby(["model", "filter_type", "filter_budget"])["is_harmful"]
        .mean()
        .reset_index()
    )
    summary.columns = ["model", "filter_type", "filter_budget", "harmful_rate"]

    # -- per-model harm vs budget ------------------------------------------------
    for model_name in summary["model"].unique():
        msub = summary[summary["model"] == model_name]
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=msub, x="filter_budget", y="harmful_rate",
            hue="filter_type", marker="o", linewidth=2,
        )
        plt.title(f"Harmful Rate vs Filter Budget: {model_name}")
        plt.xlabel("Filter Budget (max queries)")
        plt.ylabel("Harmful Rate")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"harm_vs_budget_{safe_name}.png", dpi=200)
        plt.close()

    # -- all-models overlay ------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=summary, x="filter_budget", y="harmful_rate",
        hue="model", style="filter_type", marker="o", linewidth=1.5,
    )
    plt.title("Harmful Rate vs Filter Budget (All Models)")
    plt.xlabel("Filter Budget (max queries)")
    plt.ylabel("Harmful Rate")
    plt.ylim(-0.02, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "harm_vs_budget_all_models.png", dpi=200)
    plt.close()

    # -- harm floor by model -----------------------------------------------------
    floor = summary.groupby("model")["harmful_rate"].min().reset_index()
    floor.columns = ["model", "harm_floor"]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=floor, x="model", y="harm_floor", hue="model", palette="Reds_d", legend=False)
    plt.title("Harm Floor by Model (minimum harmful rate across all filters/budgets)")
    plt.xlabel("Model")
    plt.ylabel("Harm Floor")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "harm_floor_by_model.png", dpi=200)
    plt.close()

    # -- attack_type plots (only when the column is present) ---------------------
    if _has_attack_type(ok):
        _plot_harm_vs_budget_by_attack_type(ok, out_dir)
        _plot_harm_floor_by_attack_type_and_model(ok, out_dir)

    print(f"Plots saved to {out_dir}/")


def _plot_harm_vs_budget_by_attack_type(ok: pd.DataFrame, out_dir: Path) -> None:
    """Harm rate vs filter budget, one line per attack_type."""
    at_summary = (
        ok.groupby(["attack_type", "filter_budget"])["is_harmful"]
        .mean()
        .reset_index()
    )
    at_summary.columns = ["attack_type", "filter_budget", "harmful_rate"]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=at_summary, x="filter_budget", y="harmful_rate",
        hue="attack_type", marker="o", linewidth=2,
    )
    plt.title("Harmful Rate vs Filter Budget by Attack Type")
    plt.xlabel("Filter Budget (max queries)")
    plt.ylabel("Harmful Rate")
    plt.ylim(-0.02, 1.02)
    plt.legend(title="attack_type", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "harm_vs_budget_by_attack_type.png", dpi=200)
    plt.close()


def _plot_harm_floor_by_attack_type_and_model(ok: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: harm floor (min harmful rate) per (model, attack_type)."""
    at_model = (
        ok.groupby(["model", "attack_type", "filter_type", "filter_budget"])["is_harmful"]
        .mean()
        .reset_index()
    )
    at_model.columns = ["model", "attack_type", "filter_type", "filter_budget", "harmful_rate"]

    floor = (
        at_model.groupby(["model", "attack_type"])["harmful_rate"]
        .min()
        .reset_index()
    )
    floor.columns = ["model", "attack_type", "harm_floor"]

    n_models = floor["model"].nunique()
    width = max(10, 3 * n_models)
    plt.figure(figsize=(width, 6))
    sns.barplot(
        data=floor, x="model", y="harm_floor", hue="attack_type",
        palette="Set2",
    )
    plt.title("Harm Floor by Attack Type and Model")
    plt.xlabel("Model")
    plt.ylabel("Harm Floor")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="attack_type", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "harm_floor_by_attack_type_model.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LLM evaluation results")
    parser.add_argument("--metrics", type=str, required=True, help="Path to llm_metrics.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Plot output dir")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    df = pd.read_csv(metrics_path)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = metrics_path.parent / "plots"

    generate_llm_plots(df, out_dir)


if __name__ == "__main__":
    main()
