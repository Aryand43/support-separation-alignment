"""
Cost analysis: harmful rate vs query count and estimated compute cost.

Reads the CSV produced by run_llm_eval.py and generates:
  - harmful_rate_vs_query_count.png
  - harmful_rate_vs_estimated_cost.png
  - cost_summary.csv

Usage:
    python experiments/cost_analysis.py --metrics outputs/llm_results_guardrail/llm_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COST_PER_QUERY_USD = 0.0003


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit(f"[ERROR] No rows with status='ok' in {path}")
    return ok


def compute_summary(ok: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        ok.groupby(["model", "filter_type", "filter_budget"])
        .agg(
            harmful_rate=("is_harmful", "mean"),
            mean_queries=("total_queries", "mean"),
            total_queries_sum=("total_queries", "sum"),
            n_prompts=("is_harmful", "count"),
        )
        .reset_index()
    )
    grouped["estimated_cost_usd"] = grouped["total_queries_sum"] * COST_PER_QUERY_USD
    return grouped


def plot_harm_vs_queries(summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=summary, x="mean_queries", y="harmful_rate",
        hue="filter_type", style="model", marker="o", linewidth=2,
    )
    plt.title("Harmful Rate vs Mean Query Count")
    plt.xlabel("Mean Queries per Prompt")
    plt.ylabel("Harmful Rate")
    plt.ylim(-0.02, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "harmful_rate_vs_query_count.png", dpi=200)
    plt.close()


def plot_harm_vs_cost(summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=summary, x="estimated_cost_usd", y="harmful_rate",
        hue="filter_type", style="model", marker="s", linewidth=2,
    )
    plt.title("Harmful Rate vs Estimated Compute Cost")
    plt.xlabel("Estimated Cost (USD)")
    plt.ylabel("Harmful Rate")
    plt.ylim(-0.02, 1.02)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "harmful_rate_vs_estimated_cost.png", dpi=200)
    plt.close()


def plot_filter_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    """Harm-vs-budget with all four filters on the same axes."""
    sns.set_theme(style="whitegrid", context="talk")
    avg = (
        summary.groupby(["filter_type", "filter_budget"])["harmful_rate"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=avg, x="filter_budget", y="harmful_rate",
        hue="filter_type", marker="o", linewidth=2.5,
    )
    plt.title("Harmful Rate vs Filter Budget — All Filters Compared")
    plt.xlabel("Filter Budget (max queries)")
    plt.ylabel("Harmful Rate (avg across models)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "harm_vs_budget_guardrail_comparison.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost analysis for bounded filter evaluation")
    parser.add_argument("--metrics", type=str, required=True, help="Path to llm_metrics.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots/csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    ok = load_metrics(metrics_path)

    out_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(ok)
    summary.to_csv(out_dir / "cost_summary.csv", index=False)
    print(f"Cost summary -> {out_dir / 'cost_summary.csv'}")

    plot_harm_vs_queries(summary, out_dir)
    plot_harm_vs_cost(summary, out_dir)
    plot_filter_comparison(summary, out_dir)

    print(f"Cost analysis plots -> {out_dir}/")

    print("\n--- Diminishing Returns Summary ---")
    for ft in sorted(summary["filter_type"].unique()):
        sub = summary[summary["filter_type"] == ft].sort_values("filter_budget")
        rates = sub.groupby("filter_budget")["harmful_rate"].mean()
        print(f"\n  {ft}:")
        for budget, rate in rates.items():
            print(f"    budget={budget:>3d}  harmful_rate={rate:.4f}")


if __name__ == "__main__":
    main()
