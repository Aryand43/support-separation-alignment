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


def generate_llm_plots(df: pd.DataFrame, out_dir: Path, include_all_models_overlay: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("[WARN] No successful results to plot.")
        return

    summary = (
        ok.groupby(["model", "filter_type", "filter_budget"])["is_harmful"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * (
        (
            summary["mean"]
            * (1 - summary["mean"])
            / summary["count"]
        ) ** 0.5
    )

    # -- main paper-ready thesis figure -----------------------------------------
    target_models = ["gemma-4-26b", "nemotron-3-super-120b", "llama-3.3-70b"]
    paper_budgets = [1, 4, 16, 64]

    bounded = summary[summary["filter_type"] == "bounded"].copy()
    bounded["filter_budget"] = pd.to_numeric(bounded["filter_budget"], errors="coerce")
    bounded = bounded.dropna(subset=["filter_budget"]).sort_values(["model", "filter_budget"])

    # Determine how many distinct budget points each model has.
    model_points = bounded.groupby("model")["filter_budget"].nunique().to_dict()
    model_range = (bounded.groupby("model")["mean"].max() - bounded.groupby("model")["mean"].min()).to_dict()

    # Score models by their harmful rate at the maximum observed budget (lower is better).
    last_points = (
        bounded.sort_values(["model", "filter_budget"])
        .groupby("model", as_index=False)
        .tail(1)[["model", "mean"]]
        .rename(columns={"mean": "mean_at_max_budget"})
    )
    score = dict(zip(last_points["model"], last_points["mean_at_max_budget"]))

    def _rank_models(models: list[str]) -> list[str]:
        return sorted(models, key=lambda m: (score.get(m, float("inf")), m))

    excluded_reasons: dict[str, str] = {}
    # Prefer targets first, then best available alternatives.
    candidates = [m for m in target_models if m in model_points]
    remaining = [m for m in _rank_models(list(model_points.keys())) if m not in candidates]

    chosen: list[str] = []
    min_points = 3
    # Relax minimum points automatically until we can select up to 3 models (or run out).
    while min_points >= 1 and len(chosen) < 3:
        chosen = []
        excluded_reasons = {}
        ordered = candidates + remaining
        for m in ordered:
            pts = model_points.get(m, 0)
            rng = float(model_range.get(m, 0.0))
            if rng < 1e-6:
                excluded_reasons[m] = "flatline bounded curve (no visible change across budgets)"
                continue
            if pts >= min_points and len(chosen) < 3:
                chosen.append(m)
            else:
                if m in model_points:
                    excluded_reasons[m] = f"only {pts} budget points (< {min_points})"
        if len(chosen) >= 3 or min_points == 1:
            break
        min_points -= 1

    main = bounded[bounded["model"].isin(chosen)].copy()
    included_points = {m: int(model_points.get(m, 0)) for m in chosen}

    if chosen and not main.empty:
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=main,
            x="filter_budget",
            y="mean",
            hue="model",
            marker="o",
            linewidth=2.5,
            palette=sns.color_palette("colorblind", n_colors=len(chosen)),
        )
        xticks = sorted(set(paper_budgets).intersection(set(main["filter_budget"].unique().tolist())))
        if not xticks:
            xticks = sorted(main["filter_budget"].unique().tolist())
        ax.set_xticks(xticks)
        ax.set_xlim(min(xticks), max(xticks))
        plt.title("Residual Harm Under Bounded Filtering")
        plt.xlabel("Filter Budget")
        plt.ylabel("Harmful Rate")
        plt.ylim(-0.02, 1.02)
        plt.grid(True, alpha=0.18)
        plt.legend(title="model", frameon=True)
        plt.tight_layout()
        out_path = out_dir / "main_harm_floor_plot.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        # Report selection and exclusions.
        print(f"[MAIN FIGURE] included_models={chosen}")
        print(f"[MAIN FIGURE] datapoints_per_model={included_points}")
        for tm in target_models:
            if tm not in model_points:
                print(f"[MAIN FIGURE] excluded_target={tm} reason=no bounded+ok rows")
            elif tm not in chosen:
                print(f"[MAIN FIGURE] excluded_target={tm} reason={excluded_reasons.get(tm, 'not selected')}")
        if len(chosen) < 3:
            print(f"[WARN] Only {len(chosen)} model(s) available for main figure with current CSV.")
        print(f"[MAIN FIGURE] saved={out_path}")
    else:
        print("[WARN] main_harm_floor_plot.png not generated (no bounded+ok rows found).")

    # -- per-model harm vs budget ------------------------------------------------
    for model_name in summary["model"].unique():
        msub = summary[summary["model"] == model_name]
        # Suppress visually meaningless flatline plots (e.g., tiny slices that sit at 0 or 1).
        if msub["filter_budget"].nunique() < 2 or (msub["mean"].max() - msub["mean"].min()) < 1e-6:
            print(f"[SKIP] per-model plot for {model_name}: sparse/flatline slice")
            continue
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=msub, x="filter_budget", y="mean",
            hue="filter_type", marker="o", linewidth=2,
        )
        ax = plt.gca()
        # Add 95% CI shaded bands per filter type (match line colors when possible).
        line_by_label = {line.get_label(): line for line in ax.lines}
        for ft in msub["filter_type"].unique():
            subset = msub[msub["filter_type"] == ft].sort_values("filter_budget")
            color = None
            if ft in line_by_label:
                color = line_by_label[ft].get_color()
            ax.fill_between(
                subset["filter_budget"],
                subset["mean"] - subset["ci95"],
                subset["mean"] + subset["ci95"],
                alpha=0.2,
                color=color,
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
    # This overlay is intentionally optional (it can look cluttered / toy-like with many models).
    if include_all_models_overlay:
        plt.figure(figsize=(12, 6))
        models = list(summary["model"].unique())
        model_palette = dict(zip(models, sns.color_palette(n_colors=len(models))))
        sns.lineplot(
            data=summary, x="filter_budget", y="mean",
            hue="model", style="filter_type", marker="o", linewidth=1.5,
            palette=model_palette,
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
    # Keep floor plot coherent: only models with non-flatline bounded behavior.
    good_models = [m for m in summary["model"].unique() if float(model_range.get(m, 0.0)) >= 1e-6]
    floor = summary[summary["model"].isin(good_models)].groupby("model")["mean"].min().reset_index()
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
    parser.add_argument(
        "--all-models-overlay",
        action="store_true",
        help="Generate the (potentially cluttered) all-models overlay plot",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    # Prefer cleaned metrics by default when present alongside the raw file.
    if metrics_path.name == "llm_metrics.csv":
        clean_path = metrics_path.with_name("llm_metrics_clean.csv")
        if clean_path.exists():
            print(f"[INFO] Using cleaned metrics: {clean_path}")
            metrics_path = clean_path
    df = pd.read_csv(metrics_path)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = metrics_path.parent / "plots"

    generate_llm_plots(df, out_dir, include_all_models_overlay=args.all_models_overlay)


if __name__ == "__main__":
    main()
