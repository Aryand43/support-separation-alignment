from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import sys
from typing import List

import pandas as pd

# Allow running as `python experiments/run_experiment.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.metrics import (
    estimate_harmful_mass,
    estimate_tv_distance,
    query_complexity,
    timed_call,
)
from experiments.plotting import generate_plots
from filters.bounded_filter import BoundedBlackBoxFilter, FilterSpec
from filters.trapdoor_filter import TrapdoorFilter
from models.base_generator import ToyBaseGenerator, ToyGeneratorSpec
from models.rlhf_alignment import AlignmentSpec, RLHFAlignedModel
from utils.config import ExperimentConfig, ensure_output_dirs, set_global_seed


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Support-separation alignment experiment runner")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-prompts", type=int, default=32)
    p.add_argument("--samples-per-prompt", type=int, default=512)
    p.add_argument("--tv-samples", type=int, default=4096)
    p.add_argument("--beta-values", type=str, default="0.0,0.5,1.0,2.0,4.0")
    p.add_argument("--filter-budgets", type=str, default="1,2,4,8,16,32")
    p.add_argument("--filter-capacities", type=str, default="4,8,16,32")
    p.add_argument("--output-dir", type=str, default="outputs")
    return p


def run(cfg: ExperimentConfig) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    set_global_seed(cfg.seed)
    logging.info("Config: %s", asdict(cfg))

    generator = ToyBaseGenerator(
        ToyGeneratorSpec(
            challenge_bits=cfg.challenge_bits,
            noise_bits=cfg.noise_bits,
            hidden_dim=cfg.prf_hidden_dim,
            temperature=cfg.generator_temperature,
            harmful_bias=cfg.harmful_bias,
            nonharmful_bias=cfg.nonharmful_bias,
            seed=cfg.seed,
        )
    )
    generator.train(0)

    prompts = list(range(cfg.eval_prompts))
    rows = []

    # Base model baseline
    base_harm = estimate_harmful_mass(
        generator,
        prompts,
        generator.is_harmful_with_trapdoor,
        cfg.samples_per_prompt,
    )
    rows.append(
        {
            "condition": "base_model",
            "beta": 0.0,
            "filter_budget": None,
            "filter_capacity": None,
            "harmful_mass": base_harm,
            "tv_to_ideal": None,
            "runtime_sec": 0.0,
            "avg_queries_per_output": None,
        }
    )
    logging.info("Base harmful mass: %.6f", base_harm)

    for beta in cfg.beta_values:
        aligned = RLHFAlignedModel(
            base_model=generator,
            spec=AlignmentSpec(beta=beta),
            seed=cfg.seed + int(beta * 1000),
        )
        trapdoor = TrapdoorFilter(base_generator=generator, model=aligned, seed=cfg.seed + 101)

        aligned_harm = estimate_harmful_mass(
            aligned,
            prompts,
            generator.is_harmful_with_trapdoor,
            cfg.samples_per_prompt,
        )
        rows.append(
            {
                "condition": "rlhf_aligned",
                "beta": beta,
                "filter_budget": None,
                "filter_capacity": None,
                "harmful_mass": aligned_harm,
                "tv_to_ideal": estimate_tv_distance(aligned, trapdoor, prompts, cfg.tv_samples),
                "runtime_sec": 0.0,
                "avg_queries_per_output": None,
            }
        )
        logging.info("Beta=%.2f aligned harmful mass: %.6f", beta, aligned_harm)

        trap_harm = estimate_harmful_mass(
            trapdoor,
            prompts,
            generator.is_harmful_with_trapdoor,
            cfg.samples_per_prompt,
        )
        rows.append(
            {
                "condition": "trapdoor_oracle",
                "beta": beta,
                "filter_budget": None,
                "filter_capacity": None,
                "harmful_mass": trap_harm,
                "tv_to_ideal": 0.0,
                "runtime_sec": 0.0,
                "avg_queries_per_output": None,
            }
        )

        for budget in cfg.filter_budgets:
            for capacity in cfg.filter_capacities:
                filt = BoundedBlackBoxFilter(
                    model=aligned,
                    spec=FilterSpec(
                        max_queries=budget,
                        capacity=capacity,
                        accept_threshold=0.45,
                        bootstrap_samples=max(128, 8 * capacity),
                    ),
                    seed=cfg.seed + budget + capacity,
                )
                filt.reset_stats()

                elapsed_harm, harm = timed_call(
                    estimate_harmful_mass,
                    filt,
                    prompts,
                    generator.is_harmful_with_trapdoor,
                    cfg.samples_per_prompt,
                )
                elapsed_tv, tv = timed_call(
                    estimate_tv_distance,
                    filt,
                    trapdoor,
                    prompts,
                    cfg.tv_samples,
                )
                runtime = elapsed_harm + elapsed_tv
                produced = cfg.eval_prompts * (cfg.samples_per_prompt + cfg.tv_samples)
                avg_q = query_complexity(filt, produced)

                rows.append(
                    {
                        "condition": "bounded_filter",
                        "beta": beta,
                        "filter_budget": budget,
                        "filter_capacity": capacity,
                        "harmful_mass": harm,
                        "tv_to_ideal": tv,
                        "runtime_sec": runtime,
                        "avg_queries_per_output": avg_q,
                    }
                )
                logging.info(
                    "beta=%.2f budget=%d cap=%d -> harmful=%.6f tv=%.6f runtime=%.3fs q/sample=%.2f",
                    beta,
                    budget,
                    capacity,
                    harm,
                    tv,
                    runtime,
                    avg_q,
                )

    df = pd.DataFrame(rows)
    Path(cfg.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.metrics_csv, index=False)
    generate_plots(df, cfg.plots_dir)
    return df


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cfg = ExperimentConfig(
        seed=args.seed,
        eval_prompts=args.eval_prompts,
        samples_per_prompt=args.samples_per_prompt,
        tv_samples=args.tv_samples,
        beta_values=_parse_float_list(args.beta_values),
        filter_budgets=_parse_int_list(args.filter_budgets),
        filter_capacities=_parse_int_list(args.filter_capacities),
        output_dir=args.output_dir,
        plots_dir=f"{args.output_dir}/plots",
        metrics_csv=f"{args.output_dir}/metrics.csv",
    )
    df = run(cfg)
    print(df.groupby("condition")["harmful_mass"].mean().sort_values(ascending=False))
    print(f"Wrote metrics to: {cfg.metrics_csv}")
    print(f"Wrote plots to:   {cfg.plots_dir}")


if __name__ == "__main__":
    main()
