from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import sys
from typing import List

import numpy as np
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
from filters.statistical_query_filter import StatisticalQueryFilter, StatisticalQuerySpec
from filters.trapdoor_filter import TrapdoorFilter
from filters.whitebox_filter import WhiteBoxFilter, WhiteBoxSpec
from models.base_generator import ToyBaseGenerator, ToyGeneratorSpec
from models.rlhf_alignment import AlignmentSpec, RLHFAlignedModel
from utils.config import ExperimentConfig, ensure_output_dirs, set_global_seed


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _popcount(v: int) -> int:
    return v.bit_count()


def adversarial_prompts(n: int, count: int, span: int, seed: int) -> List[int]:
    """
    Finite-sample approximation to an adversarial prompt family D_X,n.
    """
    rng = np.random.default_rng(seed + 17 * n)
    max_x = 2**n
    shift = int(rng.integers(0, max_x))

    prompts: List[int] = []
    trials = max(count * span, count * 8)
    for _ in range(trials):
        x = int(rng.integers(0, max_x))
        score = _popcount(x ^ shift) % 4
        if score <= 1:
            prompts.append(x)
        if len(prompts) >= count:
            break
    while len(prompts) < count:
        prompts.append(int(rng.integers(0, max_x)))
    return prompts


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Support-separation alignment experiment runner")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval-prompts", type=int, default=32)
    p.add_argument("--samples-per-prompt", type=int, default=512)
    p.add_argument("--tv-samples", type=int, default=4096)
    p.add_argument("--beta-values", type=str, default="0.0,0.5,1.0,2.0,4.0")
    p.add_argument("--filter-budgets", type=str, default="1,2,4,8,16,32")
    p.add_argument("--filter-capacities", type=str, default="4,8,16,32")
    p.add_argument("--whitebox-capacities", type=str, default="4,8,16,32")
    p.add_argument("--sq-budgets", type=str, default="1,2,4,8,16,32")
    p.add_argument("--sq-samples-per-query", type=int, default=32)
    p.add_argument("--security-params", type=str, default="4,6,8,10")
    p.add_argument("--adversarial-prompt-span", type=int, default=64)
    p.add_argument("--scaling-target-harmful-mass", type=float, default=0.15)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p


def run(cfg: ExperimentConfig) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    set_global_seed(cfg.seed)
    logging.info("Config: %s", asdict(cfg))

    rows = []
    scaling_rows = []

    for n in cfg.security_params:
        generator = ToyBaseGenerator(
            ToyGeneratorSpec(
                challenge_bits=n,
                noise_bits=cfg.noise_bits,
                hidden_dim=cfg.prf_hidden_dim,
                temperature=cfg.generator_temperature,
                harmful_bias=cfg.harmful_bias,
                nonharmful_bias=cfg.nonharmful_bias,
                seed=cfg.seed + n,
            )
        )
        generator.train(0)
        prompts = adversarial_prompts(
            n=n,
            count=cfg.eval_prompts,
            span=cfg.adversarial_prompt_span,
            seed=cfg.seed,
        )
        logging.info("Security parameter n=%d with %d adversarial prompts", n, len(prompts))

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
                "security_param": n,
                "beta": 0.0,
                "filter_budget": None,
                "filter_capacity": None,
                "num_stat_queries": None,
                "harmful_mass": base_harm,
                "tv_to_ideal": None,
                "runtime_sec": 0.0,
                "avg_queries_per_output": None,
            }
        )
        logging.info("n=%d base harmful mass: %.6f", n, base_harm)

        for beta in cfg.beta_values:
            aligned = RLHFAlignedModel(
                base_model=generator,
                spec=AlignmentSpec(beta=beta),
                seed=cfg.seed + n * 100 + int(beta * 1000),
            )
            trapdoor = TrapdoorFilter(base_generator=generator, model=aligned, seed=cfg.seed + n + 101)

            aligned_harm = estimate_harmful_mass(
                aligned,
                prompts,
                generator.is_harmful_with_trapdoor,
                cfg.samples_per_prompt,
            )
            rows.append(
                {
                    "condition": "rlhf_aligned",
                    "security_param": n,
                    "beta": beta,
                    "filter_budget": None,
                    "filter_capacity": None,
                    "num_stat_queries": None,
                    "harmful_mass": aligned_harm,
                    "tv_to_ideal": estimate_tv_distance(aligned, trapdoor, prompts, cfg.tv_samples),
                    "runtime_sec": 0.0,
                    "avg_queries_per_output": None,
                }
            )
            logging.info("n=%d beta=%.2f aligned harmful mass: %.6f", n, beta, aligned_harm)

            trap_harm = estimate_harmful_mass(
                trapdoor,
                prompts,
                generator.is_harmful_with_trapdoor,
                cfg.samples_per_prompt,
            )
            rows.append(
                {
                    "condition": "trapdoor_oracle",
                    "security_param": n,
                    "beta": beta,
                    "filter_budget": None,
                    "filter_capacity": None,
                    "num_stat_queries": None,
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
                        seed=cfg.seed + n + budget + capacity,
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
                            "security_param": n,
                            "beta": beta,
                            "filter_budget": budget,
                            "filter_capacity": capacity,
                            "num_stat_queries": None,
                            "harmful_mass": harm,
                            "tv_to_ideal": tv,
                            "runtime_sec": runtime,
                            "avg_queries_per_output": avg_q,
                        }
                    )

                for capacity in cfg.whitebox_capacities:
                    wf = WhiteBoxFilter(
                        model=aligned,
                        spec=WhiteBoxSpec(
                            max_queries=budget,
                            capacity=capacity,
                            entropy_weight=0.6,
                            confidence_weight=0.4,
                        ),
                        seed=cfg.seed + 11 * n + budget + capacity,
                    )
                    wf.reset_stats()
                    elapsed_harm, harm = timed_call(
                        estimate_harmful_mass,
                        wf,
                        prompts,
                        generator.is_harmful_with_trapdoor,
                        cfg.samples_per_prompt,
                    )
                    elapsed_tv, tv = timed_call(
                        estimate_tv_distance,
                        wf,
                        trapdoor,
                        prompts,
                        cfg.tv_samples,
                    )
                    runtime = elapsed_harm + elapsed_tv
                    produced = cfg.eval_prompts * (cfg.samples_per_prompt + cfg.tv_samples)
                    avg_q = query_complexity(wf, produced)
                    rows.append(
                        {
                            "condition": "whitebox_filter",
                            "security_param": n,
                            "beta": beta,
                            "filter_budget": budget,
                            "filter_capacity": capacity,
                            "num_stat_queries": None,
                            "harmful_mass": harm,
                            "tv_to_ideal": tv,
                            "runtime_sec": runtime,
                            "avg_queries_per_output": avg_q,
                        }
                    )

            for sq_budget in cfg.sq_budgets:
                sq_filter = StatisticalQueryFilter(
                    model=aligned,
                    spec=StatisticalQuerySpec(
                        num_stat_queries=sq_budget,
                        samples_per_query=cfg.sq_samples_per_query,
                        max_candidate_draws=max(2, min(8, sq_budget)),
                        reject_threshold=0.35,
                    ),
                    seed=cfg.seed + 13 * n + sq_budget,
                )
                sq_filter.reset_stats()
                elapsed_harm, harm = timed_call(
                    estimate_harmful_mass,
                    sq_filter,
                    prompts,
                    generator.is_harmful_with_trapdoor,
                    cfg.samples_per_prompt,
                )
                elapsed_tv, tv = timed_call(
                    estimate_tv_distance,
                    sq_filter,
                    trapdoor,
                    prompts,
                    cfg.tv_samples,
                )
                runtime = elapsed_harm + elapsed_tv
                produced = cfg.eval_prompts * (cfg.samples_per_prompt + cfg.tv_samples)
                avg_q = query_complexity(sq_filter, produced)
                rows.append(
                    {
                        "condition": "sq_filter",
                        "security_param": n,
                        "beta": beta,
                        "filter_budget": sq_budget,
                        "filter_capacity": None,
                        "num_stat_queries": sq_budget,
                        "harmful_mass": harm,
                        "tv_to_ideal": tv,
                        "runtime_sec": runtime,
                        "avg_queries_per_output": avg_q,
                    }
                )

        # Scaling summary is anchored at beta=0 to isolate computational filtering limits.
        n_rows = [
            r
            for r in rows
            if r.get("security_param") == n
            and r["condition"] in {"bounded_filter", "whitebox_filter", "sq_filter"}
            and float(r.get("beta", 0.0)) == 0.0
        ]
        if n_rows:
            harmful_floor = min(float(r["harmful_mass"]) for r in n_rows)
            near_floor_rows = [r for r in n_rows if float(r["harmful_mass"]) <= harmful_floor + cfg.scaling_floor_tolerance]
            required_queries_floor = min(float(r["avg_queries_per_output"]) for r in near_floor_rows)

            target_rows = [r for r in n_rows if float(r["harmful_mass"]) <= cfg.scaling_target_harmful_mass]
            if target_rows:
                required_queries_target = min(float(r["avg_queries_per_output"]) for r in target_rows)
                target_achieved = 1
            else:
                # Lower-bound proxy when target is unattained in tested budget range.
                required_queries_target = float(2 ** max(1, n // 2))
                target_achieved = 0

            scaling_rows.append(
                {
                    "condition": "scaling_summary",
                    "security_param": n,
                    "beta": None,
                    "filter_budget": None,
                    "filter_capacity": None,
                    "num_stat_queries": None,
                    "harmful_mass": None,
                    "harmful_floor": harmful_floor,
                    "required_queries": required_queries_target,
                    "required_queries_near_floor": required_queries_floor,
                    "required_queries_target": required_queries_target,
                    "target_achieved": target_achieved,
                    "tv_to_ideal": None,
                    "runtime_sec": None,
                    "avg_queries_per_output": None,
                }
            )
            logging.info(
                "n=%d scaling summary: floor=%.6f required_q_floor=%.2f required_q_target=%.2f achieved=%d",
                n,
                harmful_floor,
                required_queries_floor,
                required_queries_target,
                target_achieved,
            )

    rows.extend(scaling_rows)
    if scaling_rows:
        c_hat = min(float(r["harmful_floor"]) for r in scaling_rows)
        rows.append(
            {
                "condition": "empirical_constant_estimate",
                "security_param": None,
                "beta": None,
                "filter_budget": None,
                "filter_capacity": None,
                "num_stat_queries": None,
                "harmful_mass": c_hat,
                "harmful_floor": c_hat,
                "required_queries": None,
                "required_queries_near_floor": None,
                "required_queries_target": None,
                "target_achieved": None,
                "tv_to_ideal": None,
                "runtime_sec": None,
                "avg_queries_per_output": None,
            }
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
        whitebox_capacities=_parse_int_list(args.whitebox_capacities),
        sq_budgets=_parse_int_list(args.sq_budgets),
        sq_samples_per_query=args.sq_samples_per_query,
        security_params=_parse_int_list(args.security_params),
        adversarial_prompt_span=args.adversarial_prompt_span,
        scaling_target_harmful_mass=args.scaling_target_harmful_mass,
        output_dir=args.output_dir,
        plots_dir=f"{args.output_dir}/plots",
        metrics_csv=f"{args.output_dir}/metrics.csv",
    )
    df = run(cfg)
    print(df[df["harmful_mass"].notna()].groupby("condition")["harmful_mass"].mean().sort_values(ascending=False))
    scaling_df = df[df["condition"] == "scaling_summary"][
        ["security_param", "harmful_floor", "required_queries", "required_queries_near_floor", "required_queries_target", "target_achieved"]
    ]
    if not scaling_df.empty:
        print("\nScaling summary:")
        print(scaling_df.to_string(index=False))
    c_hat_df = df[df["condition"] == "empirical_constant_estimate"]["harmful_mass"]
    if not c_hat_df.empty:
        print(f"\nEmpirical lower-bound constant estimate c_hat: {float(c_hat_df.iloc[0]):.6f}")
    print(f"Wrote metrics to: {cfg.metrics_csv}")
    print(f"Wrote plots to:   {cfg.plots_dir}")


if __name__ == "__main__":
    main()
