from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import random

import numpy as np


@dataclass
class ExperimentConfig:
    seed: int = 7
    challenge_bits: int = 4
    noise_bits: int = 4
    prf_hidden_dim: int = 32
    generator_temperature: float = 1.0
    harmful_bias: float = 1.6
    nonharmful_bias: float = 0.2

    # RLHF-style alignment
    beta_values: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0])

    # Bounded black-box filter
    filter_budgets: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    filter_capacities: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    whitebox_capacities: List[int] = field(default_factory=lambda: [4, 8, 16, 32])

    # Security-scaling study
    security_params: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    scaling_target_harmful_mass: float = 0.15
    scaling_floor_tolerance: float = 0.02
    adversarial_prompt_span: int = 64

    # Evaluation
    eval_prompts: int = 32
    samples_per_prompt: int = 512
    tv_samples: int = 4096

    # IO
    output_dir: str = "outputs"
    plots_dir: str = "outputs/plots"
    metrics_csv: str = "outputs/metrics.csv"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs(cfg: ExperimentConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.plots_dir).mkdir(parents=True, exist_ok=True)
