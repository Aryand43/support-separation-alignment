# Support-Preserving Alignment and Residual Risk Floors

This repository studies residual harmful mass under aligned generation and bounded filtering.

## Positioning

We show:

1. Any support-preserving stochastic alignment operator cannot eliminate harmful mass that already has positive support.
2. Under bounded filtering, residual harmful mass floors arise.
3. These floors can be derived in both computational and information-theoretic settings.

The goal is not to claim alignment is impossible. The claim is that support-preserving alignment plus bounded filtering can leave nonzero residual risk.

## Framework

Alignment operator \(A\) is support-preserving if:
\[
P_\theta(y|x) > 0 \Rightarrow (A P_\theta)(y|x) > 0.
\]

This covers RLHF, PPO-style policy reweighting, DPO-like preference reweighting, and generic reward tilting:
\[
\widetilde P(y|x)\propto P_\theta(y|x)\exp(\beta r(x,y)).
\]

## Repository Structure

```
support-separation-alignment/
├── theory/
│   ├── definitions.py
│   ├── theorems.md
│   ├── proof_sketches.md
│   ├── support_persistence.tex
│   ├── info_theoretic_lower_bound.md
│   ├── complexity_model.md
│   ├── adversarial_distribution.md
│   ├── transformer_interpretation.md
│   ├── positioning.md
│   ├── appendix_proofs.md
│   └── reductions/
│       └── prf_reduction.md
├── models/
│   ├── base_generator.py
│   ├── rlhf_alignment.py
│   └── ideal_eliminator.py
├── filters/
│   ├── bounded_filter.py
│   ├── whitebox_filter.py
│   ├── statistical_query_filter.py
│   └── trapdoor_filter.py
├── experiments/
│   ├── run_experiment.py
│   ├── metrics.py
│   └── plotting.py
├── real_model_extension/
│   ├── run_real_model_eval.py
│   └── README.md
└── utils/
    ├── data_generation.py
    └── config.py
```

## Theory Modules

- `theory/support_persistence.tex`: universal support-persistence theorem in measurable spaces.
- `theory/reductions/prf_reduction.md`: computational lower bound via hidden-feature hardness (PRF instantiation).
- `theory/info_theoretic_lower_bound.md`: SQ/VC-style residual floor without cryptographic assumptions.
- `theory/positioning.md`: novelty and relation to adjacent impossibility literature.
- `theory/transformer_interpretation.md`: mapping to transformer-era alignment updates.

## Experimental Modules

`experiments/run_experiment.py` compares:

- `bounded_filter` (black-box),
- `whitebox_filter` (logit/probability access, no harmful oracle),
- `sq_filter` (statistical-query style),
- `trapdoor_oracle` (ideal reference for gap measurement).

Sweeps include security parameter, filter compute budgets, and alignment strength.

## Run

```bash
pip install numpy pandas matplotlib seaborn
python experiments/run_experiment.py
```

Example:

```bash
python experiments/run_experiment.py \
  --security-params "4,6,8,10" \
  --beta-values "0.0,0.5,1.0,2.0" \
  --filter-budgets "1,2,4,8,16,32" \
  --sq-budgets "1,2,4,8,16,32" \
  --eval-prompts 32 \
  --samples-per-prompt 512 \
  --tv-samples 2048 \
  --seed 7
```

## Outputs

- `outputs/metrics.csv`
- `outputs/plots/harmful_mass_vs_budget.png`
- `outputs/plots/harmful_mass_vs_capacity.png`
- `outputs/plots/harmful_mass_vs_num_statistical_queries.png`
- `outputs/plots/tv_vs_runtime.png`
- `outputs/plots/beta_vs_residual_harmful_mass.png`
- `outputs/plots/scaling_vs_security_param.png`

`metrics.csv` includes scaling summaries and empirical floor estimates.
