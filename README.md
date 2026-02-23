# support-separation-alignment

Formal and empirical study of support separation between:

- support-preserving alignment operators,
- bounded black-box and white-box (no-trapdoor) filters,
- trapdoor ideal elimination.

This repository is structured for paper-grade claims: explicit assumptions, security parameter \(n\), formal reductions, complexity model, and security-scaling experiments.

## Architecture

```
support-separation-alignment/
├── theory/
│   ├── definitions.py
│   ├── theorems.md
│   ├── proof_sketches.md
│   ├── support_persistence.tex
│   ├── complexity_model.md
│   ├── adversarial_distribution.md
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
│   └── trapdoor_filter.py
├── experiments/
│   ├── run_experiment.py
│   ├── metrics.py
│   └── plotting.py
├── real_model_extension/
│   ├── run_real_model_eval.py
│   └── README.md
├── utils/
│   ├── data_generation.py
│   └── config.py
└── README.md
```

## Main Contributions

- Universal support-persistence theorem for absolutely continuous alignments in measurable spaces.
- Explicit PRF-hardness reduction linking low-TV filtering to PRF distinguishing.
- Formal PPT oracle complexity model for black-box and white-box filters.
- Adversarial prompt distribution \(D_{X,n}\) used by both reduction and experiments.
- Security-scaling experiment over \(n \in \{4,6,8,10\}\), including harmful-floor and query requirements.
- White-box vs black-box comparison showing trapdoor remains the key separator.

## Theory-to-Code Mapping

- `theory/support_persistence.tex`: universal theorem statement + proof.
- `theory/reductions/prf_reduction.md`: explicit hardness-to-TV reduction.
- `theory/complexity_model.md`: PPT filter class and oracle interface.
- `theory/adversarial_distribution.md`: adversarial prompt family \(D_{X,n}\).
- `experiments/run_experiment.py`: empirical analog implementing \(D_{X,n}\), scaling, and filter comparisons.

## Experimental Design

For each security parameter \(n\):

1. Build a keyed toy generator \(P_{\theta,k}^{(n)}\) over \(y=(c,r,b)\) with trapdoor harmful predicate.
2. Apply support-preserving alignment \(A_\beta\).
3. Compare:
   - `bounded_filter` (black-box, query-limited),
   - `whitebox_filter` (logit/probability access, no trapdoor),
   - `trapdoor_oracle` (ideal eliminator baseline).
4. Estimate:
   - residual harmful mass,
   - TV distance to ideal eliminator,
   - query/runtime complexity,
   - security-scaling summary (`harmful_floor`, required queries).

## Run

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

Main experiment:

```bash
python experiments/run_experiment.py
```

Example with explicit scaling sweep:

```bash
python experiments/run_experiment.py \
  --security-params "4,6,8,10" \
  --beta-values "0.0,0.5,1.0,2.0,4.0" \
  --filter-budgets "1,2,4,8,16,32" \
  --filter-capacities "4,8,16,32" \
  --whitebox-capacities "4,8,16,32" \
  --eval-prompts 32 \
  --samples-per-prompt 512 \
  --tv-samples 2048 \
  --seed 7
```

## Outputs

- `outputs/metrics.csv`
- `outputs/plots/harmful_mass_vs_budget.png`
- `outputs/plots/harmful_mass_vs_capacity.png`
- `outputs/plots/tv_vs_runtime.png`
- `outputs/plots/beta_vs_residual_harmful_mass.png`
- `outputs/plots/scaling_vs_security_param.png`

`metrics.csv` includes `condition=scaling_summary` rows with:

- `harmful_floor` (minimum achieved harmful mass under bounded/white-box filters),
- `required_queries` (queries per output needed to reach near-floor),
- `required_queries_target` (target-risk query proxy; lower bound if unattained).

It also includes `condition=empirical_constant_estimate` with \(\hat c\), the empirical lower-bound constant estimate for residual harmful mass over tested security parameters.

## Optional Real-Model Bridge

See `real_model_extension/README.md` for a lightweight extension using `transformers` to compare base vs safety-prefix vs post-hoc filter behavior on a mini jailbreak prompt list.
