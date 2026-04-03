# Support-Preserving Alignment and Residual Risk Floors

This repository studies residual harmful mass under aligned generation and bounded filtering, evaluated on real LLMs via OpenRouter.

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
│   └── langdb_wrapper.py          # OpenRouter API wrapper
├── filters/
│   ├── bounded_filter.py           # Black-box bounded filter
│   ├── whitebox_filter.py          # Logprob-aware white-box filter
│   └── statistical_query_filter.py # SQ-style filter
├── experiments/
│   ├── run_llm_eval.py             # Main evaluation script
│   ├── metrics.py                  # Harm classification
│   └── plotting.py                 # Result visualisation
├── real_model_extension/
│   ├── run_real_model_eval.py
│   └── README.md
├── config/
│   └── models.yaml                 # OpenRouter model list
└── outputs/
    └── llm_results/
        └── llm_metrics.csv
```

## Theory Modules

- `theory/support_persistence.tex`: universal support-persistence theorem in measurable spaces.
- `theory/reductions/prf_reduction.md`: computational lower bound via hidden-feature hardness.
- `theory/info_theoretic_lower_bound.md`: SQ/VC-style residual floor without cryptographic assumptions.
- `theory/positioning.md`: novelty and relation to adjacent impossibility literature.
- `theory/transformer_interpretation.md`: mapping to transformer-era alignment updates.

## Experimental Pipeline

`experiments/run_llm_eval.py` wraps real LLMs (served via OpenRouter) with three bounded filters and measures the residual harmful rate as filter budget increases:

- **bounded**: black-box sampling filter (`filters/bounded_filter.py`)
- **whitebox**: logit/probability-aware filter (`filters/whitebox_filter.py`)
- **sq**: statistical-query style filter (`filters/statistical_query_filter.py`)

Evaluation prompts are drawn from curated adversarial prompts and, optionally, the HuggingFace `PKU-Alignment/PKU-SafeRLHF` dataset.

## Run

```bash
pip install numpy pandas matplotlib seaborn openai pyyaml tqdm datasets
```

Set API keys:

```bash
export OPENROUTER_API_KEY=<your-openrouter-key>
export HF_TOKEN=<your-hf-token>          # optional, for HF prompts
```

Run the evaluation:

```bash
python experiments/run_llm_eval.py \
    --models-config config/models.yaml \
    --filter-budgets "1,4,16,64" \
    --max-prompts 50 \
    --output-dir outputs/llm_results
```

Generate plots:

```bash
python experiments/plotting.py --metrics outputs/llm_results/llm_metrics.csv
```

## Outputs

- `outputs/llm_results/llm_metrics.csv`: per-prompt results for every (model, filter, budget) triple.
- `outputs/llm_results/plots/harm_vs_budget_<model>.png`: harmful rate vs filter budget per model.
- `outputs/llm_results/plots/harm_vs_budget_all_models.png`: combined overlay.
- `outputs/llm_results/plots/harm_floor_by_model.png`: minimum harmful rate across all filters/budgets.
