# Support-Preserving Alignment and Residual Risk Floors

This repository studies residual harmful mass under aligned generation and bounded filtering, evaluated on real LLMs via OpenRouter.

## Positioning

We show:

- Any support-preserving stochastic alignment operator cannot eliminate harmful mass that already has positive support.
- Under bounded filtering, residual harmful mass floors arise.
- These floors can be derived in both computational and information-theoretic settings.

The goal is **not** to claim alignment is impossible. The claim is that support-preserving alignment plus bounded filtering can leave nonzero residual risk.

## Framework

An alignment operator $\mathcal{A}$ is **support-preserving** if:

$$P_\theta(y \mid x) > 0 \;\Rightarrow\; (\mathcal{A}\, P_\theta)(y \mid x) > 0.$$

This covers RLHF, PPO-style policy reweighting, DPO-like preference reweighting, and generic reward tilting:

$$\widetilde{P}(y \mid x) \propto P_\theta(y \mid x) \exp\bigl(\beta\, r(x,y)\bigr).$$

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
│   └── langdb_wrapper.py              # OpenRouter API wrapper
├── filters/
│   ├── bounded_filter.py               # Black-box bounded filter
│   ├── whitebox_filter.py              # Logprob-aware white-box filter
│   └── statistical_query_filter.py     # SQ-style filter
├── experiments/
│   ├── run_llm_eval.py                 # Main evaluation script
│   ├── metrics.py                      # Harm classification
│   └── plotting.py                     # Result visualisation
├── real_model_extension/
│   ├── run_real_model_eval.py
│   └── README.md
├── config/
│   └── models.yaml                     # OpenRouter model list
└── outputs/
    └── llm_results/
        └── llm_metrics.csv
```

## Theory Modules

| Module | Description |
|--------|-------------|
| `theory/support_persistence.tex` | Universal support-persistence theorem in measurable spaces |
| `theory/reductions/prf_reduction.md` | Computational lower bound via hidden-feature hardness |
| `theory/info_theoretic_lower_bound.md` | SQ/VC-style residual floor without cryptographic assumptions |
| `theory/positioning.md` | Novelty and relation to adjacent impossibility literature |
| `theory/transformer_interpretation.md` | Mapping to transformer-era alignment updates |

## Experimental Pipeline

`experiments/run_llm_eval.py` wraps real LLMs (served via OpenRouter) with three bounded filters and measures the residual harmful rate as filter budget increases:

| Filter | Module | Description |
|--------|--------|-------------|
| `bounded` | `filters/bounded_filter.py` | Black-box sampling filter |
| `whitebox` | `filters/whitebox_filter.py` | Logit/probability-aware filter |
| `sq` | `filters/statistical_query_filter.py` | Statistical-query style filter |

Evaluation prompts are drawn from curated adversarial prompts and, optionally, the HuggingFace [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) dataset.

## Run

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn openai pyyaml tqdm datasets
```

### Set API keys

```bash
export OPENROUTER_API_KEY=<your-openrouter-key>
export HF_TOKEN=<your-hf-token>          # optional, for HF prompts
```

### Run the evaluation

```bash
python experiments/run_llm_eval.py \
    --models-config config/models.yaml \
    --filter-budgets "1,4,16,64" \
    --max-prompts 50 \
    --output-dir outputs/llm_results
```

### Generate plots

```bash
python experiments/plotting.py --metrics outputs/llm_results/llm_metrics.csv
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/llm_results/llm_metrics.csv` | Per-prompt results for every (model, filter, budget) triple |
| `outputs/llm_results/plots/harm_vs_budget_<model>.png` | Harmful rate vs. filter budget per model |
| `outputs/llm_results/plots/harm_vs_budget_all_models.png` | Combined overlay across all models |
| `outputs/llm_results/plots/harm_floor_by_model.png` | Minimum harmful rate across all filters/budgets |

## Paper

The accompanying paper is available in `belief_propagation_alignment_limits.tex`. To build:

```bash
pdflatex belief_propagation_alignment_limits
bibtex belief_propagation_alignment_limits
pdflatex belief_propagation_alignment_limits
pdflatex belief_propagation_alignment_limits
```

## Citation

```bibtex
@article{dutt2026limits,
  title   = {On the Limits of Support-Preserving Alignment and Bounded Filtering},
  author  = {Dutt, Aryan and Mao, Rui and Chattopadhyay, Anupam},
  year    = {2026},
  note    = {Nanyang Technological University}
}
```
