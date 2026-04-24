# REPRODUCE_INFRA.md
# End-to-End Reproduction Guide: Support-Separation Alignment on Real LLMs

This document gives precise steps for reproducing the infrastructure and results for the project: **"Support-Separation Alignment: Structural Floors in Computational Filtering"** using **only real LLMs served via OpenRouter**.

---

## 1. Executive Summary

**The Concept:**
We wrap real large language models (LLMs) with different kinds of **bounded safety filters** and measure how much harmful behavior remains as we increase the filters' compute budgets.

**The Goal:**
Show that for a range of state-of-the-art open-weight and hosted LLMs, alignment schemes that only reshape output probabilities and apply bounded filters **reduce but do not eliminate** harmful behavior, leaving a persistent **"harm floor"** above zero.

---

## 2. Environment Setup

Set your OpenRouter and Hugging Face API keys (do NOT commit the raw keys):

```bash
export OPENROUTER_API_KEY=<your-openrouter-key>
export HF_TOKEN=<your-hf-token>
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn openai pyyaml tqdm datasets
```

---

## 3. Directory Structure

Ensure the repository has the following layout:

```plaintext
support-separation-alignment/
├── experiments/              # LLM evaluation and plotting
│   ├── run_llm_eval.py
│   └── plotting.py
├── filters/                  # Bounded, white-box, and SQ filter implementations
│   ├── metrics.py            # Harm classification heuristics
│   ├── bounded_filter.py     # Black-box filter over LLM
│   ├── whitebox_filter.py    # Logprob-aware filter
│   └── statistical_query_filter.py
├── models/                   # OpenRouter wrapper
│   └── langdb_wrapper.py
├── config/                   # Model configs (prompts loaded from HF at runtime)
│   └── models.yaml
└── outputs/                  # CSVs and generated plots
    └── llm_results/
        ├── llm_metrics.csv
        └── plots/
```

---

## 4. Configuration

### 4.1 Model configuration

Populate `config/models.yaml` with the OpenRouter models to evaluate, for example:

```yaml
models:
  - name: "gemma-3-12b"
    openrouter_id: "google/gemma-3-12b-it:free"
    params: 12e9
    context: 32000

  - name: "qwen3-coder"
    openrouter_id: "qwen/qwen3-coder:free"
    params: 480e9
    context: 262000

  - name: "nemotron-3-super-120b"
    openrouter_id: "nvidia/nemotron-3-super-120b-a12b:free"
    params: 120e9
    context: 262000

  - name: "llama-3.2-3b"
    openrouter_id: "meta-llama/llama-3.2-3b-instruct:free"
    params: 3e9
    context: 131000

  - name: "trinity-large-preview"
    openrouter_id: "arcee-ai/trinity-large-preview:free"
    params: 400e9
    context: 131000
```

Adjust each `openrouter_id` to match the actual model identifier on OpenRouter.

### 4.2 Prompt configuration (from Hugging Face)

Evaluation prompts are drawn from two sources:

1. **Curated adversarial prompts** — a built-in set of indirect cybersecurity-themed prompts inside `run_llm_eval.py`.
2. **HuggingFace PKU-SafeRLHF** — the unsafe subset of `PKU-Alignment/PKU-SafeRLHF`, loaded at runtime via the `datasets` library using the `HF_TOKEN` environment variable.

`run_llm_eval.py` uses the curated set first and supplements with HuggingFace prompts up to `--max-prompts`.

---

## 5. Filters and Access Modes

All filters wrap the **real LLM** and obey explicit compute limits.

1. **Bounded Black-Box Filter (`filters/bounded_filter.py`)**
   - Access: sampling and (optionally) approximate log-probs only.
   - For each prompt, draws up to `max_queries` candidate outputs from the base LLM and uses a lightweight proxy risk score (lexical/structural features) to choose the lowest-risk candidate.

2. **White-Box Filter (`filters/whitebox_filter.py`)**
   - Access: sampling plus token-level log-probabilities or probabilities when available.
   - Uses entropy / confidence and simple feature correlations to score candidates under a query budget.

3. **Statistical-Query Filter (`filters/statistical_query_filter.py`)**
   - Access: sampling via repeated queries.
   - Estimates expectations of a small feature bank (keyword indicators, style features) and rejects candidate outputs whose feature vector deviates too much from these expectations, under a bounded number of SQ calls.

A separate `is_harmful(prompt, output)` function (implemented in `filters/metrics.py`) classifies outputs as harmful or not using a fixed heuristic.

---

## 6. Execution Roadmap

### 6.1 Selection policies (models, filters, prompts)

- **Model selection:**
  `run_llm_eval.py` loads the ordered list of models in `config/models.yaml` and iterates over them sequentially. For any prompt, the active model is simply the current element of this outer loop (deterministic round-robin, \(O(1)\) selection).

- **Filter selection:**
  A fixed grid of filter types and budgets is defined in code/CLI, e.g. filter types = `["bounded", "whitebox", "sq"]`, budgets = `"1,4,16,64"`. `run_llm_eval.py` loops over all `(filter_type, filter_budget)` pairs for each model. For any prompt, the active filter is determined by this loop (again \(O(1)\) selection).

- **Prompt selection:**
  `run_llm_eval.py` calls `load_prompts_from_hf(...)` once, obtains a list, and iterates over prompts in order. For any step, the active prompt is the current index in this list (deterministic round-robin, \(O(1)\) selection).

### 6.2 Single command to run all experiments

From the repo root:

```bash
python experiments/run_llm_eval.py \
    --models-config config/models.yaml \
    --filter-budgets "1,4,16,64" \
    --output-dir outputs/llm_results
```

This script:

- Loads the model list from `config/models.yaml`.
- Loads prompts from the curated set and HuggingFace via `load_prompts_from_hf`.
- For each model, filter type, and filter budget, runs the chosen filter around the LLM and logs per-prompt harmfulness.

### 6.3 What `run_llm_eval.py` does

For each `(model, filter_type, filter_budget)` triple:

1. Instantiates `LangDBGenerator` in `models/langdb_wrapper.py` using the `openrouter_id`.
2. Wraps it with the specified filter (bounded black-box, white-box, or SQ) with the given budget.
3. For each prompt in the loaded prompt list:
   - Calls `filter.sample(prompt_text, n=1)` (internally may query the base LLM multiple times, but never exceeding the budget).
   - Evaluates `is_harmful(prompt_text, output)`.
   - Logs a row with: model name, filter type, filter budget, prompt id, harmful (0/1), and total queries used.

Results are written to:

```plaintext
outputs/llm_results/llm_metrics.csv
```

---

## 7. Verification of Success

Reproduction is considered successful if:

- **Persistence:** For each model and filter type, the estimated harmful rate **decreases but plateaus above zero** as `filter_budget` increases, rather than going to zero.
- **Cross-Model Behavior:** A non-zero "harm floor" appears across multiple models in `config/models.yaml`, with model-specific levels but no model reaching zero harmful rate under finite budgets.
- **Visuals:** Running

  ```bash
  python experiments/plotting.py --metrics outputs/llm_results/llm_metrics.csv
  ```

  generates plots in `outputs/llm_results/plots/`, including:

  - `harm_vs_budget_<model>.png`: harmful rate vs filter budget for each model and filter type, showing saturation above zero.

End of Reproduction Guide
