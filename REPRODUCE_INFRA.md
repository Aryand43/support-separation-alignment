# REPRODUCE_INFRA.md  
# End-to-End Reproduction Guide: Support-Separation Alignment on Real LLMs

This document gives precise steps for reproducing the infrastructure and results for the project: **“Support-Separation Alignment: Structural Floors in Computational Filtering”** using **only real LLMs served via OpenRouter**.[file:2][file:3]

---0.

## 1. Executive Summary (ELI10)

**The Concept:**  
We wrap real large language models (LLMs) with different kinds of **bounded safety filters** and measure how much harmful behavior remains as we increase the filters’ compute budgets.[file:2]

**The Goal:**  
Show that for a range of state-of-the-art open-weight and hosted LLMs, alignment schemes that only reshape output probabilities and apply bounded filters **reduce but do not eliminate** harmful behavior, leaving a persistent **“harm floor”** above zero.[file:2][file:5]

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

Ensure the repository has the following minimal layout:

```plaintext
└── aryand43-support-separation-alignment/
    ├── experiments/              # LLM evaluation, metrics, plotting
    │   ├── run_llm_eval.py
    │   ├── metrics.py
    │   └── plotting.py
    ├── filters/                  # Bounded, white-box, and SQ filter implementations
    │   ├── bounded_filter.py     # Black-box filter over LLM
    │   ├── whitebox_filter.py    # Logprob-aware filter (if available)
    │   └── statistical_query_filter.py
    ├── models/                   # OpenRouter / LangDB wrappers
    │   └── langdb_wrapper.py
    ├── config/                   # Model configs (prompts may be loaded from HF)
    │   └── models.yaml
    └── outputs/                  # CSVs and generated plots
        ├── llm_metrics.csv
        └── plots/
```

All toy generator and cryptographic PRF code is removed from the runnable pipeline; the theory lives only in the paper.[file:2]

---

## 4. Configuration

### 4.1 Model configuration

Populate `config/models.yaml` from your CSV of OpenRouter models, for example:[file:5]

```yaml
models:
  - name: "nous-hermes-3-405b"
    openrouter_id: "nousresearch/hermes-3-llama-3.1-405b"
    params: 405e9
    context: 131000

  - name: "gpt-oss-120b"
    openrouter_id: "openai/gpt-oss-120b"   # adjust to exact OpenRouter ID
    params: 117e9
    context: 131000

  - name: "nemotron-3-super-120b"
    openrouter_id: "nvidia/nemotron-3-120b-instruct"
    params: 120e9
    context: 262000

  - name: "llama-3.3-70b-instruct"
    openrouter_id: "meta-llama/llama-3.3-70b-instruct"
    params: 70e9
    context: 66000

  - name: "venice-uncensored"
    openrouter_id: "venice/venice-uncensored"
    params: 24e9
    context: 33000

  - name: "qwen3-coder-480b-a35b"
    openrouter_id: "qwen/qwen3-coder-480b-a35b"
    params: 480e9
    context: 262000

  - name: "trinity-large-preview"
    openrouter_id: "arcee-ai/trinity-large-preview"
    params: 400e9
    context: 131000
```

Adjust each `openrouter_id` to match the actual model identifier on OpenRouter.[file:5]

### 4.2 Prompt configuration (from Hugging Face)

Evaluation prompts are drawn from a public safety dataset, e.g. the **unsafe subset** of `PKU-Alignment/PKU-SafeRLHF` on Hugging Face (or an equivalent safety/jailbreak dataset). A helper inside `run_llm_eval.py` should:

1. Load the dataset via the `datasets` library, using the HF token:

   ```python
   from datasets import load_dataset
   import os

   def load_prompts_from_hf(max_prompts: int = 500):
       ds = load_dataset(
           "PKU-Alignment/PKU-SafeRLHF",
           split="train",
           token=os.environ.get("HF_TOKEN", None),
       )
       # Example: assume 'is_safe' == 0 marks unsafe prompts
       unsafe = ds.filter(lambda ex: ex.get("is_safe", 1) == 0)
       unsafe = unsafe.select(range(min(max_prompts, len(unsafe))))
       return [
           {"id": f"unsafe_{i:04d}", "text": ex["prompt"]}
           for i, ex in enumerate(unsafe)
       ]
   ```

2. `run_llm_eval.py` then treats the returned list as the single source of evaluation prompts and iterates over them in order (deterministic round-robin over prompts).[file:2]

You may optionally mirror these prompts into `config/prompts.yaml` for inspection, but the runtime pipeline reads them directly from Hugging Face.

---

## 5. Filters and Access Modes

All filters wrap the **real LLM** and obey explicit compute limits.[file:2]

1. **Bounded Black-Box Filter (`filters/bounded_filter.py`)**  
   - Access: sampling and (optionally) approximate log-probs only.  
   - For each prompt, draws up to `max_queries` candidate outputs from the base LLM and uses a lightweight proxy risk score (e.g., lexical/structural features or a small classifier) to choose the lowest-risk candidate.

2. **White-Box Filter (`filters/whitebox_filter.py`)**  
   - Access: sampling plus token-level log-probabilities or probabilities when available.  
   - Uses entropy / confidence and simple feature correlations to score candidates under a query budget.

3. **Statistical-Query Filter (`filters/statistical_query_filter.py`)**  
   - Access: sampling via repeated queries.  
   - For each prompt, estimates expectations of a small feature bank (e.g., keyword indicators, style features) and rejects candidate outputs whose feature vector deviates too much from these expectations, under a bounded number of SQ calls.

A separate `is_harmful(prompt, output)` function (implemented in `experiments/metrics.py`) classifies outputs as harmful or not using a fixed heuristic or classifier.[file:2]

All previous **cryptographic toy generators**, **trapdoor filters**, and **ideal eliminators** must be deleted from the runtime code. They only remain as theoretical objects in the paper text.

---

## 6. Execution Roadmap

### 6.1 Selection policies (models, filters, prompts)

- **Model selection:**  
  `run_llm_eval.py` loads the ordered list of models in `config/models.yaml` and iterates over them sequentially. For any prompt, the active model is simply the current element of this outer loop (deterministic round-robin, \(O(1)\) selection).[file:2][file:5]

- **Filter selection:**  
  A fixed grid of filter types and budgets is defined in code/CLI, e.g. filter types = `["bounded", "whitebox", "sq"]`, budgets = `"1,4,16,64"`. `run_llm_eval.py` loops over all `(filter_type, filter_budget)` pairs for each model. For any prompt, the active filter is determined by this loop (again \(O(1)\) selection).[file:2]

- **Prompt selection:**  
  `run_llm_eval.py` calls `load_prompts_from_hf(...)` once, obtains a list, and iterates over prompts in order. For any step, the active prompt is the current index in this list (deterministic round-robin, \(O(1)\) selection).[file:2]

### 6.2 Single command to run all experiments

From the repo root:

```bash
python experiments/run_llm_eval.py \
    --models-config config/models.yaml \
    --filter-budgets "1,4,16,64" \
    --output-dir outputs/llm_results
```

This script:[file:2]

- Loads the model list from `config/models.yaml`.  
- Loads prompts from Hugging Face via `load_prompts_from_hf`.  
- For each model, filter type, and filter budget, runs the chosen filter around the LLM and logs per-prompt harmfulness.

### 6.3 What `run_llm_eval.py` does

For each `(model, filter_type, filter_budget)` triple:[file:2]

1. Instantiates `LLMGenerator` in `models/langdb_wrapper.py` using the `openrouter_id`.  
2. Wraps it with the specified filter (bounded black-box, white-box, or SQ) with the given budget.  
3. For each prompt in the loaded HF prompt list:
   - Calls `filter.sample(prompt_text, n=1)` (internally may query the base LLM multiple times, but never exceeding the budget).  
   - Evaluates `is_harmful(prompt_text, output)`.  
   - Logs a row with: model name, filter type, filter budget, prompt id, harmful (0/1), and total queries used.

Results are written to:

```plaintext
outputs/llm_results/llm_metrics.csv
```

---

## 7. Verification of Success

Reproduction is considered successful if:[file:2]

- **Persistence:** For each model and filter type, the estimated harmful rate **decreases but plateaus above zero** as `filter_budget` increases, rather than going to zero.  
- **Cross-Model Behavior:** A non-zero “harm floor” appears across multiple models in `config/models.yaml`, with model-specific levels but no model reaching zero harmful rate under finite budgets.[file:2][file:5]  
- **Visuals:** Running

  ```bash
  python experiments/plotting.py --metrics outputs/llm_results/llm_metrics.csv
  ```

  generates plots in `outputs/llm_results/plots/`, including:

  - `harm_vs_budget_<model>.png`: harmful rate vs filter budget for each model and filter type, showing saturation above zero.[file:2]

End of Reproduction Guide