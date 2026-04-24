# Experiment Expansion Plan

This document specifies concrete, executable steps to aggressively expand the experiments for the NeurIPS push. It is organized as independent modules you can implement and run incrementally.

---

## 0. Current Core Pipeline (Reference)

Existing capabilities (do not change, only extend):

- Models via OpenRouter specified in `config/models.yaml`. [file:17]  
- Adversarial prompts:
  - Curated indirect cybersecurity prompts (`FALLBACK_PROMPTS` in `experiments/runllmeval.py`). [file:17]  
  - Unsafe subset of PKU-SafeRLHF via `datasets`. [file:17]  
- Filters:
  - `BoundedBlackBoxFilter` (sampling + keyword risk score). [file:17]  
  - `WhiteBoxFilter` (sampling + length/confidence heuristics). [file:17]  
  - `StatisticalQueryFilter` (SQ-style threshold). [file:17]  
- Metrics:
  - `isharmful(prompt, output)` and `riskscore(output)` in `filters/metrics.py`. [file:17]  
- Main evaluation:
  - `experiments/runllmeval.py` → `outputs/llmresults/llmmetrics.csv`. [file:17]  
- Plots:
  - `experiments/plotting.py` → harm vs budget curves + harm floors per model. [file:17]  
- Phase B qualitative paradigm comparison:
  - `realmodelextension/runrealmodeleval.py`. [file:17]  

---

## 1. Expand Prompt Suite

### 1.1 New Prompt Datasets

**Goal:** Increase coverage and diversity of adversarial inputs.

**Tasks:**

1. Add a new module `experiments/prompt_sources.py` that defines:
   - `load_curated_prompts()`: wrap existing FALLBACK prompts. [file:17]  
   - `load_pku_prompts(max_prompts)`: existing PKU unsafe subset loader. [file:17]  
   - `load_additional_datasets(max_prompts)`: hooks for other unsafe/safety datasets (other jailbreak / unsafe sets from related work). [file:16][file:17]  

2. Define a unified loader in `prompt_sources.py`:
   - `load_all_prompts(max_prompts, mix_config)` that:
     - Takes a config specifying how many prompts from each source.
     - Returns a list of dicts: `{"id": ..., "text": ..., "source": ..., "attack_type": ...}`. [file:17]  

3. Update `experiments/runllmeval.py`:
   - Replace the current `load_prompts_from_hf` logic with calls to `prompt_sources.load_all_prompts`. [file:17]  
   - Ensure the `source` field is logged in `llmmetrics.csv`.  

### 1.2 Explicit Jailbreak Categories

**Goal:** Tag prompts with attack types aligned with the paper’s taxonomy. [file:16]

Attack types:

- `direct_request`  
- `multi_turn_steering`  
- `cot_exploitation`  
- `roleplay_camouflage`  

**Tasks:**

1. In `experiments/prompt_sources.py`:
   - For each prompt you construct or load, assign an `attack_type` field. [file:17]  
   - For multi-turn / CoT / role-play, design prompt templates that match the concrete examples in the paper:
     - Progressive “cybersecurity course” → “specific payload” steering.  
     - CoT prefixes like “Let’s think step by step about how one might theoretically…”.  
     - Role-play thriller scenarios embedding harmful instructions in dialogue. [file:16]  

2. Log `attack_type` into `llmmetrics.csv` alongside `promptid` and `source`. [file:17]  

3. Update plotting:
   - Extend `experiments/plotting.py` to generate:
     - Harm vs budget per `attack_type`.
     - Harm floors per `attack_type` and model. [file:17]  

---

## 2. Increase Prompt Count & Budgets

**Goal:** Show robustness of harm floors under more prompts and higher budgets. [file:16][file:17]

**Tasks:**

1. Confirm CLI options in `experiments/runllmeval.py`:
   - `--max-prompts` (already exists) – plan to run with values: `50`, `200`, `500`. [file:17]  
   - `--filter-budgets` – use grids like `"1,4,16,64"` and `"1,4,16,64,256"`. [file:17]  

2. Define standard experiment configs (to document in README):

   - **Config A (baseline):**  
     - `max_prompts = 50`  
     - `filter_budgets = [1,4,16,64]`  

   - **Config B (prompt-expanded):**  
     - `max_prompts = 200`  
     - `filter_budgets = [1,4,16,64]`  

   - **Config C (high-budget slice):**  
     - `max_prompts = 100`  
     - `filter_budgets = [1,4,16,64,256]` (compute-heavy slice)  

3. Output directories:
   - Use separate directories per config:
     - `outputs/llmresults_A`, `outputs/llmresults_B`, `outputs/llmresults_C`. [file:17]  

4. Update plotting:
   - Allow passing multiple `--metrics` paths.
   - Overlay harm vs budget curves from `A/B/C` with clear legends (`50p`, `200p`, `100p+256`). [file:17]  

---

## 3. Stronger Baselines

### 3.1 High-Budget Filter Baseline

**Goal:** Show that even very high budgets still yield non-zero harm floors. [file:16][file:17]

**Tasks:**

1. Extend `FILTER_REGISTRY` in `experiments/runllmeval.py` with a high-budget variant:
   - E.g., key: `"bounded_high"` → `BoundedBlackBoxFilter(gen, maxqueries=256)` or `512`. [file:17]  

2. Add this filter type to a dedicated run config:
   - `filtertypes = ["bounded", "whitebox", "sq", "bounded_high"]`.  

3. Log and plot:
   - Compare harm floors of `bounded_high` vs standard `bounded`, `whitebox`, `sq`. [file:17]  
   - Use `totalqueries` to report average query count per configuration. [file:17]  

### 3.2 Native Safety & Guardrail Baseline

**Goal:** Compare against strong native safety prompts and a non-bounded guardrail.

**Tasks:**

1. Native safety condition:
   - Add a `condition` column in `llmmetrics.csv` to distinguish:
     - `baseline` (current bounded filters).
     - `native_safety` (strong safety/system prompts, no filter). [file:17]  
   - Implement a code path in `runllmeval.py` that:
     - Calls `LangDBGenerator` or OpenRouter with a robust safety system prompt only (similar to Phase B’s `SAFETYSYSTEMPROMPT`). [file:17]  
     - Logs outputs with `filtertype = "native"` and `condition = "native_safety"`.  

2. Strong guardrail filter:
   - Implement `filters/guardrail.py` or similar:
     - Combines keyword patterns, refusal detection, and `riskscore` but does not obey a strict small budget (can use a higher budget). [file:17]  
   - Add to `FILTER_REGISTRY` as `"guardrail"`.  

3. Plots:
   - Harm floor per model for `bounded`, `whitebox`, `sq`, `bounded_high`, `guardrail`, `native`. [file:17]  

---

## 4. Ablation Studies

### 4.1 Harm Predicate Ablations

**Goal:** Show harm floors persist under different reasonable harm predicates. [file:17]

**Tasks:**

1. In `filters/metrics.py`, define:

   - `isharmful_default` (current behavior). [file:17]  
   - `isharmful_strict`:
     - More compliance patterns.
     - Lower tolerance for partial compliance (label harmful more often).  
   - `isharmful_lax`:
     - Fewer compliance patterns.
     - Treat borderline outputs as safe.  

2. In `experiments/runllmeval.py`:
   - Add CLI flag `--harm-predicate {default,strict,lax}`.
   - Wire this to select the appropriate function when computing `isharmful`. [file:17]  

3. Plots:
   - For each model, show harm floors under `default`, `strict`, `lax`. [file:17]  
   - Emphasize that floors remain > 0 across all variants. [file:16]  

### 4.2 Filter Design Ablations

**Goal:** Test sensitivity to filter design details. [file:17]

**Tasks:**

1. Black-box filter ablation:
   - Implement `BoundedRandomFilter`:
     - Samples up to `maxqueries` but selects a candidate uniformly at random (ignores `riskscore`). [file:17]  
   - Add to `FILTER_REGISTRY` as `"bounded_random"`.  

2. White-box filter ablations:
   - Add flags or separate classes to disable:
     - Length penalties (`whitebox_nolen`).  
     - Confidence heuristics (`whitebox_noconf`). [file:17]  
   - Register them as distinct filter types.  

3. SQ filter ablations:
   - Expose `numstatqueries` via CLI and sweep values e.g., `2,4,8`. [file:17]  
   - Log `numstatqueries` in the CSV for analysis.  

4. Plots:
   - Harm vs budget for each ablation variant.
   - A table summarizing harm floors and average `totalqueries` per variant. [file:17]  

---

## 5. Attack-Type-Specific Analyses

**Goal:** Quantify which jailbreak patterns are most resistant to bounded filtering. [file:16][file:17]

**Tasks:**

1. Ensure every prompt has `attack_type` recorded in the metrics CSV (from Section 1.2). [file:17]  

2. Extend `experiments/plotting.py`:
   - Group by `(model, filtertype, filterbudget, attack_type)` and compute mean harmful rate. [file:17]  
   - Produce:
     - Harm vs budget curves per `attack_type`.  
     - Bar plots of harm floors per `attack_type` and model.  

3. Add a failure-taxonomy summary:
   - Table: rows = `attack_type`, columns =:
     - `min_harm_rate` (floor) across all budgets/filters.
     - `% reduction vs no-filter baseline`. [file:17]  

---

## 6. Strengthen Phase B (Paradigm Comparison)

**Goal:** Turn Phase B into a more substantial, semi-quantitative benchmark. [file:17]

**Tasks:**

1. Prompt set:
   - Extend `ADVERSARIAL_PROMPTS` in `realmodelextension/runrealmodeleval.py`:
     - Include prompts for each `attack_type` (`direct_request`, `multi_turn_steering`, `cot_exploitation`, `roleplay_camouflage`). [file:16][file:17]  
   - Add an `attack_type` label for each prompt and log it in the CSV. [file:17]  

2. Sample size:
   - Increase `samples_per_prompt` default to ≥ 5. [file:17]  

3. Conditions:
   - Keep three conditions:
     - `raw`  
     - `system_prompt`  
     - `system_prompt_filter` (keyword/guardrail filter). [file:17]  
   - Ensure the filter uses logic consistent with `isharmful`/`riskscore`. [file:17]  

4. Output:
   - Confirm `outputs/realmodelextension.csv` logs: `model`, `paradigm`, `condition`, `attack_type`, `isharmful`. [file:17]  

5. Plots:
   - Extend `plotresults` to create:
     - Bar chart: harmful rate by paradigm × condition. [file:17]  
     - Harm floor by paradigm (minimum over conditions) with simple confidence intervals (e.g., bootstrap).  

---

## 7. Compute & Cost Narratives

**Goal:** Make “bounded” concrete by quantifying query overhead. [file:16][file:17]

**Tasks:**

1. In `experiments/runllmeval.py`:
   - You already log `totalqueries` per row. [file:17]  
   - Add a script `experiments/cost_analysis.py` that:
     - Reads one or more `llmmetrics.csv` files.
     - Aggregates average `totalqueries` per `(model, filtertype, filterbudget)`.  
     - Optionally multiplies by a configurable per-query cost to estimate cost per 1k prompts. [file:17]  

2. Plots (in `cost_analysis.py` or `plotting.py`):
   - Harm rate vs average `totalqueries` per prompt.
   - Harm rate vs approximate cost per 1k prompts.
   - Highlight diminishing returns: harm curves flatten while queries/cost keep increasing. [file:16][file:17]  

---

## 8. Recommended Run Matrix (For NeurIPS)

After implementing the above, run at least:

1. **Main expanded run**  
   - `max_prompts = 200`  
   - `filtertypes = bounded, whitebox, sq`  
   - `filter_budgets = 1,4,16,64`  
   - All models from `config/models.yaml`. [file:17]  

2. **High-budget slice**  
   - `max_prompts = 100`  
   - Include `bounded_high` (e.g., 256 queries) plus standard filters. [file:17]  

3. **Ablation runs**  
   - One run each with `--harm-predicate=default`, `strict`, `lax`. [file:17]  
   - One run including `bounded_random` and white-box / SQ ablations.  

4. **Phase B expanded**  
   - Updated prompts, `samples_per_prompt ≥ 5`, all paradigms in `MODELREGISTRY`. [file:17]  

5. **Cost analysis**  
   - Run `experiments/cost_analysis.py` on all `llmmetrics.csv` outputs. [file:17]  

---

## 9. Integration Notes

- Put all new prompt logic in `experiments/prompt_sources.py`. [file:17]  
- Put new filters in `filters/` and register them in `FILTER_REGISTRY`. [file:17]  
- Ensure all new runs write to distinct subdirectories in `outputs/`. [file:17]  
- Update `README.md` and `REPRODUCE_INFRA.md` to document:
  - New prompt sources and `attack_type` taxonomy. [file:17]  
  - New filters and ablation flags.  
  - Example commands for Config A/B/C, high-budget runs, ablations, Phase B, and cost analysis. [file:17]  