# Phase B: real model extension

Validates a structural **harm floor** across several SOTA models via the LangDB gateway: adversarial prompts are tested under raw generation, a safety system prompt, and system prompt plus a small keyword post-filter.

## What it does

- Calls each model with a fixed set of adversarial prompts (multiple samples per prompt).
- Records three **conditions** per run: `raw`, `system_prompt`, and `system_prompt+filter` (lexical block list).
- Writes a CSV and bar plots of harmful rate by model and condition.

## Run

```bash
pip install matplotlib seaborn pandas pyyaml tqdm
python experiments/phase_b_real_model_extension/run_real_model_eval.py
```

Optional:

```bash
python experiments/phase_b_real_model_extension/run_real_model_eval.py \
  --models gpt-5,claude-opus-4.5 \
  --samples-per-prompt 3 \
  --output-dir outputs
```

Outputs:

- `outputs/phase_b_real_models.csv`
- `outputs/plots/phase_b_harmful_rate_by_model.png` (and related figures)

This extension is qualitative evidence only; it is not a full benchmark or retraining study.
