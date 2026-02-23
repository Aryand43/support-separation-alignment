# Real Model Extension (Optional)

This folder provides a lightweight bridge from the toy hardness setup to a small open chat model.

## What it does

- Generates responses on a mini jailbreak-style prompt list.
- Compares three conditions:
  - `base` (raw generation),
  - `rlhf_proxy` (safety-prefixed generation),
  - `guard_proxy` (post-hoc lexical safety gate).
- Writes aggregate safety rates to CSV.

## Run

```bash
pip install transformers accelerate torch
python real_model_extension/run_real_model_eval.py
```

Output:

- `outputs/real_model_extension.csv`

This is not a full retraining pipeline; it is a minimal extension to demonstrate qualitative residual-risk plateau behavior on real model outputs.
