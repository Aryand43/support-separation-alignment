# Real Model Extension (Qualitative)

This folder provides a lightweight transformer-compatible qualitative check.

## What it does

- Runs a small open model on a mini jailbreak-style prompt set.
- Sweeps:
  - multiple sampling temperatures,
  - multiple safety prefix strengths.
- Compares:
  - `base` generation,
  - `prefix_only` alignment proxy,
  - `guard_proxy` post-hoc lexical safety filter.
- Reports residual unsafe rate and plots unsafe-rate plateau trends.

## Run

```bash
pip install transformers accelerate torch matplotlib seaborn pandas
python real_model_extension/run_real_model_eval.py
```

Optional:

```bash
python real_model_extension/run_real_model_eval.py \
  --temperatures "0.3,0.7,1.0" \
  --prefix-strengths "0.0,0.25,0.5,0.75,1.0"
```

Outputs:

- `outputs/real_model_extension.csv`
- `outputs/plots/real_model_unsafe_vs_prefix_strength.png`

This extension is qualitative evidence only; it is not a full benchmark or retraining study.
