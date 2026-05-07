# Confidence Interval Integration for NeurIPS Statistical Rigor

## Objective

Add statistical confidence intervals to the existing harmful-rate evaluation plots to improve empirical rigor and reduce reviewer criticism that the results are merely illustrative or anecdotal.

This change does NOT require:
- new experiments
- new datasets
- rerunning models
- modifying filters
- changing repository architecture

The goal is purely to improve statistical presentation quality for NeurIPS reviewers.

---

# What “Anecdotal” Means

Anecdotal results are results that appear based on isolated observations rather than statistically supported evidence.

Right now the plots effectively say:

```text
harmful_rate = 0.23
```

But reviewers do not know:
- whether this number is stable
- whether it changes significantly under different prompt samples
- or whether the curve is noisy

This makes the plots potentially look:
- illustrative
- heuristic
- observational

rather than statistically rigorous.

---

# What Confidence Intervals Do

Confidence intervals estimate uncertainty around the measured harmful rate.

Instead of saying:

```text
harmful_rate = 0.23
```

the plot now visually communicates:

```text
harmful_rate ≈ 0.23 ± uncertainty
```

This produces:
- a solid mean curve
- plus a shaded confidence band around the curve

Current visual:

```text
──────
```

Desired visual:

```text
░░────░░
```

Where:
- the solid line is the average harmful rate
- the shaded region is the 95% confidence interval

Small confidence bands:
- imply low variance
- imply stable results
- increase reviewer trust

Large confidence bands:
- imply noisier estimates
- imply higher uncertainty

---

# Why This Matters for NeurIPS

Without confidence intervals:
- plots can look anecdotal
- reviewers may view them as unstable empirical observations

With confidence intervals:
- results appear statistically evaluated
- plots resemble standard NeurIPS empirical presentation quality
- reviewer confidence improves substantially

This is an extremely high ROI improvement because:
- implementation is fast
- no reruns are needed
- no infrastructure changes are needed

---

# Repository Scope

Modify ONLY:

```plaintext
experiments/plotting.py
```

No changes are required to:
- filters
- evaluation pipelines
- OpenRouter integration
- datasets
- CSV logging
- model infrastructure

---

# Statistical Method

For each group:

```python
(model, filter_type, filter_budget)
```

compute:
- mean harmful rate
- sample count
- approximate 95% confidence interval

The harmfulness labels already exist in:

```python
is_harmful
```

inside:

```plaintext
llm_metrics.csv
```

so no new data generation is needed.

---

# Exact Implementation

## Step 1 — Replace Existing Summary Block

Inside:

```python
generate_llm_plots()
```

after:

```python
ok = df[df["status"] == "ok"].copy()
```

REPLACE the current summary block:

```python
summary = (
    ok.groupby(["model", "filter_type", "filter_budget"])["is_harmful"]
    .mean()
    .reset_index()
)
summary.columns = ["model", "filter_type", "filter_budget", "harmful_rate"]
```

WITH:

```python
summary = (
    ok.groupby(["model", "filter_type", "filter_budget"])["is_harmful"]
    .agg(["mean", "count", "std"])
    .reset_index()
)

summary["ci95"] = 1.96 * (
    (
        summary["mean"]
        * (1 - summary["mean"])
        / summary["count"]
    ) ** 0.5
)
```

This computes:
- average harmful rate
- sample count
- standard deviation
- approximate 95% confidence interval

---

# Step 2 — Update Plotting Variables

Replace all references to:

```python
harmful_rate
```

with:

```python
mean
```

inside the plotting calls.

---

# Step 3 — Add Confidence Bands to Per-Model Curves

Inside:

```python
for model_name in summary["model"].unique():
```

after the existing `sns.lineplot(...)` call, add:

```python
for ft in msub["filter_type"].unique():
    subset = msub[msub["filter_type"] == ft]

    plt.fill_between(
        subset["filter_budget"],
        subset["mean"] - subset["ci95"],
        subset["mean"] + subset["ci95"],
        alpha=0.2,
    )
```

This creates the shaded uncertainty region around each line.

---

# Step 4 — Add Confidence Bands to All-Models Overlay

After the global `sns.lineplot(...)` call for:

```python
harm_vs_budget_all_models.png
```

add:

```python
for (model_name, ft), subset in summary.groupby(["model", "filter_type"]):
    plt.fill_between(
        subset["filter_budget"],
        subset["mean"] - subset["ci95"],
        subset["mean"] + subset["ci95"],
        alpha=0.12,
    )
```

---

# Expected Output Change

The updated plots will now show:
- mean harmful-rate curves
- uncertainty bands
- statistically grounded empirical variation

This significantly improves:
- reviewer trust
- empirical credibility
- presentation quality
- perceived rigor

without changing the underlying experiments.

---

# Important Constraints

DO NOT:
- rerun evaluations
- modify datasets
- change CSV schemas
- rewrite plotting infrastructure
- alter filters
- add external statistical dependencies

Only improve statistical presentation.

---

# Final Deliverable

Updated plots containing:
- 95% confidence bands
- statistically grounded harmful-rate visualization
- stronger NeurIPS-style empirical presentation quality