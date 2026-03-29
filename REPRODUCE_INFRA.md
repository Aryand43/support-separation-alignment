# REPRODUCE_INFRA.md
# End-to-End Reproduction Guide: Support-Separation Alignment

This document provides the technical instructions for an AI agent to reproduce the infrastructure and results for the project: **"Support-Separation Alignment: Structural Floors in Computational Filtering"**.

---

## 1. Executive Summary (ELI10)
**The Concept:** We are proving that if a "bad behavior" is hidden inside a model using a secret mathematical rule (a "Trapdoor"), a standard security guard (a Filter) cannot find it without an impossible amount of work.

**The Goal:** Show that even the world's most advanced models (**GPT-5, Claude 4.5**) have a "Harm Floor" that stays above zero, proving that alignment is a structural problem, not just a training one.

---

## 2. ## Credentials
Set the following environment variable before running:

export OPENROUTER_API_KEY=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

pip install numpy pandas matplotlib seaborn openai tqdm
```

---

## 3. Directory Structure
Ensure the agent maintains the following hierarchy:

```plaintext
└── aryand43-support-separation-alignment/
    ├── experiments/          # Metrics and plotting logic
    ├── filters/              # Bounded, SQ, and Whitebox filter implementations
    ├── models/               # Base generators and LangDB wrappers
    ├── theory/               # Math proofs and PRF reduction logic
    ├── real_model_extension/ # Scripts for SOTA LLM evaluation
    └── outputs/              # CSVs and generated plots
```

---

## 4. Execution Roadmap

### Phase A: Toy Model Baseline
Run the synthetic experiments to establish the theoretical "Harm Floor" (c) using the PRF security parameter.

```bash
python experiments/run_experiment.py \
    --security-params "4,8,12" \
    --beta-values "0.0,1.0,2.0,4.0" \
    --filter-budgets "1,4,16,64" \
    --output-dir "./outputs/toy_results"
```

### Phase B: SOTA Benchmark (Section 4.5)
Validate the "Structural Floor" across five distinct alignment paradigms. This uses the LangDB gateway to hit the specific models identified for the paper.

| Model              | Alignment Paradigm        |
|--------------------|--------------------------|
| gpt-5              | RLHF / PPO               |
| claude-opus-4.5    | Constitutional AI        |
| gemini-2.5-pro     | RLAIF                    |
| grok-4             | Direct Preference (DPO)  |
| deepseek-r1-0528   | Reasoning-based Alignment|

Run command:

```bash
python real_model_extension/run_real_model_eval.py \
    --project-id "separation-alignmentc4780488-d081-7035-7b42-fe29463fd867" \
    --models "gpt-5,claude-opus-4.5,gemini-2.5-pro,grok-4,deepseek-r1-0528"
```

---

## 5. LangDB Gateway Implementation
The agent should utilize the following logic within `models/langdb_wrapper.py` to interface with the infra:

```python
from openai import OpenAI

class LangDBGenerator:
    def __init__(self, model_name, api_key):
        self.client = OpenAI(
            base_url="https://api.langdb.ai",
            api_key=api_key
        )
        self.model = model_name

    def sample(self, prompt, n_samples=1):
        """
        Samples from LangDB Gateway using the OpenAI-compatible SDK.
        """
        responses = []
        # Optimization: LangDB supports n-parameter for multiple completions
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n_samples
        )
        return [choice.message.content for choice in res.choices]
```

---

## 6. Verification of Success
Reproduced results are considered valid if:

- **Persistence:** The `harmful_mass` in `metrics.csv` plateaus at a non-zero value as `filter_budget` increases.
- **Cross-Model Invariance:** The "Harm Floor" appears consistently across all five models in Section 4.5.
- **Visuals:** `outputs/plots/harmful_mass_vs_budget.png` shows that no black-box filter reaches the performance of the `trapdoor_oracle`.

End of Reproduction Guide