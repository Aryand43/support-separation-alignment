# support-separation-alignment

Research-grade toy repository for demonstrating **support separation**:

1. **Structural claim**: support-preserving RLHF-style alignment cannot remove harmful support.
2. **Computational claim**: bounded black-box filters cannot approximate trapdoor/ideal elimination under hardness assumptions.
3. **Empirical claim**: residual harmful mass plateaus above zero for bounded filtering, while trapdoor elimination reaches near-zero.

---

## Repository Architecture

```
support-separation-alignment/
├── theory/
│   ├── definitions.py
│   ├── theorems.md
│   ├── proof_sketches.md
├── models/
│   ├── base_generator.py
│   ├── rlhf_alignment.py
│   ├── ideal_eliminator.py
├── filters/
│   ├── bounded_filter.py
│   ├── trapdoor_filter.py
├── experiments/
│   ├── run_experiment.py
│   ├── metrics.py
│   ├── plotting.py
├── utils/
│   ├── data_generation.py
│   ├── config.py
└── README.md
```

### Separation of Concerns

- `theory/` is formal-only and does not import empirical/model code.
- `models/` defines conditional distributions and alignment/elimination operators.
- `filters/bounded_filter.py` uses only black-box sampling/log-prob APIs.
- `filters/trapdoor_filter.py` represents the unattainable oracle baseline.
- `experiments/` orchestrates sweeps, metrics, and plotting.

---

## Theory Mapping

- **Formal objects and assumptions**: `theory/definitions.py`
  - Prompt space \(X\), response space \(Y\), harmful predicate \(h\)
  - Base model \(P_\theta(y\mid x)\)
  - Support-preserving alignment \(A\)
  - Ideal eliminator \(P^\star(y\mid x)\)
- **Publication-style theorem statements**: `theory/theorems.md`
  - Theorem 1: Structural Support Persistence
  - Theorem 2: Computational Separation
  - Theorem 3: Residual Harmful Mass Lower Bound
- **Proof sketches**: `theory/proof_sketches.md`

---

## Model and Filter Design

### Base Generator (`models/base_generator.py`)

- Output is \(y=(c,r,b)\):
  - `c`: challenge bitstring
  - `r`: noise bitstring
  - `b`: pseudo-PRF prediction bit
- Trapdoor harmfulness is keyed via hidden MLP PRF:
  - harmful iff `b == PRF_k(x, c)`
- Distribution remains strictly positive over finite support.

### RLHF-style Alignment (`models/rlhf_alignment.py`)

- Reweighting:
  \[
  \tilde P(y|x)\propto P_\theta(y|x)\exp(\beta r_\phi(x,y))
  \]
- Reward model uses only observable proxies (no trapdoor).
- No hard-zeroing, so support is preserved.

### Ideal Eliminator (`models/ideal_eliminator.py`)

- Oracle baseline:
  \[
  P^\star(y|x)\propto P(y|x)\mathbf{1}[h(y)=0]
  \]
- Can zero harmful mass using trapdoor harmfulness oracle.

### Filters

- `bounded_filter.py`: polynomial-time bounded query filter (`max_queries = T`), black-box only.
- `trapdoor_filter.py`: oracle/trapdoor eliminator baseline.

---

## Running Experiments

## 1) Install dependencies

```bash
pip install numpy pandas matplotlib seaborn
```

## 2) Run CLI

```bash
python experiments/run_experiment.py
```

Optional sweep overrides:

```bash
python experiments/run_experiment.py \
  --beta-values "0.0,0.5,1.0,2.0,4.0" \
  --filter-budgets "1,2,4,8,16,32" \
  --filter-capacities "4,8,16,32" \
  --eval-prompts 32 \
  --samples-per-prompt 512 \
  --tv-samples 4096 \
  --seed 7
```

Outputs:

- `outputs/metrics.csv`
- `outputs/plots/harmful_mass_vs_budget.png`
- `outputs/plots/harmful_mass_vs_capacity.png`
- `outputs/plots/tv_vs_runtime.png`
- `outputs/plots/beta_vs_residual_harmful_mass.png`

---

## Expected Empirical Pattern

The experimental target is:

1. RLHF decreases harmful probability but leaves non-zero residual mass.
2. Bounded black-box filtering reduces harmful mass further but plateaus above zero.
3. Trapdoor eliminator reaches near-zero harmful mass.
4. Better filtering requires more compute (query/runtime scaling).

These outcomes operationalize the support-separation claim.
