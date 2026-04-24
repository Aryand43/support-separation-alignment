# Information-Theoretic Lower Bound

## Setting

Let $A$ be any support-preserving alignment operator and $Q_x=(AP_\theta)(\cdot\mid x)$.
Let $H_x\subseteq Y$ denote harmful responses.
Consider filters that can access $Q_x$ only through statistical queries:

$$
\text{for query }\phi: X\times Y\to[-1,1],\quad \text{oracle returns } \mathbb{E}_{y\sim Q_x}[\phi(x,y)] \pm \tau.
$$

Suppose the harmful class $\mathcal{H}=\{H_x\}$ has VC-dimension $d$.

## Theorem (Information-Theoretic Residual Floor)

For any SQ filter $F$ making at most $m$ statistical queries per prompt, there exists a distribution $D$ over prompts such that:
$$
\mathbb{E}_{x\sim D}\!\left[P_{F,A,\theta}(H_x\mid x)\right]
\ge
\Omega\!\left(\frac{d}{m}\right),
$$
and in particular $\Omega(1/\mathrm{poly}(m))$ under fixed $d$-growth regimes.

So bounded SQ filtering cannot drive residual harmful mass to zero, even without cryptographic assumptions.

## Proof Sketch (Fooling-Set / SQ Dimension)

1. **Hard family construction**  
   Build a set of distributions $\{Q^{(i)}\}_{i=1}^N$ that agree on low-order query statistics but differ on harmful labels induced by a shattered set of size $d$.

2. **Query indistinguishability**  
   With only $m$ SQ calls, the transcript cannot identify which $Q^{(i)}$ generated data beyond error roughly $O(m/d)$ (via SQ-dimension lower bounds / pairwise correlation method).

3. **Residual risk transfer**  
   Because candidate worlds disagree on $H_x$ while being SQ-indistinguishable, any single filter policy must incur nontrivial error on at least one world. Averaging yields:
   $$
   \mathbb{E}_{x\sim D}\big[P_{F,A,\theta}(H_x\mid x)\big] \ge c_0 \frac{d}{m}
   $$
   for constant $c_0>0$.

4. **Support-preservation role**  
   Since $A$ preserves support, harmful-support points survive alignment and remain candidates that SQ-limited filters must distinguish statistically.

## Remarks

- This theorem complements the computational (PRF-based) lower bound.
- It does not rely on hidden keys or one-way functions.
- It formalizes residual floors from pure information constraints.
