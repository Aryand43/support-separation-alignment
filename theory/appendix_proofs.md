# Appendix-Level Proof Details

## A. TV-to-Event Bound Used in Reduction

For distributions $P,Q$ on measurable space $(Y,\mathcal{Y})$, and event $E\in\mathcal{Y}$:
$$
|P(E)-Q(E)| \le \mathrm{TV}(P,Q).
$$
Proof: follows from variational characterization
$$
\mathrm{TV}(P,Q)=\sup_{A\in\mathcal{Y}}|P(A)-Q(A)|.
$$

## B. Residual Harmful Mass Bound from TV Approximation

Let $H_x$ be harmful event and $P_x^\star(H_x)=0$. Then for any $P_x$:
$$
P_x(H_x)\le \mathrm{TV}(P_x,P_x^\star).
$$
This is Theorem 2's key conversion from distributional closeness to harmful-risk closeness.

## C. Hybrid Argument Template

Define worlds:

1. $\mathsf{Real}$: oracle bit comes from PRF $f_k$,
2. $\mathsf{Rand}$: oracle bit comes from random function $g$.

For any PPT procedure $\Pi$, if
$$
\left|\Pr[\Pi^{f_k}=1]-\Pr[\Pi^{g}=1]\right|
$$
is non-negligible, then PRF security is violated.

Reduction instantiates $\Pi$ as:

- run filter $F_n$,
- estimate residual harmful mass,
- threshold the estimate.

The residual-mass gap implies non-negligible distinguishing gap after concentration slack.

## D. Universal Support Persistence Technical Note

Absolute continuity alone does not force positivity on all sets.
The theorem therefore assumes strictly positive Radon--Nikodym density on harmful support almost everywhere:
$$
g_x(y)>0 \;\text{for}\; P_\theta(\cdot\mid x)\text{-a.e. }y\in H_x.
$$
Without this extra condition, an absolutely continuous map could still set $g_x=0$ on a positive-mass subset.

## E. Empirical Constant Estimator for Theorem 3

Define lower-bound estimator at security level $n$:
$$
\widehat{c}_n := \min_{F\in\mathcal{F}_{\mathrm{grid}}}\widehat{\rho}_{F,n},
$$
where $\mathcal{F}_{\mathrm{grid}}$ is tested filter hyperparameter grid.
Global empirical constant estimate:
$$
\widehat{c} := \min_{n\in\mathcal{N}} \widehat{c}_n.
$$
Confidence intervals can be obtained by bootstrap over prompts.
