# Support-Preserving Alignment Theorems

Let $X$ be a prompt space, $Y$ a response space, and $h: X \times Y \to \{0,1\}$ a harmful predicate.
Let $P_\theta(\cdot \mid x)$ be a base conditional distribution and $H_x := \{y\in Y : h(x,y)=1\}$.

Define a support-preserving stochastic alignment operator $A$ by:
$$
P_\theta(y\mid x) > 0 \implies (A P_\theta)(y\mid x) > 0.
$$

This includes as special cases RLHF/PPO reweighting, DPO-like preference reweighting, and generic reward-tilting operators
$$
\widetilde P(y\mid x)\propto P_\theta(y\mid x)\exp(\beta r(x,y)),
$$
provided they do not hard-zero support points.

Define the ideal eliminator:
$$
P^\star(y\mid x)\propto P_\theta(y\mid x)\mathbf{1}[h(x,y)=0].
$$

## Theorem 1 (Universal Support Persistence)

Assume for each $x$, $(A P_\theta)(\cdot\mid x)\ll P_\theta(\cdot\mid x)$, and the Radon--Nikodym derivative
$\frac{d(A P_\theta)(\cdot\mid x)}{dP_\theta(\cdot\mid x)}$ is positive $P_\theta$-a.e. on $H_x$.
If $P_\theta(H_x\mid x)>0$, then:
$$
(A P_\theta)(H_x\mid x)>0.
$$
So support-preserving alignment cannot eliminate harmful support mass.

## Theorem 2 (Computational Separation via Hidden Feature Hardness)

Assume one-way functions exist. Then there exists a family of generators whose harmfulness depends on a computationally hidden harmful feature, and a constant $\varepsilon_0>0$, such that:

for every PPT black-box filter $F$, for infinitely many security parameters $n$,
$$
\mathbb{E}_{x\sim D_{X,n}}\!\left[\mathrm{TV}\!\left(P_{F,A,\theta}^{(n)}(\cdot\mid x),\,P^{\star,(n)}(\cdot\mid x)\right)\right]\ge \varepsilon_0.
$$

Thus bounded black-box filtering cannot approximate ideal elimination arbitrarily well.

## Theorem 3 (Information-Theoretic Residual Floor)

There exists a hypothesis class of harmful events with VC-dimension $d$ such that for any support-preserving alignment $A$, any statistical-query filter using at most $m$ queries, and some distribution $D$:
$$
\mathbb{E}_{x\sim D}\!\left[P_{F,A,\theta}(H_x\mid x)\right]\ge \Omega\!\left(\frac{d}{m}\right),
$$
equivalently at least $\Omega(1/\mathrm{poly}(m))$ for fixed $d$-scaling regimes.

So residual harmful floors arise even without cryptographic assumptions.

## Theorem 4 (Residual Harmful Mass Lower Bound)

Combining Theorems 1--3: under bounded filtering (computational or SQ-limited), there exists $c>0$ such that residual harmful mass remains bounded away from zero on some inputs/distributions.

## White-Box Clarification (Restricted Claim)

White-box access to logits/parameters without harmful ground-truth oracle access does not, in general, collapse the separation unless the alignment map exposes harm-relevant sufficient statistics. This is an assumption-sensitive statement, not a blanket impossibility theorem for all realistic neural settings.
