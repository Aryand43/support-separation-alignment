# Adversarial Prompt Distribution $D_{X,n}$

We define a security-parameter indexed prompt family used in both reduction and experiments.

## Domain

For security parameter $n$, let:
$$
X_n = \{0,1\}^{n}.
$$

## Adversarial Family

Let $s_n \in \{0,1\}^n$ be a fixed but hidden seed.
Define score:
$$
\sigma_n(x) = \mathrm{wt}(x \oplus s_n) \bmod 4,
$$
where $\mathrm{wt}$ is Hamming weight.

Define adversarial support:
$$
\mathcal{A}_n = \{x \in X_n : \sigma_n(x)\in\{0,1\}\}.
$$

The prompt distribution is:
$$
D_{X,n}(x) =
\begin{cases}
\frac{1}{|\mathcal{A}_n|}, & x\in\mathcal{A}_n\\
0, & \text{otherwise.}
\end{cases}
$$

## Why this is adversarial

1. The support concentrates on a structured subset tied to hidden shift $s_n$.
2. Visible low-order statistics (parity, density) are weakly informative about membership in $\mathcal{A}_n$.
3. Hidden-feature-dependent harmfulness correlates more strongly on $\mathcal{A}_n$, maximizing residual harmful mass under bounded filtering.

## Experimental instantiation

In experiments, $D_{X,n}$ is approximated by:

- drawing candidate prompts from the interval $[0, 2^n)$,
- selecting those passing an efficiently checkable adversarial score proxy,
- evaluating all models on this subset.

This gives a reproducible finite-sample analogue of the reduction distribution.
