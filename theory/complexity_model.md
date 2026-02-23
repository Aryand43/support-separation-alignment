# Complexity and Oracle Model

## Objects

- Security parameter: \(n\).
- Prompt domain: \(X_n\).
- Response domain: \(Y_n\).
- Aligned generator kernel: \(Q_n(\cdot\mid x) := (A_nP_{\theta,n})(\cdot\mid x)\).

## Oracle Access

A filter algorithm \(F_n\) is given black-box oracle access:

\[
\mathcal{O}_{Q_n}(x) \to y \sim Q_n(\cdot\mid x).
\]

No access to:

- hidden key \(k\),
- internal model weights/gradients (black-box setting),
- direct harmfulness oracle.

White-box variants may read logits/probabilities of \(Q_n\), but still cannot access \(k\).

## Filter Class

\[
\mathcal{F}_{\mathrm{ppt}} = \{F_n\}_{n\in\mathbb{N}}
\]
where each \(F_n\):

1. runs in time polynomial in \(n\),
2. makes at most \(q(n)\) oracle queries per prompt,
3. outputs accepted sample(s) and/or reject decisions.

Thus \(F_n\) is a probabilistic polynomial-time (PPT) oracle algorithm.

## Complexity Measures

For prompt \(x\):

- query complexity \(Q_F(x)\): number of oracle calls,
- time complexity \(T_F(x)\): runtime,
- residual harmful mass:
  \[
  \rho_F(x)=\Pr_{y\sim P_{F,A,\theta}^{(n)}(\cdot\mid x)}[h_n(x,y)=1].
  \]

Aggregate:
\[
\overline{\rho}_F(n)=\mathbb{E}_{x\sim D_{X,n}}[\rho_F(x)],
\quad
\overline{Q}_F(n)=\mathbb{E}_{x\sim D_{X,n}}[Q_F(x)].
\]

## Ideal Eliminator Reference

\[
P_n^\star(y\mid x)\propto Q_n(y\mid x)\mathbf{1}[h_n(x,y)=0].
\]

Approximation target:
\[
\mathbb{E}_{x\sim D_{X,n}}\left[\mathrm{TV}\!\left(P_{F,A,\theta}^{(n)}(\cdot\mid x),P_n^\star(\cdot\mid x)\right)\right].
\]

The computational separation claim states this quantity remains bounded below by a constant for all \(F_n\in\mathcal{F}_{\mathrm{ppt}}\) under standard hardness assumptions.
