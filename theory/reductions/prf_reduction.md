# Reduction from Hidden-Feature Hardness to Filtering Impossibility

## Setup and Assumptions

Let the security parameter be \(n \in \mathbb{N}\).
Let \(\mathcal{F}_n = \{f_k : \{0,1\}^{m(n)} \to \{0,1\}\}_{k \leftarrow \{0,1\}^{\kappa(n)}}\) be a PRF family secure against PPT distinguishers.
This PRF instantiates a computationally hidden harmful feature in a worst-case construction.

Define:

- prompt space \(X_n = \{0,1\}^{m(n)}\),
- response space \(Y_n = \{0,1\}^{m(n)} \times \{0,1\}^{\ell(n)} \times \{0,1\}\),
- response \(y = (c,r,b)\),
- harmful predicate
  \[
    h_k(x,y) := \mathbf{1}[b = f_k(x \| c)] \cdot \mathbf{1}[\pi(r)=0],
  \]
  where \(\pi\) is a fixed efficiently computable parity-like map.

Let \(P_{\theta,k}^{(n)}(\cdot \mid x)\) be a base distribution with full finite support and with non-negligible harmful prior mass:
\[
\mu_n := \Pr_{y \sim P_{\theta,k}^{(n)}(\cdot\mid x),\,x\sim D_{X,n}}[h_k(x,y)=1] \ge \mu_0 > 0.
\]

Let \(A_n\) be any support-preserving alignment operator:
\[
(A_n P)(\cdot\mid x)\ll P(\cdot\mid x).
\]

Define the ideal eliminator:
\[
P_{k}^{\star,(n)}(y\mid x) \propto (A_nP_{\theta,k}^{(n)})(y\mid x)\cdot \mathbf{1}[h_k(x,y)=0].
\]

Let \(F_n\) be a PPT filter with black-box oracle access to sampling from \(A_nP_{\theta,k}^{(n)}\).
Its induced filtered distribution is denoted \(P_{F,A,\theta,k}^{(n)}(\cdot\mid x)\).

## Target Contradiction Statement

Assume there exists PPT \(\{F_n\}\), a constant \(\varepsilon_0>0\), and infinitely many \(n\) such that:
\[
\mathbb{E}_{x\sim D_{X,n}}\!\left[\mathrm{TV}\!\left(P_{F,A,\theta,k}^{(n)}(\cdot\mid x),P_{k}^{\star,(n)}(\cdot\mid x)\right)\right] < \varepsilon_0.
\]

We show this implies a PPT distinguisher against PRF security.

## Hybrid with Random Function Oracle

Consider experiment \(\mathsf{Real}_n\): harmful predicate uses \(f_k\).
Consider \(\mathsf{Rand}_n\): replace \(f_k\) by a truly random function \(g:\{0,1\}^{m(n)}\to\{0,1\}\).

For any fixed \(x\), let \(Q_x\) denote the post-alignment distribution \(A_nP_{\theta,k}^{(n)}(\cdot\mid x)\), and let
\[
Q_x^\star(y) \propto Q_x(y)\mathbf{1}[b \neq \omega(x,c)\ \lor\ \pi(r)\neq 0],
\]
where \(\omega = f_k\) in \(\mathsf{Real}_n\), \(\omega=g\) in \(\mathsf{Rand}_n\).

By definition of TV and data processing over event \(H_x:=\{y: h(x,y)=1\}\):
\[
\left|\Pr_{y\sim P_{F,A,\theta,k}^{(n)}(\cdot\mid x)}[H_x] - \Pr_{y\sim P_k^{\star,(n)}(\cdot\mid x)}[H_x]\right|
\le
\mathrm{TV}\!\left(P_{F,A,\theta,k}^{(n)}(\cdot\mid x),P_k^{\star,(n)}(\cdot\mid x)\right).
\]
Since \(P_k^{\star,(n)}(H_x\mid x)=0\), we get:
\[
\Pr_{y\sim P_{F,A,\theta,k}^{(n)}(\cdot\mid x)}[H_x]
\le
\mathrm{TV}\!\left(P_{F,A,\theta,k}^{(n)}(\cdot\mid x),P_k^{\star,(n)}(\cdot\mid x)\right).
\]
Averaging over \(x\sim D_{X,n}\):
\[
\rho_n^{\mathsf{Real}} := \Pr_{x,y}[h_k(x,y)=1 \text{ after filtering}] \le \varepsilon_0.
\]

In the random-function world, \(F_n\) has no hidden-feature structure to exploit.
Standard unpredictability for black-box PPT interaction with random labels implies:
\[
\rho_n^{\mathsf{Rand}} \ge \mu_0 - \mathrm{negl}(n).
\]

Therefore, for infinitely many \(n\):
\[
|\rho_n^{\mathsf{Rand}}-\rho_n^{\mathsf{Real}}|
\ge
\mu_0-\varepsilon_0-\mathrm{negl}(n).
\]

Taking \(\varepsilon_0 \le \mu_0/4\), for large enough \(n\):
\[
|\rho_n^{\mathsf{Rand}}-\rho_n^{\mathsf{Real}}|
\ge
\mu_0/2.
\]

## Explicit Distinguisher Construction

Construct PPT distinguisher \(\mathcal{D}\) with oracle access to \(\mathcal{O}\in\{f_k,g\}\):

1. Instantiate the generator/filtering game where harmfulness oracle uses \(\mathcal{O}(x\|c)\).
2. Run \(F_n\) with its allowed black-box sample budget \(q(n)\) per prompt.
3. Sample \(N(n)=\Theta(\mu_0^{-2}\log n)\) prompts from \(D_{X,n}\), generate filtered outputs, estimate residual harmful rate \(\widehat{\rho}\).
4. Output `PRF` iff \(\widehat{\rho} < (\mu_0/2)\), else output `RAND`.

By Hoeffding concentration:
\[
\Pr\big[|\widehat{\rho}-\rho_n|>\mu_0/8\big] \le 2\exp\!\left(-2N(n)(\mu_0/8)^2\right)\le n^{-3}.
\]
Hence:
\[
\mathrm{Adv}_{\mathcal{D}}^{\mathrm{PRF}}(n)
\ge
\frac{\mu_0}{4} - \mathrm{negl}(n).
\]
This is non-negligible, contradicting PRF security.

## Reduction Loss and Final Bound

Let \(q(n)\) be filter queries per prompt and \(N(n)\) prompts used by \(\mathcal{D}\).
Runtime overhead is polynomial:
\[
\mathrm{Time}_{\mathcal{D}}(n) = \mathrm{poly}(n, q(n), N(n)).
\]
Advantage loss is additive concentration slack \(\delta_n\) and simulator overhead negl term:
\[
\mathrm{Adv}_{\mathcal{D}}^{\mathrm{PRF}}(n)
\ge
\mu_0 - \varepsilon_0 - \delta_n - \mathrm{negl}(n).
\]
Thus any infinitely-often PPT filter with \(\varepsilon_0 < \mu_0\) yields a PRF break. Therefore such a filter cannot exist under PRF hardness.

## Interpretation for Neural Models

This reduction should be interpreted as a structural lower-bound template, not as a claim that real LLMs contain literal cryptographic keys.

- **Computationally hidden harmful feature** corresponds to latent harm-relevant structure that is not directly exposed to bounded filtering interfaces.
- In neural terms, this can reflect **deep feature entanglement**, **emergent circuits**, or **underspecified latent features** where harmfulness depends on distributed patterns.
- The claim is worst-case: if alignment preserves support and filtering is bounded, residual harmful mass can persist when harmful structure is computationally difficult to infer.
