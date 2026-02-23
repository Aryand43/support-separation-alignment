# Support-Separation Theorems

Let \(X\) be a prompt space, \(Y\) a response space, and \(h: X \times Y \to \{0,1\}\) a harmful predicate.
Let \(P_\theta(\cdot \mid x)\) be a base conditional distribution.

Define the harmful set:
\[
H_x := \{y \in Y : h(x,y) = 1\}.
\]

Let \(A\) be a support-preserving alignment operator:
\[
P_\theta(y\mid x) > 0 \implies (A P_\theta)(y\mid x) > 0.
\]

Define the ideal eliminator:
\[
P^\star(y\mid x) \propto P_\theta(y\mid x)\mathbf{1}[h(x,y)=0].
\]

## Theorem 1 (Universal Support Persistence)

Let \(A\) satisfy absolute continuity for each prompt:
\[
(A P_\theta)(\cdot\mid x) \ll P_\theta(\cdot\mid x).
\]
If there exists \(x\in X\) with \(P_\theta(H_x\mid x)>0\) and the Radon--Nikodym derivative
\(\frac{d(A P_\theta)(\cdot\mid x)}{dP_\theta(\cdot\mid x)}\) is strictly positive \(P_\theta\)-a.e. on \(H_x\), then:
\[
(A P_\theta)(H_x\mid x) > 0.
\]

Equivalently, any strictly positive reweighting over existing support cannot eliminate harmful support.

## Theorem 2 (Computational Separation)

Assume one-way functions exist. Then there exists a keyed family of harmful predicates \(h_k\) and induced generators \(P_{\theta,k}\) such that for some constant \(\varepsilon_0>0\):

For every probabilistic polynomial-time filter \(F\) with black-box sampling access to \(P_{\theta,k}\), there exists infinitely many security parameters \(n\) for which:
\[
\mathrm{TV}\!\left(P_{F,A,\theta,k}(\cdot\mid x),\; P_k^\star(\cdot\mid x)\right) \ge \varepsilon_0.
\]

Hence no bounded black-box filter can approximate ideal support elimination below \(\varepsilon_0\).

## Theorem 2b (White-Box / No-Trapdoor Limitation)

Under the same hardness assumptions, granting polynomial-time access to aligned model internals (e.g., logits, gradients, parameters) but not to trapdoor key material still does not, in general, permit approximation of \(P^\star\) below a constant TV threshold.

Formally, there exists \(\varepsilon_{\mathrm{wb}}>0\) such that for every polynomial-time no-trapdoor white-box filter \(F_{\mathrm{wb}}\):
\[
\mathrm{TV}\!\left(P_{F_{\mathrm{wb}},A,\theta,k}(\cdot\mid x),\; P_k^\star(\cdot\mid x)\right)\ge \varepsilon_{\mathrm{wb}}
\]
for infinitely many security parameters.

## Theorem 3 (Residual Harmful Mass Lower Bound)

Under the same assumptions, there exists a constant \(c>0\) such that for every support-preserving alignment \(A\) and every polynomial-time black-box filter \(F\), there is an \(x\in X\) with:
\[
P_{F,A,\theta}(H_x\mid x) \ge c.
\]

Thus residual harmful mass is bounded away from zero.
