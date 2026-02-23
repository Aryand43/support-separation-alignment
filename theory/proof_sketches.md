# Proof Sketches

## Theorem 1 (Universal Support Persistence)

Use Radon--Nikodym derivatives. If \((A P_\theta)(\cdot\mid x)\ll P_\theta(\cdot\mid x)\), then
\[
(A P_\theta)(H_x\mid x)=\int_{H_x} g_x(y)\,dP_\theta(y\mid x).
\]
If \(P_\theta(H_x\mid x)>0\) and \(g_x>0\) almost everywhere on \(H_x\), the integral is strictly positive.
Hence harmful support mass persists for any strictly positive reweighting operator on measurable spaces.

## Theorem 2 (Computational Separation)

Construct \(h_k\) via keyed PRF agreement and define \(P_k^\star\) as exact harmful-support eliminator.
If a PPT filter achieved TV\(<\varepsilon_0\) to \(P_k^\star\), then residual harmful mass under filtering would be
uniformly tiny. Replacing PRF by random function yields a hybrid gap in residual harmful mass; this gap is converted
into a non-negligible PRF distinguishing advantage, contradicting PRF security.

## Theorem 3 (Residual Harmful Mass Lower Bound)

Combine Theorem 1 (harmful support persists under alignment) with Theorem 2 (bounded filters cannot emulate trapdoor
elimination). This yields a non-zero residual floor \(c\) on harmful mass for some prompts. Empirically, this appears as
a plateau: increasing bounded compute reduces harm initially but cannot cross the hardness-induced floor.
