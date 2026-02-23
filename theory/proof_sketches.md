# Proof Sketches

## Theorem 1 (Structural Support Persistence)

Take any \(x\) and any harmful \(y\in H_x\) with \(P_\theta(y\mid x)>0\). By support-preservation,
\((A P_\theta)(y\mid x)>0\). Summing over all harmful \(y\) in \(H_x\) gives
\((A P_\theta)(H_x\mid x) > 0\). Therefore harmful support cannot be fully removed by any support-preserving operator.

## Theorem 2 (Computational Separation)

Construct \(h_k\) using a PRF-like predicate keyed by secret \(k\). The base model places substantial mass on outputs
whose harmfulness depends on PRF agreement. The ideal eliminator knows \(k\) and removes exactly those outputs.
A bounded filter with only black-box samples cannot recover the keyed predicate beyond negligible advantage under the
one-way function assumption. If it could drive TV distance to ideal below \(\varepsilon_0\), we could build a distinguisher
against PRF pseudorandomness, violating hardness.

## Theorem 3 (Residual Harmful Mass Lower Bound)

Combine Theorem 1 (harmful support persists under alignment) with Theorem 2 (bounded filters cannot emulate trapdoor
elimination). This yields a non-zero residual floor \(c\) on harmful mass for some prompts. Empirically, this appears as
a plateau: increasing bounded compute reduces harm initially but cannot cross the hardness-induced floor.
