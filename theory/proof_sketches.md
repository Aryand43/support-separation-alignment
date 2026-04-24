# Proof Sketches

## Theorem 1 (Universal Support Persistence)

Use Radon--Nikodym derivatives. If $(A P_\theta)(\cdot\mid x)\ll P_\theta(\cdot\mid x)$, then
$$
(A P_\theta)(H_x\mid x)=\int_{H_x} g_x(y)\,dP_\theta(y\mid x).
$$
If $P_\theta(H_x\mid x)>0$ and $g_x>0$ almost everywhere on $H_x$, the integral is strictly positive.
Hence harmful support mass persists for any strictly positive reweighting operator on measurable spaces.

## Theorem 2 (Computational Separation)

Construct $h_k$ via keyed PRF agreement and define $P_k^\star$ as exact harmful-support eliminator.
If a PPT filter achieved $\mathrm{TV} < \varepsilon_0$ to $P_k^\star$, then residual harmful mass under filtering would be
uniformly tiny. Replacing PRF by random function yields a hybrid gap in residual harmful mass; this gap is converted
into a non-negligible PRF distinguishing advantage, contradicting PRF security.

## Theorem 3 (Information-Theoretic Residual Floor)

Use a fooling-set/SQ-dimension construction where multiple harmful-label worlds are nearly indistinguishable under at most
$m$ statistical queries. Since $A$ preserves support, harmful candidates remain in the aligned distribution; an SQ-limited
filter cannot identify all harmful regions, yielding residual error $\Omega\!\left(\frac{d}{m}\right)$ on a hard distribution.

## Theorem 4 (Residual Harmful Mass Lower Bound)

Combine Theorem 1 with either Theorem 2 (computational hardness) or Theorem 3 (information-theoretic SQ hardness).
Both routes imply a non-zero residual floor $c$ on harmful mass under bounded filtering. Empirically this appears as a
plateau: extra filtering compute helps initially but eventually saturates above zero unless stronger information is exposed.
