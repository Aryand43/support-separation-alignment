# Transformer-Compatible Interpretation

This note maps the abstract framework to transformer-era alignment pipelines.

## Mapping of Terms

- **Messages / responses** \(y\) → autoregressive token sequences produced through attention propagation.
- **Computationally hidden harmful feature** → distributed activation pattern or circuit not linearly decodable from shallow probes.
- **Alignment operator** \(A\) → gradient-based policy update changing probabilities while preserving support.

## Support-Preserving Alignment Form

We model aligned policies as:
\[
\widetilde P(y\mid x)\propto P_\theta(y\mid x)\exp(\beta r(x,y)),
\]
where \(r(x,y)\) is learned reward/preference score.

This covers common methods:

- RLHF/PPO (policy gradient + KL regularization),
- DPO/IPO-style preference objectives,
- reward reweighting and rejection-sampling style updates with finite temperature.

As long as the update does not hard-zero nonzero-probability responses, support persists.

## Why Hidden Features Matter

In deep models, harmfulness can depend on entangled latent features that are:

- not explicitly represented in output logits as separable coordinates,
- weakly correlated with easily queried observables,
- recoverable only with substantial inference/modeling effort or unavailable supervision.

This mirrors the worst-case hidden-feature constructions used for lower bounds.

## Scope

This interpretation is structural: it does not claim real models literally implement cryptographic primitives.
It claims that support preservation plus bounded filtering can induce residual-risk floors when harm-relevant features are not directly exposed.
