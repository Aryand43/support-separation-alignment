# Positioning and Novelty

## Related Impossibility Lines

1. **Social-choice style impossibility**  
   Prior work studies conflicting normative constraints and aggregation impossibility.

2. **Harm-specification impossibility**  
   Prior work highlights underspecification and incomplete reward/safety objectives.

3. **Filtering hardness results**  
   Prior work gives specific hardness barriers for moderation or detector pipelines.

## What This Repository Adds

1. **Unified support-preserving framework**  
   A single operator-level abstraction captures RLHF/PPO, DPO-like updates, and reward reweighting.

2. **Residual harmful mass floor constant**  
   Explicit lower-bound viewpoint on persistent residual risk under bounded filtering.

3. **Alignment-filtering computational reduction**  
   A PRF-based reduction tailored to post-alignment filtering, framed via hidden harmful features.

4. **Information-theoretic strengthening**  
   SQ/VC-style lower bound showing residual floors without cryptographic assumptions.

5. **Empirical scaling protocol**  
   Security-parameter and query-budget sweeps with black-box, white-box, and SQ-style filters.

## Claim Boundaries

- We do **not** claim alignment is impossible.
- We claim support-preserving alignment plus bounded filtering can leave nonzero residual risk floors.
- White-box access helps only when it exposes harm-relevant sufficient statistics or supervision.
