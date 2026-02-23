from __future__ import annotations

from models.ideal_eliminator import IdealEliminator


class TrapdoorFilter(IdealEliminator):
    """
    Gold-standard unattainable filter with trapdoor harmfulness access.
    """

    def __init__(self, base_generator, model=None, seed: int = 0):
        super().__init__(
            base_model=model if model is not None else base_generator,
            harmful_oracle=base_generator.is_harmful_with_trapdoor,
            seed=seed,
        )
