# SPDX-License-Identifier: Apache-2.0
"""Factory for creating stochastic SDE samplers."""

from sglang.multimodal_gen.runtime.models.sampler.base import BaseStochasticSampler
from sglang.multimodal_gen.runtime.models.sampler.cps_sampler import CPSSampler
from sglang.multimodal_gen.runtime.models.sampler.dance_sampler import DanceSampler
from sglang.multimodal_gen.runtime.models.sampler.sde_sampler import SDESampler

SAMPLER_REGISTRY: dict[str, type[BaseStochasticSampler]] = {
    "sde": SDESampler,
    "cps": CPSSampler,
    "dance": DanceSampler,
}


def create_stochastic_sampler(
    sde_type: str,
    eta: float = 1.0,
    sde_indices: set[int] | None = None,
) -> BaseStochasticSampler:
    """Create a stochastic sampler by name.

    Args:
        sde_type: One of "sde", "cps", or "dance".
        eta: Noise level controlling stochasticity.
        sde_indices: Timestep indices for SDE steps; None means all steps.

    Returns:
        A configured BaseStochasticSampler instance.
    """
    if sde_type not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sde_type {sde_type!r}. Choose from {list(SAMPLER_REGISTRY)}"
        )
    cls = SAMPLER_REGISTRY[sde_type]
    return cls(eta=eta, sde_indices=sde_indices)
