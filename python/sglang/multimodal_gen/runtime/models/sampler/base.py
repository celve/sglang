# SPDX-License-Identifier: Apache-2.0
"""Base class for stochastic SDE samplers with log-probability computation."""

from abc import ABC, abstractmethod

import torch


class BaseStochasticSampler(ABC):
    """Base class for stochastic SDE samplers with log-probability computation."""

    def __init__(self, eta: float = 1.0, sde_indices: set[int] | None = None):
        self.eta = eta
        self.sde_indices = sde_indices  # None = all steps are SDE
        self._last_log_prob: torch.Tensor | None = None

    def should_use_sde(self, step_index: int) -> bool:
        if self.sde_indices is None:
            return True
        return step_index in self.sde_indices

    def clear_last_output(self):
        self._last_log_prob = None

    @property
    def last_log_prob(self) -> torch.Tensor | None:
        return self._last_log_prob

    def compute_sde_step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """SDE step with log_prob. Returns prev_sample, sets self._last_log_prob."""
        noise_pred = noise_pred.float()
        sample = sample.float()

        # Expand sigma for broadcasting
        sigma_b = sigma.clone()
        sigma_next_b = sigma_next.clone()
        if sigma_b.dim() == 0:
            sigma_b = sigma_b.unsqueeze(0)
        if sigma_next_b.dim() == 0:
            sigma_next_b = sigma_next_b.unsqueeze(0)
        while sigma_b.dim() < sample.dim():
            sigma_b = sigma_b.unsqueeze(-1)
            sigma_next_b = sigma_next_b.unsqueeze(-1)

        # Delegate to subclass for mean, std, and log_prob formula
        prev_sample_mean, std_dev = self._compute_mean_and_std(
            noise_pred, sample, sigma_b, sigma_next_b
        )

        # Generate stochastic sample
        noise = torch.randn(
            sample.shape, dtype=sample.dtype, device=sample.device, generator=generator
        )
        prev_sample = prev_sample_mean + std_dev * noise

        # Compute log_prob and store
        self._last_log_prob = self._compute_log_prob(
            prev_sample.detach(), prev_sample_mean, std_dev
        )
        return prev_sample

    @abstractmethod
    def _compute_mean_and_std(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prev_sample_mean, std_dev) for the SDE transition."""
        ...

    @abstractmethod
    def _compute_log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        std_dev: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample log_prob [B] from the transition."""
        ...
