# SPDX-License-Identifier: Apache-2.0
"""Standard SDE formulation from flow-GRPO."""

import math

import torch

from sglang.multimodal_gen.runtime.models.sampler.base import BaseStochasticSampler


class SDESampler(BaseStochasticSampler):
    """Standard SDE formulation from flow-GRPO."""

    def _compute_mean_and_std(self, noise_pred, sample, sigma, sigma_next):
        dt = sigma_next - sigma
        std_dev_t = torch.sqrt(sigma / (1 - sigma.clamp(max=1 - 1e-8))) * self.eta
        prev_sample_mean = sample * (
            1 + std_dev_t**2 / (2 * sigma.clamp(min=1e-8)) * dt
        ) + noise_pred * (
            1 + std_dev_t**2 * (1 - sigma) / (2 * sigma.clamp(min=1e-8))
        ) * dt
        std_dev = std_dev_t * torch.sqrt(-dt)
        return prev_sample_mean, std_dev

    def _compute_log_prob(self, prev_sample, prev_sample_mean, std_dev):
        variance = std_dev**2 + 1e-12
        log_prob = (
            -((prev_sample - prev_sample_mean) ** 2) / (2 * variance)
            - torch.log(std_dev + 1e-12)
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
