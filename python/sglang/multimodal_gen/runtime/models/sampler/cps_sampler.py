# SPDX-License-Identifier: Apache-2.0
"""Coefficient-Preserving Sampling formulation."""

import math

import torch

from sglang.multimodal_gen.runtime.models.sampler.base import BaseStochasticSampler


class CPSSampler(BaseStochasticSampler):
    """Coefficient-Preserving Sampling formulation."""

    def _compute_mean_and_std(self, noise_pred, sample, sigma, sigma_next):
        std_dev = sigma_next * math.sin(self.eta * math.pi / 2)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = pred_original * (1 - sigma_next) + noise_estimate * torch.sqrt(
            sigma_next**2 - std_dev**2
        )
        return prev_sample_mean, std_dev

    def _compute_log_prob(self, prev_sample, prev_sample_mean, std_dev):
        # Simplified (constants removed per CPS formulation)
        log_prob = -((prev_sample - prev_sample_mean) ** 2)
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
