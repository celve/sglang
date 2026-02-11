# SPDX-License-Identifier: Apache-2.0
"""DanceGRPO SDE formulation with score-based correction."""

import math

import torch

from sglang.multimodal_gen.runtime.models.sampler.base import BaseStochasticSampler


class DanceSampler(BaseStochasticSampler):
    """DanceGRPO SDE formulation with score-based correction."""

    def _compute_mean_and_std(self, noise_pred, sample, sigma, sigma_next):
        dsigma = sigma_next - sigma
        delta_t = (sigma - sigma_next).clamp(min=1e-12)
        std_dev = self.eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * noise_pred
        prev_sample_mean = sample + dsigma * noise_pred

        # Score correction term
        score_estimate = -(sample - pred_original * (1 - sigma)) / (sigma**2 + 1e-12)
        log_term = -0.5 * (self.eta**2) * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

        return prev_sample_mean, std_dev

    def _compute_log_prob(self, prev_sample, prev_sample_mean, std_dev):
        log_prob = (
            -((prev_sample - prev_sample_mean) ** 2) / (2 * std_dev**2 + 1e-12)
            - torch.log(std_dev + 1e-12)
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
