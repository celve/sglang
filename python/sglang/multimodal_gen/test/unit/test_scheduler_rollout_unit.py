# SPDX-License-Identifier: Apache-2.0
"""Unit tests for flow SDE sampling alignment with FlowGRPO reference."""

import math
import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines.patches.flow_matching_with_logprob import (
    sde_step_with_logprob,
)


class _DummyScheduler:
    """Minimal scheduler mock for sde_step_with_logprob().

    Provides sigmas, timesteps, and index_for_timestep() matching
    FlowMatchEulerDiscreteScheduler conventions.
    """

    def __init__(self):
        self.sigmas = torch.tensor(
            [1.0, 0.8, 0.5, 0.3, 0.1, 0.0], dtype=torch.float32
        )
        self.timesteps = torch.tensor(
            [1000.0, 800.0, 500.0, 300.0, 100.0], dtype=torch.float32
        )

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()


def _flowgrpo_sde_step_with_logprob(
    *,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    variance_noise: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    sigma_max: float,
    noise_level: float,
    sde_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim from FlowGRPO sd3_sde_with_logprob.py ``sde_step_with_logprob``.

    Returns (prev_sample, log_prob, prev_sample_mean, noise_std_dev).
    ``sigma`` / ``sigma_prev`` follow FlowGRPO convention (current / next).
    """
    model_output = model_output.float()
    sample = sample.float()

    dt = sigma_prev - sigma

    if sde_type == "sde":
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
            * noise_level
        )
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample = prev_sample_mean + noise_std_dev * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, noise_std_dev


class TestFlowGRPOStepAlignmentUnit(unittest.TestCase):
    """Verify sglang's sde_step_with_logprob matches FlowGRPO reference."""

    def test_single_step_matches_flowgrpo_reference(self):
        """prev_sample_mean and log_prob must match FlowGRPO for SDE and CPS."""
        scheduler = _DummyScheduler()
        # timestep=500 -> index=2 -> sigma=0.5, sigma_prev=sigmas[3]=0.3
        timestep = torch.tensor(500.0)
        shape = (1, 16, 1, 32, 32)
        noise_level = 0.5
        atol = 1e-6

        for sde_type in ("sde", "cps"):
            for seed in (0, 1, 2, 3):
                with self.subTest(sde_type=sde_type, seed=seed):
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    model_output = torch.randn(
                        shape, generator=g, dtype=torch.float32
                    )
                    sample = torch.randn(shape, generator=g, dtype=torch.float32)
                    variance_noise = torch.randn(
                        shape, generator=g, dtype=torch.float32
                    )

                    # Reference result
                    ref_prev, ref_log_prob, ref_mean, ref_noise_std = (
                        _flowgrpo_sde_step_with_logprob(
                            model_output=model_output,
                            sample=sample,
                            variance_noise=variance_noise,
                            sigma=torch.tensor(0.5),
                            sigma_prev=torch.tensor(0.3),
                            sigma_max=scheduler.sigmas[1].item(),
                            noise_level=noise_level,
                            sde_type=sde_type,
                        )
                    )

                    # sglang result — pass ref's prev_sample so both use same sample
                    sgl_prev, sgl_log_prob, sgl_mean, sgl_std_dev_t = (
                        sde_step_with_logprob(
                            scheduler,
                            model_output=model_output,
                            timestep=timestep,
                            sample=sample,
                            noise_level=noise_level,
                            prev_sample=ref_prev,
                            sde_type=sde_type,
                        )
                    )

                    errs = {
                        "prev_sample": float(
                            (sgl_prev - ref_prev).abs().max().item()
                        ),
                        "prev_sample_mean": float(
                            (sgl_mean - ref_mean).abs().max().item()
                        ),
                        "log_prob": float(
                            (sgl_log_prob - ref_log_prob).abs().max().item()
                        ),
                    }

                    for name, err in errs.items():
                        self.assertLessEqual(
                            err,
                            atol,
                            msg=f"{sde_type} seed={seed} {name} max_abs={err:.9f}",
                        )


if __name__ == "__main__":
    unittest.main()
