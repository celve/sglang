# SPDX-License-Identifier: Apache-2.0
"""Flow-matching rollout step utilities for log-prob computation."""

import math
from typing import Any, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


def _as_timestep_tensor(
    timestep: Union[float, torch.Tensor], batch_size: int, device: torch.device
) -> torch.Tensor:
    """Normalize timestep input to a 1D tensor on the target device."""
    if torch.is_tensor(timestep):
        ts = timestep.to(device=device)
    else:
        ts = torch.tensor([timestep], device=device)

    if ts.ndim == 0:
        ts = ts.view(1)
    else:
        ts = ts.view(-1)

    # Broadcast scalar timestep to match batch size.
    if ts.numel() == 1 and batch_size > 1:
        ts = ts.repeat(batch_size)
    return ts


def sde_step_with_logprob(
    self: Any,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
    sde_type: str = "sde",
    use_sde_solver: bool = False,
):
    """Run one rollout step and compute per-sample log_prob.

    sde_type="sde" (aliases: "flow", "flux_flow") uses the Gaussian transition objective.
    sde_type="cps" uses the simplified CPS objective.
    sde_type="dance"/"flux_dance" uses the DanceGRPO formulation.
    """
    sample_dtype = sample.dtype
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    batch_size = sample.shape[0]
    timestep_tensor = _as_timestep_tensor(timestep, batch_size, sample.device)
    step_indices = torch.tensor(
        [self.index_for_timestep(t.to(self.timesteps.device)) for t in timestep_tensor],
        device=sample.device,
        dtype=torch.long,
    )
    prev_step_indices = (step_indices + 1).clamp_max(len(self.sigmas) - 1)
    step_indices = step_indices.to(device=self.sigmas.device)
    prev_step_indices = prev_step_indices.to(device=self.sigmas.device)

    sigma = self.sigmas[step_indices].to(sample.device).to(sample.dtype)
    sigma_prev = self.sigmas[prev_step_indices].to(sample.device).to(sample.dtype)
    sigma = sigma.view(-1, *([1] * (sample.ndim - 1)))
    sigma_prev = sigma_prev.view(-1, *([1] * (sample.ndim - 1)))
    sigma_max = self.sigmas[min(1, len(self.sigmas) - 1)].to(
        device=sample.device, dtype=sample.dtype
    )
    dt = sigma_prev - sigma

    # Resolve aliases
    sde_type = {"flux_flow": "sde", "flow": "sde"}.get(sde_type, sde_type)

    if sde_type == "sde":
        denom_sigma = 1 - torch.where(sigma == 1, sigma_max, sigma)
        std_dev_t = torch.sqrt((sigma / denom_sigma).clamp_min(1e-12)) * noise_level
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        sqrt_neg_dt = torch.sqrt((-dt).clamp_min(1e-12))
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * sqrt_neg_dt * variance_noise

        std = (std_dev_t * sqrt_neg_dt).clamp_min(1e-12)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std**2))
            - torch.log(std)
            - torch.log(torch.sqrt(torch.as_tensor(2 * math.pi, device=std.device)))
        )
    elif sde_type == "cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        sigma_delta = (sigma_prev**2 - std_dev_t**2).clamp_min(0.0)
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_delta)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # Keep the same simplified cps objective used in the original patch.
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
    elif sde_type in ("dance", "flux_dance"):
        eta = noise_level
        dsigma = sigma_prev - sigma  # negative
        delta_t = (sigma - sigma_prev).clamp(min=1e-12)  # positive
        std_dev_t = eta * torch.sqrt(delta_t)

        pred_original = sample - sigma * model_output
        prev_sample_mean = sample + dsigma * model_output

        if use_sde_solver:
            score_estimate = -(sample - pred_original * (1 - sigma)) / (
                sigma**2 + 1e-12
            )
            log_term = -0.5 * eta**2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * std_dev_t**2 + 1e-12)
            - torch.log(std_dev_t + 1e-12)
            - 0.5
            * torch.log(torch.as_tensor(2 * math.pi, device=std_dev_t.device))
        )
    else:
        raise ValueError(
            f"Unsupported sde_type: {sde_type}. "
            "Valid types: sde, cps, dance, flux_dance, flow, flux_flow."
        )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample.to(sample_dtype), log_prob, prev_sample_mean, std_dev_t
