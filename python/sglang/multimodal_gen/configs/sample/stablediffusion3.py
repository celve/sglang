# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 sampling parameters configuration."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class StableDiffusion3SamplingParams(SamplingParams):
    """Sampling parameters for StableDiffusion3."""

    # None → text_encoding uses zero uncond embeddings, matching diffusionrl training.
    negative_prompt: str | None = None
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    guidance_rescale: float = 0.0
