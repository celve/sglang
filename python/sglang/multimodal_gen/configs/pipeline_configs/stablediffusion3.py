# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 pipeline configuration."""

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import StableDiffusion3TransformerConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import TextEncoderArchConfig
from sglang.multimodal_gen.configs.models.encoders.clip import (
    CLIPTextArchConfig,
    CLIPTextConfig,
)
from sglang.multimodal_gen.configs.models.encoders.t5 import (
    T5ArchConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
    preprocess_text,
)


def sd3_clip_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract pre-final hidden state for SD3 CLIP encoders."""
    if outputs.hidden_states is None:
        raise ValueError(
            "SD3 CLIP postprocessing requires hidden_states from encoder output."
        )
    return outputs.hidden_states[-2]


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class SD3CLIPTextArchConfig(CLIPTextArchConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs.update(
            {
                "max_length": self.text_len,
                "padding": "max_length",
            }
        )


@dataclass
class SD3CLIPTextConfig(CLIPTextConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=SD3CLIPTextArchConfig)


@dataclass
class SD3T5ArchConfig(T5ArchConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.tokenizer_kwargs.update({"max_length": 256})


@dataclass
class SD3T5Config(T5Config):
    arch_config: TextEncoderArchConfig = field(default_factory=SD3T5ArchConfig)


@dataclass
class StableDiffusion3PipelineConfig(SpatialImagePipelineConfig):
    """Configuration for SD3 image generation pipeline.

    This config intentionally relies on SD3-specific encoder configs to provide
    tokenizer kwargs, instead of stage-level tokenizer overrides.
    """

    task_type: ModelTaskType = ModelTaskType.T2I

    dit_config: DiTConfig = field(default_factory=StableDiffusion3TransformerConfig)
    vae_config: VAEConfig = field(default_factory=StableDiffusion3VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            SD3CLIPTextConfig(),
            SD3CLIPTextConfig(),
            SD3T5Config(),
        )
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16", "fp32")
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (
            preprocess_text,
            preprocess_text,
            preprocess_text,
        )
    )

    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict], torch.Tensor], ...
    ] = field(
        default_factory=lambda: (
            sd3_clip_postprocess_text,
            sd3_clip_postprocess_text,
            t5_postprocess_text,
        )
    )

    should_use_guidance: bool = False
    guidance_scale: float = 7.0
    use_precision_specific_weights: bool = True
    vae_model_name: str = "diffusion_pytorch_model"

    def __post_init__(self) -> None:
        configs = list(self.text_encoder_configs)
        configs[0].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[1].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[2].update_model_arch({"_class_name": "T5EncoderModel"})
        self.text_encoder_configs = tuple(configs)

    def get_encoder_attention_mask(self, encoder_index, text_inputs, device):
        """SD3 does not pass attention masks to its text encoders."""
        return None

    def extract_pooled_output(self, encoder_index, encoder_outputs):
        """SD3 CLIP encoders (indices 0, 1) produce pooled outputs."""
        if encoder_index <= 1:
            return encoder_outputs.pooler_output
        return None

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.pooled_embeds[0] if batch.pooled_embeds else None
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.neg_pooled_embeds[0] if batch.neg_pooled_embeds else None
            )
        }

    # SD3 image latents are spatial (B, C, H, W), not video-like (B, C, T, H, W).
    def prepare_latent_shape(self, batch, batch_size, num_frames):  # noqa: ARG002
        spatial_ratio = self.vae_config.arch_config.spatial_compression_ratio
        in_channels = self.dit_config.arch_config.in_channels
        return (
            batch_size,
            in_channels,
            batch.height // spatial_ratio,
            batch.width // spatial_ratio,
        )
