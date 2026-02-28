"""Pydantic request/response models for RL endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RLGenerateRequest(BaseModel):
    """Request schema for /v1/rl/generate."""

    prompts: List[str] = Field(..., min_length=1)
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = 1024

    # Rollout parameters
    sde_type: str = "sde"
    noise_level: float = 0.7
    use_sde_solver: bool = False
    sde_indices: Optional[List[int]] = None

    # What to return
    return_trajectory_latents: bool = True
    return_trajectory_timesteps: bool = True
    return_trajectory_log_probs: bool = True
    return_trajectory_decoded: bool = False
    return_decoded: bool = False
    response_format: str = Field(
        default="safetensors",
        description="Response format: 'safetensors' for binary blob, 'json_meta' for debug info only.",
    )


class EncodePromptRequest(BaseModel):
    """Request schema for /v1/rl/encode_prompt."""

    prompts: List[str] = Field(..., min_length=1)
    response_format: str = Field(
        default="safetensors",
        description="Response format: 'safetensors' for binary blob, 'json_meta' for debug info only.",
    )
