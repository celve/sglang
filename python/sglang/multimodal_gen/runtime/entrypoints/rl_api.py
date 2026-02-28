"""RL-specific HTTP endpoints for diffusionrl integration.

Exposes /v1/rl/generate and /v1/rl/encode_prompt, returning raw tensors
(latent trajectories, log-probs, prompt embeddings) in safetensors format.
"""

import json
import time
import uuid

import torch

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse, Response

from sglang.multimodal_gen.runtime.entrypoints.rl_protocol import (
    EncodePromptRequest,
    RLGenerateRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.tensor_io import (
    tensors_to_json_meta,
    tensors_to_safetensors_bytes,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    EncodePromptReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

router = APIRouter(prefix="/v1/rl", tags=["rl"])


@router.post("/generate")
async def rl_generate(body: RLGenerateRequest, request: Request):
    """Generate latent trajectories with log-probs for RL training.

    Returns a safetensors binary blob containing:
      - latents: [B, C, H, W] or [B, C, T, H, W]  (final denoised latents)
      - trajectory_latents: [B, num_steps, C, ...]   (full trajectory)
      - trajectory_timesteps: [num_steps]             (sigma schedule)
      - trajectory_log_probs: [B, num_sde_steps]      (per-step log-probs)
    """
    server_args: ServerArgs = request.app.state.server_args
    request_id = f"rl_{uuid.uuid4()}"

    # Build sampling params with rollout enabled
    sp_kwargs = {
        "prompt": body.prompts[0],  # SamplingParams takes a single prompt
        "seed": body.seed,
        "rollout": True,
        "rollout_sde_type": body.sde_type,
        "rollout_noise_level": body.noise_level,
        "rollout_use_sde_solver": body.use_sde_solver,
        "return_trajectory_latents": body.return_trajectory_latents,
        "return_trajectory_decoded": body.return_trajectory_decoded,
        "save_output": False,
        "suppress_logs": True,
    }
    if body.height is not None:
        sp_kwargs["height"] = body.height
    if body.width is not None:
        sp_kwargs["width"] = body.width
    if body.num_frames is not None:
        sp_kwargs["num_frames"] = body.num_frames
    if body.num_inference_steps is not None:
        sp_kwargs["num_inference_steps"] = body.num_inference_steps
    if body.guidance_scale is not None:
        sp_kwargs["guidance_scale"] = body.guidance_scale
    if body.sde_indices is not None:
        sp_kwargs["rollout_sde_indices"] = body.sde_indices

    try:
        sp = build_sampling_params(request_id, **sp_kwargs)
    except Exception as e:
        logger.error("Failed to build sampling params: %s", e, exc_info=True)
        return ORJSONResponse(
            {"error": f"Invalid parameters: {e}"}, status_code=400
        )

    # Build and forward request
    try:
        req_obj = prepare_request(server_args, sampling_params=sp)
        start_time = time.monotonic()
        response = await async_scheduler_client.forward(req_obj)
        inference_time_s = time.monotonic() - start_time
    except Exception as e:
        logger.error("RL generate forward failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"error": f"Generation failed: {e}"}, status_code=500
        )

    # Check for errors
    if response.error:
        status = 400 if "sleeping" in str(response.error).lower() else 500
        return ORJSONResponse({"error": response.error}, status_code=status)

    # Extract tensors from OutputBatch
    tensors = {}
    metadata = {
        "contract_version": "v1",
        "request_id": request_id,
    }

    # Always set trajectory_format and timestep_type (required by validate_contract)
    if server_args.pipeline_config.task_type.is_image_gen():
        metadata["trajectory_format"] = "dense_latent"
    else:
        metadata["trajectory_format"] = "video_dense_latent"
    metadata["timestep_type"] = "sigma"

    # Trajectory latents (gated)
    if body.return_trajectory_latents and response.trajectory_latents is not None:
        traj = response.trajectory_latents
        tensors["trajectory_latents"] = traj
        metadata["num_steps"] = str(traj.shape[1] - 1) if traj.dim() >= 2 else "0"

    # Trajectory timesteps (gated)
    if body.return_trajectory_timesteps and response.trajectory_timesteps is not None:
        tensors["trajectory_timesteps"] = response.trajectory_timesteps

    # Trajectory log probs (gated)
    if body.return_trajectory_log_probs and response.trajectory_log_probs is not None:
        tensors["trajectory_log_probs"] = response.trajectory_log_probs

    # Trajectory decoded — all steps VAE-decoded (gated, expensive)
    if body.return_trajectory_decoded and response.trajectory_decoded is not None:
        tensors["trajectory_decoded"] = torch.stack(response.trajectory_decoded, dim=1)

    # Final decoded output for reward computation (gated, cheap)
    if body.return_decoded and response.output is not None:
        decoded = response.output
        if isinstance(decoded, list):
            decoded = decoded[0]
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.from_numpy(decoded)
        tensors["decoded_output"] = decoded

    if not tensors:
        return ORJSONResponse(
            {"error": "No output data returned. Check rollout/return flags."},
            status_code=500,
        )

    # Add batch/inference metadata
    if "trajectory_latents" in tensors:
        metadata["batch_size"] = str(tensors["trajectory_latents"].shape[0])
    if body.sde_indices is not None:
        metadata["sde_indices"] = json.dumps(body.sde_indices)
    if response.peak_memory_mb:
        metadata["peak_memory_mb"] = f"{response.peak_memory_mb:.1f}"
    metadata["inference_time_s"] = f"{inference_time_s:.3f}"

    # Serialize
    if body.response_format == "json_meta":
        return ORJSONResponse(tensors_to_json_meta(tensors, metadata))

    try:
        blob = tensors_to_safetensors_bytes(tensors, metadata)
    except Exception as e:
        logger.error("Tensor serialization failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"error": f"Serialization failed: {e}"}, status_code=500
        )

    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={"X-Tensor-Format": "safetensors"},
    )


@router.post("/encode_prompt")
async def encode_prompt(body: EncodePromptRequest, request: Request):
    """Encode text prompts into embeddings without running diffusion.

    Returns a safetensors blob with prompt_embeds, pooled_embeds, and
    attention masks for each prompt.
    """
    req = EncodePromptReqInput(prompts=body.prompts)

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        logger.error("Encode prompt failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"error": f"Encode prompt failed: {e}"}, status_code=500
        )

    if response.error:
        return ORJSONResponse({"error": response.error}, status_code=500)

    result = response.output
    if not isinstance(result, dict):
        return ORJSONResponse(
            {"error": "Unexpected response format from encoder"}, status_code=500
        )

    # Separate tensors from non-tensor metadata
    tensors = {}
    metadata = {"contract_version": "v1", "num_prompts": str(len(body.prompts))}

    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value
        else:
            metadata[key] = str(value)

    if not tensors:
        return ORJSONResponse(
            {"error": "No embeddings returned from encoder"}, status_code=500
        )

    if body.response_format == "json_meta":
        return ORJSONResponse(tensors_to_json_meta(tensors, metadata))

    try:
        blob = tensors_to_safetensors_bytes(tensors, metadata)
    except Exception as e:
        logger.error("Tensor serialization failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"error": f"Serialization failed: {e}"}, status_code=500
        )

    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={"X-Tensor-Format": "safetensors"},
    )
