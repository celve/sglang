"""Tensor serialization utilities for RL endpoints.

Uses the safetensors format (already a transitive dependency via diffusers/transformers)
to efficiently serialize named tensors with string metadata over HTTP.
"""

import json
from typing import Dict, Optional

import torch
from safetensors.torch import save

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def tensors_to_safetensors_bytes(
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Serialize named tensors + metadata into a safetensors binary blob.

    Args:
        tensors: Mapping of tensor names to torch tensors (must be contiguous CPU).
        metadata: Optional string-to-string metadata dict embedded in the header.

    Returns:
        Raw bytes in safetensors format, suitable for an octet-stream HTTP response.
    """
    # Ensure all tensors are contiguous CPU tensors
    cpu_tensors: Dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for key '{name}', got {type(t)}")
        t = t.detach().cpu().contiguous()
        cpu_tensors[name] = t

    return save(cpu_tensors, metadata=metadata)


def tensors_to_json_meta(
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
) -> dict:
    """Return tensor shapes/dtypes and metadata as a JSON-serializable dict.

    Useful for debugging without transferring actual tensor data.
    """
    tensor_info = {}
    for name, t in tensors.items():
        tensor_info[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": t.numel(),
        }
    result: dict = {"tensors": tensor_info}
    if metadata:
        result["metadata"] = metadata
    return result
