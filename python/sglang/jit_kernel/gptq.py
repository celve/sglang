from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, is_hip_runtime, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gptq_module() -> Module:
    extra_ldflags = ["-lhipblas"] if is_hip_runtime() else ["-lcublas"]
    return load_jit(
        "gptq",
        cuda_files=["gemm/gptq/gptq_kernel.cuh"],
        cuda_wrappers=[
            ("gptq_gemm", "gptq_gemm"),
            ("gptq_shuffle", "gptq_shuffle"),
        ],
        extra_ldflags=extra_ldflags,
    )


def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
) -> torch.Tensor:
    M, K = a.shape[0], a.shape[1]
    N = b_q_weight.shape[1]
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    temp_dq = torch.empty(
        (b_q_weight.shape[0] * 32 // bit, N), dtype=a.dtype, device=a.device
    )

    # Handle empty g_idx (meta device in torch.compile or empty tensor)
    if b_g_idx.device.type == "meta":
        b_g_idx = torch.empty(0, dtype=torch.int32, device=a.device)

    module = _jit_gptq_module()
    module.gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, c, temp_dq,
        use_shuffle, bit,
    )
    return c


def gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int,
) -> None:
    if q_perm.device.type == "meta" or q_perm.numel() == 0:
        q_perm = torch.empty(0, dtype=torch.int32, device=q_weight.device)

    module = _jit_gptq_module()
    module.gptq_shuffle(q_weight, q_perm, bit)
