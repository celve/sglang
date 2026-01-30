"""JIT-compiled gptq_marlin kernel manager.

Compiles only the template instantiations needed for a specific weight type,
rather than all ~225+ variants in the static kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module

# Maps ScalarType.id -> (wtype_define_name, human_name, cpp_scalar_type)
_WTYPE_MAP: dict[int, tuple[str, str]] = {}


def _init_wtype_map() -> None:
    global _WTYPE_MAP
    from sgl_kernel.scalar_types import (
        float4_e2m1f,
        float8_e4m3fn,
        uint4,
        uint4b8,
        uint8b128,
    )

    _WTYPE_MAP = {
        uint4.id: ("MARLIN_WTYPE_KU4", "ku4"),
        uint4b8.id: ("MARLIN_WTYPE_KU4B8", "ku4b8"),
        uint8b128.id: ("MARLIN_WTYPE_KU8B128", "ku8b128"),
        float4_e2m1f.id: ("MARLIN_WTYPE_KFE2M1F", "kfe2m1f"),
        float8_e4m3fn.id: ("MARLIN_WTYPE_KFE4M3FN", "kfe4m3fn"),
    }


# Map torch dtype -> CUDA C++ type name (matching utils.cuh: fp16_t = __half, bf16_t = __nv_bfloat16)
_DTYPE_TO_CPP = {
    torch.float16: "half",
    torch.bfloat16: "nv_bfloat16",
}


@cache_once
def _jit_gptq_marlin_module(wtype_id: int, dtype: torch.dtype) -> Module:
    """JIT-compile the gptq_marlin kernel for a specific weight type and compute dtype."""
    if not _WTYPE_MAP:
        _init_wtype_map()
    define_name, human_name = _WTYPE_MAP[wtype_id]
    cpp_type = _DTYPE_TO_CPP[dtype]
    return load_jit(
        "gptq_marlin",
        human_name,
        cpp_type,
        cuda_files=["marlin/gptq_marlin.cuh"],
        cuda_wrappers=[
            ("gptq_marlin_gemm", f"GptqMarlinGemm<{cpp_type}>::run"),
        ],
        extra_cuda_cflags=[
            "-DMARLIN_JIT",
            f"-D{define_name}",
            "--expt-extended-lambda",
        ],
    )


def jit_gptq_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: object,  # sglang ScalarType
    size_m: int,
    size_n: int,
    size_k: int,
    *,
    c: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    b_zeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """JIT-compiled gptq_marlin_gemm.

    Drop-in replacement for torch.ops.sgl_kernel.gptq_marlin_gemm,
    but compiles only the kernels needed for the given weight type.
    """
    device = a.device
    dtype = a.dtype

    # Output tensor
    if c is None:
        c = torch.empty((size_m, size_n), dtype=dtype, device=device)
    if size_m == 0:
        return c

    # SM count for c_tmp sizing
    dev_id = a.get_device()
    sms = torch.cuda.get_device_properties(dev_id).multi_processor_count

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce:
        max_m_block_size = (size_m + 16 - 1) // 16 * 16
        max_m_block_size = min(max_m_block_size, 64)
        max_thread_n = 256  # marlin::max_thread_n
        max_c_tmp_size = sms * max_m_block_size * max_thread_n
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Empty placeholders for optional tensors
    empty_opt = torch.empty(0, dtype=dtype, device=device)

    if global_scale is None:
        global_scale = empty_opt
    if b_zeros is None:
        b_zeros = empty_opt
    if g_idx is None:
        g_idx = empty_opt
    if perm is None:
        perm = empty_opt

    # Allocate a_tmp for act_order
    has_act_order = g_idx.numel() > 0 and perm.numel() > 0
    if has_act_order:
        a_tmp = torch.empty((size_m, size_k), dtype=dtype, device=device)
    else:
        a_tmp = empty_opt

    # JIT compile and call
    module = _jit_gptq_marlin_module(b_q_type.id, dtype)
    module.gptq_marlin_gemm(
        a,
        c,
        c_tmp,
        a_tmp,
        b_q_weight,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
    return c
