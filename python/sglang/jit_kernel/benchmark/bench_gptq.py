import os

import torch
import triton
import triton.testing

from sglang.jit_kernel.gptq import gptq_gemm as jit_fn
from sglang.jit_kernel.gptq import gptq_shuffle as jit_shuffle

try:
    from sgl_kernel import gptq_gemm as aot_fn
    from sgl_kernel import gptq_shuffle as aot_shuffle

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Fixed problem dimensions
K = 4096
N = 4096
BIT = 4
GROUP_SIZE = 128
USE_SHUFFLE = True

_cache = {}


def _get_inputs(M):
    if M not in _cache:
        pack_factor = 32 // BIT
        groups = K // GROUP_SIZE

        a = torch.randn((M, K), dtype=torch.float16, device="cuda")
        b_q_weight = torch.randint(
            0, 2**31, (K // pack_factor, N), dtype=torch.int32, device="cuda"
        )
        b_gptq_qzeros = torch.randint(
            0, 2**31, (groups, N // pack_factor), dtype=torch.int32, device="cuda"
        )
        b_gptq_scales = torch.randn(
            (groups, N), dtype=torch.float16, device="cuda"
        )
        b_g_idx = torch.empty(0, dtype=torch.int32, device="cuda")

        # Shuffle weights for exllama path
        q_perm = torch.empty(0, dtype=torch.int32, device="cuda")
        jit_shuffle(b_q_weight, q_perm, BIT)

        _cache[M] = (a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx)
    return _cache[M]


def check_correctness():
    if not AOT_AVAILABLE:
        print("sgl_kernel AOT not available, skipping correctness check")
        return

    M = 32
    pack_factor = 32 // BIT
    groups = K // GROUP_SIZE

    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b_q_weight_jit = torch.randint(
        0, 2**31, (K // pack_factor, N), dtype=torch.int32, device="cuda"
    )
    b_q_weight_aot = b_q_weight_jit.clone()
    b_gptq_qzeros = torch.randint(
        0, 2**31, (groups, N // pack_factor), dtype=torch.int32, device="cuda"
    )
    b_gptq_scales = torch.randn((groups, N), dtype=torch.float16, device="cuda")
    b_g_idx = torch.empty(0, dtype=torch.int32, device="cuda")
    q_perm = torch.empty(0, dtype=torch.int32, device="cuda")

    jit_shuffle(b_q_weight_jit, q_perm, BIT)
    aot_shuffle(b_q_weight_aot, q_perm, BIT)

    out_jit = jit_fn(
        a, b_q_weight_jit, b_gptq_qzeros, b_gptq_scales, b_g_idx, USE_SHUFFLE, BIT
    )
    out_aot = aot_fn(
        a, b_q_weight_aot, b_gptq_qzeros, b_gptq_scales, b_g_idx, USE_SHUFFLE, BIT
    )
    max_diff = torch.mean(torch.abs(out_jit - out_aot)) / torch.mean(
        torch.abs(out_aot)
    )
    assert max_diff < 0.04, f"relative mean error {max_diff:.6f} exceeds 0.04"
    print(f"Correctness check passed (JIT vs AOT, rel_mean_err={max_diff:.6f})")


if IS_CI:
    m_range = [1, 8, 32, 128]
else:
    m_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

if AOT_AVAILABLE:
    line_vals = ["jit", "aot"]
    line_names = ["JIT Kernel", "AOT Kernel"]
    styles = [("blue", "-"), ("green", "-")]
else:
    line_vals = ["jit"]
    line_names = ["JIT Kernel"]
    styles = [("blue", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=m_range,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name=f"gptq-gemm-K{K}-N{N}-{BIT}bit",
        args={},
    )
)
def benchmark(M, provider):
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx = _get_inputs(M)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "jit":
        fn = lambda: jit_fn(
            a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, USE_SHUFFLE, BIT
        )
    elif provider == "aot":
        fn = lambda: aot_fn(
            a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, USE_SHUFFLE, BIT
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    check_correctness()
    benchmark.run(print_data=True)
