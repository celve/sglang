/*
 * TVM FFI entry point for JIT-compiled gptq_marlin kernel.
 * Replaces the original gptq_marlin_gemm() PyTorch function with
 * TensorView-based interface for JIT compilation via TVM FFI.
 */
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include "gptq_marlin_impl.cuh"

namespace {

template <typename scalar_t>
struct GptqMarlinGemm {
  static void run(
      tvm::ffi::TensorView a,             // [M, K] half or bf16
      tvm::ffi::TensorView c,             // [M, N] output (pre-allocated)
      tvm::ffi::TensorView c_tmp,         // float32 temp (or shape [0])
      tvm::ffi::TensorView a_tmp,         // [M, K] temp for act_order (or shape [0])
      tvm::ffi::TensorView b_q_weight,    // quantized weights
      tvm::ffi::TensorView b_scales,      // quantization scales
      tvm::ffi::TensorView global_scale,  // shape [0] if unused
      tvm::ffi::TensorView b_zeros,       // shape [0] if unused
      tvm::ffi::TensorView g_idx,         // shape [0] if unused
      tvm::ffi::TensorView perm,          // shape [0] if unused
      tvm::ffi::TensorView workspace,     // workspace for locks
      int64_t b_q_type_id,
      int64_t size_m, int64_t size_n, int64_t size_k,
      bool is_k_full, bool use_atomic_add,
      bool use_fp32_reduce, bool is_zp_float) {
    using namespace host;

    // Resolve device and stream from tensor `a`
    DLDevice device = a.device();
    int dev = device.device_id;
    cudaStream_t stream = LaunchKernel::resolve_device(device);

    // Get SM count
    int sms = -1;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    // Reconstruct ScalarType from id
    sglang::ScalarType b_q_type = sglang::ScalarType::from_id(b_q_type_id);

    // Derive act_order from g_idx/perm
    int64_t g_idx_len = (g_idx.dim() > 0) ? g_idx.size(g_idx.dim() - 1) : 0;
    int64_t perm_len = (perm.dim() > 0) ? perm.size(perm.dim() - 1) : 0;
    bool has_act_order = (g_idx_len > 0 && perm_len > 0);

    // Derive num_groups, group_size from b_scales
    int num_groups = static_cast<int>(b_scales.size(0));
    int group_size = -1;

    if (has_act_order) {
      if (is_k_full) {
        MARLIN_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
        MARLIN_CHECK(size_k % num_groups == 0,
                     "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
        group_size = static_cast<int>(size_k / num_groups);
      } else {
        group_size = 0;
      }
    } else {
      if (num_groups > 1) {
        MARLIN_CHECK(size_k % num_groups == 0,
                     "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
        group_size = static_cast<int>(size_k / num_groups);
      } else {
        group_size = -1;
      }
    }

    // Derive has_zp
    int64_t b_zeros_last = (b_zeros.dim() > 0) ? b_zeros.size(b_zeros.dim() - 1) : 0;
    bool has_zp = (b_zeros_last > 0);

    // Validate has_zp + is_zp_float combo
    if (has_zp && is_zp_float) {
      MARLIN_CHECK(std::is_same<scalar_t, half>::value,
                   "Computation type must be float16 (half) when using float zero points.");
    }

    // Call marlin_mm
    marlin::marlin_mm<scalar_t>(
        a.data_ptr(),
        b_q_weight.data_ptr(),
        c.data_ptr(),
        c_tmp.data_ptr(),
        b_scales.data_ptr(),
        global_scale.data_ptr(),
        b_zeros.data_ptr(),
        g_idx.data_ptr(),
        perm.data_ptr(),
        a_tmp.data_ptr(),
        static_cast<int>(size_m),
        static_cast<int>(size_n),
        static_cast<int>(size_k),
        static_cast<int>(a.stride(0)),  // lda
        workspace.data_ptr(),
        b_q_type,
        has_act_order,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        dev,
        stream,
        -1,  // thread_k (auto)
        -1,  // thread_n (auto)
        sms,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float);
  }
};

}  // namespace
