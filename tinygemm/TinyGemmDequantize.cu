// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#include "tinygemm/Dequantization.cuh"
#include "tinygemm/TinyGemm.h"
#include "tinygemm/TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

namespace tinygemm {

__global__ void tinygemm_dequant_int4_kernel(
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> in,
    at::PackedTensorAccessor32<at::BFloat16, 1, at::RestrictPtrTraits> out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  // FIXME: .bf16 mma m16n8k16 requires >= 800
  // Maybe reference implementation
  CUDA_KERNEL_ASSERT(false);
#else
  bf16x2x4* out8 = reinterpret_cast<bf16x2x4*>(&out[0]);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < in.size(0);
       i += gridDim.x * blockDim.x) {
    convert_i4x8_to_f16x2x4(in[i], out8[i]);
  }
#endif
}

torch::Tensor tinygemm_dequant_int4(torch::Tensor in) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dtype() == torch::kInt32);
  TORCH_CHECK(in.dim() == 1);

  auto size16 = in.size(0) / 16;

  auto grid = dim3(1024);
  auto block = dim3(128);

  auto out = torch::empty(
      {in.numel() * 8},
      torch::TensorOptions().dtype(at::kBFloat16).device(in.device()));

  tinygemm_dequant_int4_kernel<<<grid, block, 0, stream>>>(
      in.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      out.packed_accessor32<at::BFloat16, 1, at::RestrictPtrTraits>());
  C10_CUDA_CHECK(cudaGetLastError());

  return out;
}

} // namespace tinygemm
