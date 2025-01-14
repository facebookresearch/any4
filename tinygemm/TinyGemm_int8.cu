// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#include "tinygemm/MatrixLayoutA.cuh"
#include "tinygemm/MatrixLayoutB.cuh"
#include "tinygemm/StaticUtils.h"
#include "tinygemm/TinyGemm.h"
#include "tinygemm/TinyGemmImpl.cuh"
#include "tinygemm/TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

namespace tinygemm {

template <FloatType FT>
torch::Tensor tinygemm_y_FT16TC_x_FT16TC_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      A.device() == B.device() && A.device() == qScaleAndZeros.device());

  // General TC format validation
  TORCH_CHECK(A.is_contiguous());
  TORCH_CHECK(A.dim() == 4);
  TORCH_CHECK(A.size(2) == kWarpSize);

  TORCH_CHECK(B.is_contiguous());
  TORCH_CHECK(B.dim() == 4);
  TORCH_CHECK(B.size(2) == kWarpSize);

  int A_innerKTiles = 1;
  int B_innerKTiles = 1;

  if (weightOnRight) {
    TORCH_CHECK(A.size(3) == 8); // activations
    TORCH_CHECK(B.size(3) == 1 || B.size(3) == 2 || B.size(3) == 4); // weights
    TORCH_CHECK(B.dtype() == torch::kInt32);

    A_innerKTiles = 1;
    B_innerKTiles = B.size(3);
  } else {
    TORCH_CHECK(A.size(3) == 2 || A.size(3) == 4); // weights
    TORCH_CHECK(B.size(3) == 4 || B.size(3) == 8); // activations
    TORCH_CHECK(A.dtype() == torch::kInt32);

    A_innerKTiles = A.size(3) / 2;
    B_innerKTiles = B.size(3) / 4;
  }

  auto mTiles = A.size(0);
  auto m = mTiles * kMTileSize;

  auto nTiles = B.size(0);
  auto n = nTiles * kNTileSize;

  // validate k dimension equality
  auto kTileA = A.size(1) * A_innerKTiles;
  auto kTileB = B.size(1) * B_innerKTiles;
  TORCH_CHECK(kTileA == kTileB);
  auto kTiles = kTileA;

  auto k = kTiles * kKTileSize;

  // Validate the scale and zero point tensor for dequantization
  // These are the only versions handled at the moment
  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
      qGroupSize == 256);

  TORCH_CHECK(qScaleAndZeros.dim() == 3);
  auto numQGroups = qScaleAndZeros.size(0);
  TORCH_CHECK(k / numQGroups == qGroupSize);

  // Right now, the dequantization code assumes full n (or m) tiles for the
  // weights, so qScaleAndZeros must be sized as such (even if not all rows
  // are valid)
  TORCH_CHECK(qScaleAndZeros.size(1) == (weightOnRight ? n : m));
  TORCH_CHECK(qScaleAndZeros.size(2) == 2);

  // Output produced here
  torch::Tensor C_final;

  if (weightOnRight) {
    // Output is the same format as the A matrix expects, except we can only
    // produce 8 elements along n at a time instead of 16, hence dividing
    // n-tiles by 2
    C_final = torch::empty(
        // output is produced in `A` format, so 8 as innermost dim
        {mTiles, divUp(nTiles, 2), kWarpSize, 8},
        torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  } else {
    // Output is the same format as the B matrix expects
    C_final = torch::empty(
        {nTiles, divUp(mTiles, B_innerKTiles), kWarpSize, B_innerKTiles * 4},
        torch::TensorOptions().dtype(B.dtype()).device(B.device()));
  }

  auto dqInfo = DequantInfo::empty();
  dqInfo.qInfo1 = qScaleAndZeros.data_ptr();

#define RUN_GEMM(WARPS, K_TILES_PER_WARP, A_LAYOUT, B_LAYOUT, C_LAYOUT)  \
  do {                                                                   \
    launch_tinygemm_kernel<                                              \
        FT,                                                              \
        A_LAYOUT,                                                        \
        B_LAYOUT,                                                        \
        C_LAYOUT,                                                        \
        WARPS,                                                           \
        K_TILES_PER_WARP>(                                               \
        A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream); \
  } while (false)

  constexpr int kWarps = 8;
  constexpr int kKTilesPerWarp = 8;

  TORCH_CHECK(
      kKTilesPerWarp >= A_innerKTiles && kKTilesPerWarp >= B_innerKTiles);

#define B_LAYOUT_INT8_QGROUP(B_INNER_K_TILES)                            \
  switch (qGroupSize) {                                                  \
    case 32: {                                                           \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 32>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 64: {                                                           \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 64>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 128: {                                                          \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 128>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 256: {                                                          \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 256>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
  }

#define A_LAYOUT_INT8_QGROUP(A_INNER_K_TILES)                            \
  switch (qGroupSize) {                                                  \
    case 32: {                                                           \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 32>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 64: {                                                           \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 64>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 128: {                                                          \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 128>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 256: {                                                          \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 256>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
  }

  if (weightOnRight) {
    using ALayout = ALayout_TC<kWarps, FT>;
    using CLayout = ALayout;
    if (B_innerKTiles == 1) {
      B_LAYOUT_INT8_QGROUP(1);
    } else if (B_innerKTiles == 2) {
      B_LAYOUT_INT8_QGROUP(2);
    } else if (B_innerKTiles == 4) {
      B_LAYOUT_INT8_QGROUP(4);
    }
  } else {
    if (B_innerKTiles == 1) {
      using BLayout = BLayout_TC<kWarps, FT, 1>;
      using CLayout = BLayout;

      if (A_innerKTiles == 1) {
        A_LAYOUT_INT8_QGROUP(1);
      } else if (A_innerKTiles == 2) {
        A_LAYOUT_INT8_QGROUP(2);
      }
    } else if (B_innerKTiles == 2) {
      using BLayout = BLayout_TC<kWarps, FT, 2>;
      using CLayout = BLayout;

      if (A_innerKTiles == 1) {
        A_LAYOUT_INT8_QGROUP(1);
      } else if (A_innerKTiles == 2) {
        A_LAYOUT_INT8_QGROUP(2);
      }
    }
  }

#undef A_LAYOUT_INT8_QGROUP
#undef B_LAYOUT_INT8_QGROUP
#undef RUN_GEMM

  return C_final;
}

template <FloatType FT>
torch::Tensor tinygemm_y_FT16RM_x_FT16RM_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      A.device() == B.device() && A.device() == qScaleAndZeros.device());

  int32_t m;
  int32_t mTiles;
  int32_t n;
  int32_t nTiles;
  int32_t k;
  int32_t kTiles;

  int A_innerKTiles = 1;
  int B_innerKTiles = 1;

  if (weightOnRight) {
    TORCH_CHECK(A.dim() == 2 && A.is_contiguous());
    TORCH_CHECK(
        B.dim() == 4 && B.dtype() == torch::kInt32 && B.is_contiguous());

    // A: row major layout
    m = A.size(0);
    mTiles = divUp(m, kMTileSize);
    k = A.size(1);
    kTiles = divUp(k, kKTileSize);
    A_innerKTiles = 0; // unused

    // B: tensor core layout
    nTiles = B.size(0);
    n = nTiles * kNTileSize;
    B_innerKTiles = B.size(3);

    // validation
    TORCH_CHECK(B.size(1) == divUp(kTiles, B_innerKTiles));
    TORCH_CHECK(B.size(2) == kWarpSize);
    TORCH_CHECK(B_innerKTiles == 1 || B_innerKTiles == 2 || B_innerKTiles == 4);
  } else {
    TORCH_CHECK(
        A.dim() == 4 && A.dtype() == torch::kInt32 && A.is_contiguous());
    TORCH_CHECK(B.dim() == 2 && B.is_contiguous());
    TORCH_CHECK(isEvenDivisor(A.size(3), 2));

    // A: tensor core layout
    mTiles = A.size(0);
    m = mTiles * kMTileSize;
    A_innerKTiles = A.size(3) / 2;

    // B: row major layout
    n = B.size(0);
    nTiles = divUp(n, kNTileSize);
    k = B.size(1);
    kTiles = divUp(k, kKTileSize);
    B_innerKTiles = 0; // unused

    // validation
    TORCH_CHECK(A.size(1) == divUp(kTiles, A_innerKTiles));
    TORCH_CHECK(A.size(2) == kWarpSize);
    TORCH_CHECK(A_innerKTiles == 1 || A_innerKTiles == 2);
  }

  // Validate the scale and zero point tensor for dequantization
  // These are the only versions handled at the moment
  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
      qGroupSize == 256);

  TORCH_CHECK(qScaleAndZeros.dim() == 3);
  auto numQGroups = qScaleAndZeros.size(0);
  TORCH_CHECK(
      kTiles * kKTileSize >= qGroupSize &&
      isEvenDivisor(kTiles * kKTileSize, qGroupSize));
  TORCH_CHECK(qScaleAndZeros.size(1) == (weightOnRight ? n : m));
  TORCH_CHECK(qScaleAndZeros.size(2) == 2);

  // Output is a standard row-major matrix
  torch::Tensor C_final;

  if (weightOnRight) {
    C_final = torch::empty(
        {m, n}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  } else {
    C_final = torch::empty(
        {n, m}, torch::TensorOptions().dtype(B.dtype()).device(B.device()));
  }

  auto dqInfo = DequantInfo::empty();
  dqInfo.qInfo1 = qScaleAndZeros.data_ptr();

#define RUN_GEMM(WARPS, K_TILES_PER_WARP, A_LAYOUT, B_LAYOUT, C_LAYOUT)  \
  do {                                                                   \
    launch_tinygemm_kernel<                                              \
        FT,                                                              \
        A_LAYOUT,                                                        \
        B_LAYOUT,                                                        \
        C_LAYOUT,                                                        \
        WARPS,                                                           \
        K_TILES_PER_WARP>(                                               \
        A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream); \
  } while (false)

  constexpr int kWarps = 8;
  constexpr int kKTilesPerWarp = 8;

  TORCH_CHECK(
      kKTilesPerWarp >= A_innerKTiles && kKTilesPerWarp >= B_innerKTiles);

#define B_LAYOUT_INT8_QGROUP(B_INNER_K_TILES)                            \
  switch (qGroupSize) {                                                  \
    case 32: {                                                           \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 32>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 64: {                                                           \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 64>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 128: {                                                          \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 128>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 256: {                                                          \
      using BLayout = BLayout_TC_int8<kWarps, FT, B_INNER_K_TILES, 256>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
  }

#define A_LAYOUT_INT8_QGROUP(A_INNER_K_TILES)                            \
  switch (qGroupSize) {                                                  \
    case 32: {                                                           \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 32>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 64: {                                                           \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 64>;  \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 128: {                                                          \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 128>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
    case 256: {                                                          \
      using ALayout = ALayout_TC_int8<kWarps, FT, A_INNER_K_TILES, 256>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);       \
    } break;                                                             \
  }

  if (weightOnRight) {
    using ALayout = ALayout_RM<kWarps, FT>;
    using CLayout = ALayout;
    if (B_innerKTiles == 1) {
      B_LAYOUT_INT8_QGROUP(1);
    } else if (B_innerKTiles == 2) {
      B_LAYOUT_INT8_QGROUP(2);
    } else if (B_innerKTiles == 4) {
      B_LAYOUT_INT8_QGROUP(4);
    }
  } else {
    using BLayout = BLayout_RM<kWarps, FT>;
    using CLayout = BLayout;

    if (A_innerKTiles == 1) {
      A_LAYOUT_INT8_QGROUP(1);
    } else if (A_innerKTiles == 2) {
      A_LAYOUT_INT8_QGROUP(2);
    }
  }

#undef A_LAYOUT_INT8_QGROUP
#undef B_LAYOUT_INT8_QGROUP
#undef RUN_GEMM

  return C_final;
}

torch::Tensor tinygemm_y_f16TC_x_f16TC_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight) {
  if (weightOnRight) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 || A.dtype() == torch::kFloat16);

    if (A.dtype() == torch::kBFloat16) {
      return tinygemm_y_FT16TC_x_FT16TC_w_int8TC<FloatType::BFloat16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    } else {
      return tinygemm_y_FT16TC_x_FT16TC_w_int8TC<FloatType::Float16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    }
  } else {
    TORCH_CHECK(B.dtype() == torch::kBFloat16 || B.dtype() == torch::kFloat16);

    if (B.dtype() == torch::kBFloat16) {
      return tinygemm_y_FT16TC_x_FT16TC_w_int8TC<FloatType::BFloat16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    } else {
      return tinygemm_y_FT16TC_x_FT16TC_w_int8TC<FloatType::Float16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    }
  }
}

torch::Tensor tinygemm_y_f16RM_x_f16RM_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight) {
  if (weightOnRight) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 || A.dtype() == torch::kFloat16);

    if (A.dtype() == torch::kBFloat16) {
      return tinygemm_y_FT16RM_x_FT16RM_w_int8TC<FloatType::BFloat16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    } else {
      return tinygemm_y_FT16RM_x_FT16RM_w_int8TC<FloatType::Float16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    }
  } else {
    TORCH_CHECK(B.dtype() == torch::kBFloat16 || B.dtype() == torch::kFloat16);

    if (B.dtype() == torch::kBFloat16) {
      return tinygemm_y_FT16RM_x_FT16RM_w_int8TC<FloatType::BFloat16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    } else {
      return tinygemm_y_FT16RM_x_FT16RM_w_int8TC<FloatType::Float16>(
          A, B, qGroupSize, qScaleAndZeros, weightOnRight);
    }
  }
}

} // namespace tinygemm
