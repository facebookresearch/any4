// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "MatrixLayoutA.cuh"
#include "MatrixLayoutB.cuh"
#include "StaticUtils.h"
#include "TinyGemm.h"
#include "TinyGemmImpl.cuh"
#include "TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <torch/types.h>

namespace tinygemm {

// weightOnRight: (m, k) x (n x k)^t -> (m, n)
//
//   tc (ceil(m / 16), ceil(k / 16), 32, 8) x
//     tc (ceil(n / 8), ceil(k / (innerKTiles * 16)), 32, innerKTiles * 4) ->
//       tc (ceil(m / 16), ceil(n / 16), 32, 8)
//
// !weightOnRight: ((m, k) x (n x k)^t)^t -> (n, m)
//
//   tc (ceil(m / 16), ceil(k / 16), 32, 8) x
//     tc (ceil(n / 8), ceil(k / (innerKTiles * 16)), 32, innerKTiles * 4) ->
//        tc (ceil(n / 8), ceil(m / 16), 32, innerKTiles * 4)
template <FloatType FT>
torch::Tensor tinygemm_y_FT16TC_x_FT16TC_w_FT16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight) {
  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(A.device() == B.device());

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto mTiles = A.size(0);
  auto m = mTiles * kMTileSize;

  auto nTiles = B.size(0);
  auto n = nTiles * kNTileSize;

  auto kTiles = A.size(1);
  auto k = kTiles * kKTileSize;

  // A only has 1 k-tile in the innermost dimension
  TORCH_CHECK(A.is_contiguous());
  TORCH_CHECK(A.dim() == 4);
  TORCH_CHECK(A.size(2) == kWarpSize);
  TORCH_CHECK(A.size(3) == 8);

  // B has 1 or 2 k-tiles in the innermost dimension
  auto B_innerKTiles = B.size(3) / 4;
  TORCH_CHECK(
      isEvenDivisor(B.size(3), 4) && (B_innerKTiles == 1 || B_innerKTiles == 2))

  TORCH_CHECK(B.is_contiguous());
  TORCH_CHECK(B.dim() == 4);
  TORCH_CHECK(B.size(1) == divUp(kTiles, B_innerKTiles));
  TORCH_CHECK(B.size(2) == kWarpSize);
  // Each k-tile is 4 bfloat16 words
  TORCH_CHECK(B.size(3) == 4 * B_innerKTiles);

  torch::Tensor C_final;

  if (weightOnRight) {
    // Output is the same format as the A matrix expects, except we can only
    // produce 8 elements along n at a time instead of 16, hence dividing
    // n-tiles by 2
    C_final = torch::empty(
        {mTiles, divUp(nTiles, 2), kWarpSize, 8},
        torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  } else {
    // Output is the same format as the B matrix expects (including the same
    // number of inner k-tiles as B)
    C_final = torch::empty(
        {nTiles, divUp(mTiles, B_innerKTiles), kWarpSize, B_innerKTiles * 4},
        torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  }

  constexpr int kWarps = 8;
  constexpr int kKTilesPerWarp = 8;
  using ALayout = ALayout_TC<kWarps, FT>;

  auto dqInfo = DequantInfo::empty();

  if (weightOnRight) {
    if (B_innerKTiles == 1) {
      using BLayout = BLayout_TC<kWarps, FT, 1>;
      using CLayout = ALayout;
      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    } else if (B_innerKTiles == 2) {
      using BLayout = BLayout_TC<kWarps, FT, 2>;
      using CLayout = ALayout;
      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    }
  } else {
    if (B_innerKTiles == 1) {
      using BLayout = BLayout_TC<kWarps, FT, 1>;
      using CLayout = BLayout;
      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    } else if (B_innerKTiles == 2) {
      using BLayout = BLayout_TC<kWarps, FT, 2>;
      using CLayout = BLayout;
      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    }
  }

  return C_final;
}

// weightOnRight: (m, k) x (n x k)^t -> (m, n)
//
//   rm (m, k) x
//     tc (ceil(n / 8), ceil(k / (innerKTiles * 16)), 32, innerKTiles * 4) ->
//       rm (m, n)
//
// !weightOnRight: ((m, k) x (n x k)^t)^t -> (n, m)
//
//   tc (ceil(m / 16), ceil(k / 16), 32, 8) x
//     rm (n x k) ->
//       rm (m, n)
template <FloatType FT>
torch::Tensor tinygemm_y_FT16RM_x_FT16RM_w_FT16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight) {
  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(A.device() == B.device());

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  int32_t m;
  int32_t mTiles;
  int32_t n;
  int32_t nTiles;
  int32_t k;
  int32_t kTiles;

  int32_t A_innerKTiles;
  int32_t B_innerKTiles;

  TORCH_CHECK(A.is_contiguous());
  TORCH_CHECK(B.is_contiguous());

  if (weightOnRight) {
    TORCH_CHECK(A.dim() == 2);
    TORCH_CHECK(B.dim() == 4);

    // A: row major layout
    m = A.size(0);
    mTiles = divUp(m, kMTileSize);
    k = A.size(1);
    kTiles = divUp(k, kKTileSize);
    A_innerKTiles = 0; // unused

    // B: tensor core layout
    nTiles = B.size(0);
    n = nTiles * kNTileSize;
    B_innerKTiles = B.size(3) / 4;

    // validation
    TORCH_CHECK(B.size(1) == kTiles / B_innerKTiles);
    TORCH_CHECK(B.size(2) == kWarpSize);
    TORCH_CHECK(
        isEvenDivisor(B.size(3), 4) &&
        (B_innerKTiles == 1 || B_innerKTiles == 2));
  } else {
    TORCH_CHECK(A.dim() == 4);
    TORCH_CHECK(B.dim() == 2);

    // A: tensor core layout
    mTiles = A.size(0);
    m = mTiles * kMTileSize;
    A_innerKTiles = 1; // always 1

    // B: row major layout
    n = B.size(0);
    nTiles = divUp(n, kNTileSize);
    k = B.size(1);
    kTiles = divUp(k, kKTileSize);
    B_innerKTiles = 0; // unused

    // validate A
    TORCH_CHECK(A.dim() == 4);
    TORCH_CHECK(A.size(1) == kTiles / A_innerKTiles);
    TORCH_CHECK(A.size(2) == kWarpSize);
    TORCH_CHECK(A.size(3) == 8);
  }

  // Output is a standard row-major matrix
  torch::Tensor C_final;

  if (weightOnRight) {
    C_final = torch::empty(
        {m, n}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  } else {
    C_final = torch::empty(
        {n, m}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  }

  constexpr int kWarps = 8;
  constexpr int kKTilesPerWarp = 8;
  auto dqInfo = DequantInfo::empty();

  if (weightOnRight) {
    using ALayout = ALayout_RM<kWarps, FT>;
    using CLayout = ALayout;

    if (B_innerKTiles == 1) {
      using BLayout = BLayout_TC<kWarps, FT, 1>;

      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    } else if (B_innerKTiles == 2) {
      using BLayout = BLayout_TC<kWarps, FT, 2>;

      launch_tinygemm_kernel<
          FT,
          ALayout,
          BLayout,
          CLayout,
          kWarps,
          kKTilesPerWarp>(
          A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
    }
  } else {
    using ALayout = ALayout_TC<kWarps, FT>;
    using BLayout = BLayout_RM<kWarps, FT>;
    using CLayout = BLayout;

    launch_tinygemm_kernel<
        FT,
        ALayout,
        BLayout,
        CLayout,
        kWarps,
        kKTilesPerWarp>(
        A, B, dqInfo, C_final, mTiles, nTiles, kTiles, m, n, k, stream);
  }

  return C_final;
}

torch::Tensor tinygemm_y_f16TC_x_f16TC_w_f16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16 || A.dtype() == torch::kFloat16);
  TORCH_CHECK(B.dtype() == torch::kBFloat16 || B.dtype() == torch::kFloat16);
  TORCH_CHECK(A.dtype() == B.dtype());

  if (A.dtype() == torch::kBFloat16) {
    return tinygemm_y_FT16TC_x_FT16TC_w_FT16TC<FloatType::BFloat16>(
        A, B, weightOnRight);
  } else {
    return tinygemm_y_FT16TC_x_FT16TC_w_FT16TC<FloatType::Float16>(
        A, B, weightOnRight);
  }
}

torch::Tensor tinygemm_y_f16RM_x_f16RM_w_f16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight) {
  TORCH_CHECK(A.dtype() == torch::kBFloat16 || A.dtype() == torch::kFloat16);
  TORCH_CHECK(B.dtype() == torch::kBFloat16 || B.dtype() == torch::kFloat16);
  TORCH_CHECK(A.dtype() == B.dtype());

  if (A.dtype() == torch::kBFloat16) {
    return tinygemm_y_FT16RM_x_FT16RM_w_FT16TC<FloatType::BFloat16>(
        A, B, weightOnRight);
  } else {
    return tinygemm_y_FT16RM_x_FT16RM_w_FT16TC<FloatType::Float16>(
        A, B, weightOnRight);
  }
}

} // namespace tinygemm
