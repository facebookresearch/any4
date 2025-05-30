// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "StaticUtils.h"
#include "TinyGemm.h"
#include "TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

namespace tinygemm {

// FIXME: parallelize better, smem staging etc?
template <typename T, int InnerKTiles>
__global__ void matrix_to_m16n8k16_B_layout(
    // size [n][k]
    const at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> in,
    // size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles * 4]
    at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> out) {
  static_assert(sizeof(T) == 2, "");
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto kTile = blockIdx.x;
  auto nTile = blockIdx.y;

  auto nBase = nTile * kNTileSize;
  auto kBase = kTile * kKTileSize;

  auto t = threadIdx.x;

  auto n0 = nBase + (t / 4);
  bool n0Valid = n0 < in.size(0);

  auto k_tile_idx = kTile / InnerKTiles;
  auto inner_idx = (kTile % InnerKTiles) * 4;

  // It is possible that the current warp k-tile is
  // beyond the valid tiles (if InnerKTiles == 2 and k = 16, say), in which case
  // the boundary checking below will still result in zeros being filled
  int32_t ks[4];
  ks[0] = kBase + (t % 4) * 2;
  ks[1] = ks[0] + 1;
  ks[2] = ks[0] + 8;
  ks[3] = ks[0] + 8 + 1;

  const T* pIn = &in[n0][0];
  T inV[4];

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    inV[i] = (n0Valid && (ks[i] < in.size(1))) ? pIn[ks[i]] : T(0);
  }

  T* pOut = &out[nTile][k_tile_idx][t][inner_idx];
  auto inV_vec = reinterpret_cast<uint2*>(&inV[0]);
  auto pOut_vec = reinterpret_cast<uint2*>(&pOut[0]);
  *pOut_vec = *inV_vec;
}

// Convert a row-major matrix to "B" A100 m16n8k16 tensor core layout
//
// input is [n][k]
// output is [ceil(n / 8)][ceil(k / (16 * innerKTiles))][32][innerKTiles * 4]
//
// n and k can be arbitrary, but the TC format is padded to n8 x k32 tiles with
// zeros. innerKTiles of 1 means 8 bytes are packed innermost, 2 means 16 bytes
// are packed innermost
torch::Tensor convert_matrix_to_m16n8k16_B_layout(
    const torch::Tensor& in,
    int64_t innerKTiles) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dtype() == torch::kFloat16 || in.dtype() == torch::kBFloat16);
  TORCH_CHECK(in.is_contiguous());
  TORCH_CHECK(innerKTiles == 1 || innerKTiles == 2);

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto nTiles = divUp(in.size(0), kNTileSize);
  // The matrix could have k <= 32 or an odd number of k-tiles, but with
  // innerKTiles == 2 this means that one of the k-tiles must still be present
  // yet filled with zeros
  auto outerKTiles = divUp(in.size(1), kKTileSize * innerKTiles);
  auto totalKTiles = outerKTiles * innerKTiles;

  // 2 k-tiles are packed back to back in the innermost dimension in order to
  // allow for 16 byte loads
  auto out = torch::empty(
      {nTiles, outerKTiles, 32, innerKTiles * 4},
      torch::TensorOptions().dtype(in.dtype()).device(in.device()));

  // k is innermost in data so should be in execution, also each warp only
  // handles a single k-tile
  // any k-tiles past kValidTiles will be filled with zeros
  dim3 grid(totalKTiles, nTiles);

#define DO_CONVERT(TYPE, INNER_K_TILES)                               \
  do {                                                                \
    matrix_to_m16n8k16_B_layout<TYPE, INNER_K_TILES>                  \
        <<<grid, kWarpSize, 0, stream>>>(                             \
            in.packed_accessor32<TYPE, 2, at::RestrictPtrTraits>(),   \
            out.packed_accessor32<TYPE, 4, at::RestrictPtrTraits>()); \
  } while (false)

  if (in.dtype() == torch::kFloat16) {
    if (innerKTiles == 1) {
      DO_CONVERT(at::Half, 1);
    } else if (innerKTiles == 2) {
      DO_CONVERT(at::Half, 2);
    }
  } else if (in.dtype() == torch::kBFloat16) {
    if (innerKTiles == 1) {
      DO_CONVERT(at::BFloat16, 1);
    } else if (innerKTiles == 2) {
      DO_CONVERT(at::BFloat16, 2);
    }
  }

#undef DO_CONVERT

  return out;
}

// FIXME: parallelize better, smem staging etc?
template <typename T, int InnerKTiles>
__global__ void matrix_from_m16n8k16_B_layout(
    // size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles * 4]
    const at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> in,
    // size [n][k]
    at::PackedTensorAccessor32<T, 2, at::RestrictPtrTraits> out) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto kTile = blockIdx.x;
  auto nTile = blockIdx.y;

  auto nBase = nTile * kNTileSize;
  auto kBase = kTile * kKTileSize;

  auto t = threadIdx.x;

  auto inV = *reinterpret_cast<const VectorType<T, 4>*>(
      &in[nTile][kTile / InnerKTiles][t][(kTile % InnerKTiles) * 4]);

  auto pOut = &out[nBase][kBase];

  auto n0 = nBase + (t / 4);
  bool n0Valid = n0 < out.size(0);

  // Each lane has the same n dimension for all outputs
  if (n0Valid) {
    int32_t ks[4];
    ks[0] = kBase + (t % 4) * 2;
    ks[1] = ks[0] + 1;
    ks[2] = ks[0] + 8;
    ks[3] = ks[0] + 8 + 1;

    for (int i = 0; i < 4; ++i) {
      if (ks[i] < out.size(1)) {
        out[n0][ks[i]] = inV.vals[i];
      }
    }
  }
}

// Convert a "B" A100 m16n8k16 tensor core layout to row major matrix
//
// input is [ceil(n / 8)][ceil(k / (16 * innerKTiles))][32][4 * innerKTiles]
// output is [n][k]
//
// n and k can be arbitrary, but the TC format is padded to n8 x k16 tiles with
// zeros. innerKTiles of 1 means 8 bytes are packed innermost, 2 means 16 bytes
// are packed innermost, which is determined by inspecting the input
torch::Tensor convert_matrix_from_m16n8k16_B_layout(
    const torch::Tensor& in,
    int64_t n,
    int64_t k) {
  c10::cuda::CUDAGuard g(in.device());
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dtype() == torch::kFloat16 || in.dtype() == torch::kBFloat16);
  TORCH_CHECK(in.is_contiguous());
  TORCH_CHECK(in.dim() == 4);

  TORCH_CHECK(isEvenDivisor(in.size(3), 4));
  auto innerKTiles = in.size(3) / 4;
  TORCH_CHECK(innerKTiles == 1 || innerKTiles == 2);

  auto nTiles = divUp(n, kNTileSize);
  // k tiles with valid data
  auto kTiles = divUp(k, kKTileSize);

  TORCH_CHECK(nTiles == in.size(0));
  TORCH_CHECK(divUp(k, kKTileSize * innerKTiles) == in.size(1));
  TORCH_CHECK(kWarpSize == in.size(2));

  // 2 k-tiles are packed back to back in the innermost dimension in order to
  // allow for 16 byte loads
  auto out = torch::empty(
      {n, k}, torch::TensorOptions().dtype(in.dtype()).device(in.device()));

  // k is innermost in data so should be in execution, also each warp only
  // handles a single k-tile
  // we need not consider all k-tiles in `in`, just the ones that correspond
  // to valid data
  dim3 grid(kTiles, nTiles);

#define DO_CONVERT(TYPE, INNER_K_TILES)                               \
  do {                                                                \
    matrix_from_m16n8k16_B_layout<TYPE, INNER_K_TILES>                \
        <<<grid, kWarpSize, 0, stream>>>(                             \
            in.packed_accessor32<TYPE, 4, at::RestrictPtrTraits>(),   \
            out.packed_accessor32<TYPE, 2, at::RestrictPtrTraits>()); \
  } while (false)

  if (in.dtype() == torch::kFloat16) {
    if (innerKTiles == 1) {
      DO_CONVERT(at::Half, 1);
    } else if (innerKTiles == 2) {
      DO_CONVERT(at::Half, 2);
    }
  } else if (in.dtype() == torch::kBFloat16) {
    if (innerKTiles == 1) {
      DO_CONVERT(at::BFloat16, 1);
    } else if (innerKTiles == 2) {
      DO_CONVERT(at::BFloat16, 2);
    }
  }

#undef DO_CONVERT

  return out;
}

// FIXME: parallelize better, smem staging etc?
template <int InnerKTiles>
__global__ void matrix_to_m16n8k16_Bint4_layout(
    // size [n][k]
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> in,
    // size [ceil(n / 8)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles / 2]
    at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> out) {
  // int4 values are packed into int32 values, which require at least 8. Given
  // m16n8k16 B layout requires 4 scalar values/lane, the minimum number of
  // innermost k-tiles that we can use is 2.
  static_assert(InnerKTiles >= 2 && isPowerOf2(InnerKTiles), "");

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  // gridDim.x corresponds to the number of k-tiles divided by InnerKTiles
  auto kOuterTile = blockIdx.x;
  auto nTile = blockIdx.y;
  auto t = threadIdx.x;

  // Two k-tiles are packed into an int32 at a time
#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; innerKTile += 2) {
    // n dimension that this lane loads from
    auto n0 = nTile * kNTileSize + (t / 4);

    bool n0Valid = n0 < in.size(0);

    int32_t ks[8];

    auto kBase0 = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize;
    ks[0] = kBase0 + (t % 4) * 2;
    ks[1] = ks[0] + 1;
    ks[2] = ks[0] + 8;
    ks[3] = ks[0] + 8 + 1;

    auto kBase1 = kBase0 + kKTileSize;
    ks[4] = kBase1 + (t % 4) * 2;
    ks[5] = ks[4] + 1;
    ks[6] = ks[4] + 8;
    ks[7] = ks[4] + 8 + 1;

    auto pIn = &in[n0][0];

    uint32_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = (n0Valid && ks[i] < in.size(1)) ? pIn[ks[i]] : uint32_t(0);
    }

    int32_t pack = (v[7] << 28) | (v[5] << 24) | (v[3] << 20) | (v[1] << 16) |
        (v[6] << 12) | (v[4] << 8) | (v[2] << 4) | v[0];

    // inner k-tiles pack two at a time
    out[nTile][kOuterTile][t][innerKTile / 2] = pack;
  }
}

// input is [n][k] (int32 dtype)
// output is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles / 2]
torch::Tensor convert_matrix_to_m16n8k16_Bint4_layout(
    const torch::Tensor& in,
    int64_t innerKTiles = 2) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dim() == 2);
  TORCH_CHECK(in.dtype() == torch::kInt32);
  TORCH_CHECK(in.is_contiguous());

  // At least 2 k-tiles need to be packed back to back in the innermost
  // dimension, as the m16n8k16 tensor core tile presents 4 scalar values for
  // the B matrix, but the minimum word size for the packed format is 4 bytes
  // (int32). 4 inner K-tiles = 8 byte load, 8 inner k-tiles = 16 byte load
  // which is the maximum vectorized load/store size
  TORCH_CHECK(innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8);

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto nTiles = divUp(in.size(0), kNTileSize);

  // k-tiles are packed back to back in the innermost dimension in order to
  // allow for 4/8/16 byte loads
  TORCH_CHECK(isEvenDivisor(in.size(1), innerKTiles * kKTileSize));
  // kSuperTiles is the number of k-tiles assuming k is innerKTiles * kKTileSize
  auto kSuperTiles = divUp(in.size(1), innerKTiles * kKTileSize);

  // each block handles `innerKTiles` k-tiles.
  // 2 k-tiles are a single int32
  auto out = torch::empty(
      {nTiles, kSuperTiles, 32, innerKTiles / 2},
      torch::TensorOptions().dtype(torch::kInt32).device(in.device()));

  dim3 grid(kSuperTiles, nTiles);

  if (innerKTiles == 2) {
    matrix_to_m16n8k16_Bint4_layout<2><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 4) {
    matrix_to_m16n8k16_Bint4_layout<4><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 8) {
    matrix_to_m16n8k16_Bint4_layout<8><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  }

  return out;
}

// FIXME: parallelize better, smem staging etc?
template <int InnerKTiles>
__global__ void matrix_to_m16n8k16_Bint8_layout(
    // size [n][k]
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> in,
    // size [ceil(n / 8)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles]
    at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> out) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  // gridDim.x corresponds to the number of k-tiles divided by InnerKTiles
  auto kOuterTile = blockIdx.x;
  auto nTile = blockIdx.y;
  auto t = threadIdx.x;

  // each inner k-tile is a single int32, we pack them one at a time
#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; ++innerKTile) {
    // n dimension that this lane loads from
    auto n0 = nTile * kNTileSize + (t / 4);

    bool n0Valid = n0 < in.size(0);

    int32_t ks[4];

    auto kBase0 = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize;
    ks[0] = kBase0 + (t % 4) * 2;
    ks[1] = ks[0] + 1;
    ks[2] = ks[0] + 8;
    ks[3] = ks[0] + 8 + 1;

    auto pIn = &in[n0][0];

    uint32_t v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = (n0Valid && ks[i] < in.size(1)) ? pIn[ks[i]] : uint32_t(0);
    }

    int32_t pack = (v[3] << 24) | (v[1] << 16) | (v[2] << 8) | v[0];

    out[nTile][kOuterTile][t][innerKTile] = pack;
  }
}

// input is [n][k] (int32 dtype)
// output is [n / 8][k / (InnerKTiles * 16)][32][innerKTiles]
// FIXME: combine with above
torch::Tensor convert_matrix_to_m16n8k16_Bint8_layout(
    const torch::Tensor& in,
    int64_t innerKTiles = 1) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dim() == 2);
  TORCH_CHECK(in.dtype() == torch::kInt32);
  TORCH_CHECK(in.is_contiguous());

  // 1 inner k-tile is a 4 byte load
  // 2 inner k-tiles is an 8 byte load
  // 4 inner k-tiles is a 16 byte load
  TORCH_CHECK(innerKTiles == 1 || innerKTiles == 2 || innerKTiles == 4);

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto nTiles = divUp(in.size(0), kNTileSize);

  // k-tiles are packed back to back in the innermost dimension in order to
  // allow for 4/8/16 byte loads
  TORCH_CHECK(isEvenDivisor(in.size(1), innerKTiles * kKTileSize));
  // kSuperTiles is the number of k-tiles assuming k is innerKTiles * kKTileSize
  auto kSuperTiles = divUp(in.size(1), innerKTiles * kKTileSize);

  // each block handles `innerKTiles` k-tiles.
  auto out = torch::empty(
      {nTiles, kSuperTiles, 32, innerKTiles},
      torch::TensorOptions().dtype(torch::kInt32).device(in.device()));

  dim3 grid(kSuperTiles, nTiles);

  if (innerKTiles == 1) {
    matrix_to_m16n8k16_Bint8_layout<1><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 2) {
    matrix_to_m16n8k16_Bint8_layout<2><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 4) {
    matrix_to_m16n8k16_Bint8_layout<4><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  }

  return out;
}

} // namespace tinygemm
