// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#include "StaticUtils.h"
#include "TinyGemm.h"
#include "TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

namespace tinygemm {

template <typename T, int Warps, bool KAligned16Byte>
__global__ void matrix_to_m16n8k16_A_layout(
    // size [m][k]
    const T* __restrict__ in,
    // size [m / 16][k / 16][32][8]
    T* __restrict__ out,
    int32_t m,
    int32_t k,
    int32_t mTiles,
    int32_t kTiles) {
  // only works on 16 byte types at the moment
  static_assert(sizeof(T) == 2, "");

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto warp = threadIdx.y;
  auto t = threadIdx.x;

  auto kTile = blockIdx.x * Warps + warp;
  auto mTile = blockIdx.y;

  if (kTile >= kTiles) {
    return;
  }

  auto mBase = mTile * kMTileSize;
  auto kBase = kTile * kKTileSize;

  constexpr int kSmemRowSize = kKTileSize;
  __shared__ T inTile[Warps][kMTileSize][kSmemRowSize];

  if constexpr (KAligned16Byte) {
    // The output matrix is row-aligned to 16 bytes
    static_assert(sizeof(uint4) / sizeof(T) == 8, "");

    // each 2 contiguous lanes handles a tile row
    static_assert(kKTileSize * sizeof(T) / sizeof(uint4) == 2, "");
    auto mOutTile = t / 2;
    auto kOutTile = (t % 2) * (sizeof(uint4) / sizeof(T));

    auto mIn = mBase + mOutTile;
    auto kIn = kBase + kOutTile;

    auto pIn = in + mIn * k + kIn;
    auto pTile = &inTile[warp][mOutTile][kOutTile];

    if (mIn < m && kIn < k) {
      *reinterpret_cast<uint4*>(pTile) = *reinterpret_cast<const uint4*>(pIn);
    } else {
      *reinterpret_cast<uint4*>(pTile) = uint4{0, 0, 0, 0};
    }
  } else {
    // The input matrix is not row-aligned to 16 bytes, instead we just assume
    // word (2 byte) alignment

    // each half warp handles a tile row
    constexpr auto kHalfWarpSize = kWarpSize / 2;
    static_assert(kKTileSize == kHalfWarpSize, "");

    auto mOutTile = t / kHalfWarpSize;
    auto kOutTile = t % kHalfWarpSize;

    auto mIn = mBase + mOutTile;
    auto kIn = kBase + kOutTile;

    auto pIn = in + mIn * k + kIn;
    auto pTile = &inTile[warp][mOutTile][kOutTile];

    if (kIn < k) {
#pragma unroll
      for (int i = 0; i < kMTileSize; i += 2) {
        if constexpr (std::is_same<T, __half>::value) {
          pTile[i * kSmemRowSize] = (mIn + i < m) ? pIn[i * k] : __float2half(0.0f);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
          pTile[i * kSmemRowSize] = (mIn + i < m) ? pIn[i * k] : __float2bfloat16(0.0f);
        } else {
          pTile[i * kSmemRowSize] = (mIn + i < m) ? pIn[i * k] : T(0.0f);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kMTileSize; i += 2) {
        if constexpr (std::is_same<T, __half>::value) {
          pTile[i * kSmemRowSize] = __float2half(0.0f);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
          pTile[i * kSmemRowSize] = __float2bfloat16(0.0f);
        } else {
          pTile[i * kSmemRowSize] = T(0.0f);
        }
      }
    }
  }

  __syncwarp();

  using Vec8 = VectorType<T, 8>;
  static_assert(sizeof(Vec8) == sizeof(T) * 8, "");
  Vec8 inV;

  // Fill the packed vector with a given lane's tensor core values
  auto m0 = (t / 4);
  auto m1 = m0 + 8;

  auto k0 = (t % 4) * 2;
  auto k1 = k0 + 1;
  auto k2 = k0 + 8;
  auto k3 = k0 + 8 + 1;

  inV.vals[0] = inTile[warp][m0][k0];
  inV.vals[1] = inTile[warp][m0][k1];
  inV.vals[2] = inTile[warp][m1][k0];
  inV.vals[3] = inTile[warp][m1][k1];
  inV.vals[4] = inTile[warp][m0][k2];
  inV.vals[5] = inTile[warp][m0][k3];
  inV.vals[6] = inTile[warp][m1][k2];
  inV.vals[7] = inTile[warp][m1][k3];

  auto pOut = reinterpret_cast<Vec8*>(
      out + (((mTile * kTiles) + kTile) * kWarpSize + t) * 8);
  *pOut = inV;
}

// Convert a row-major matrix to "A" A100 m16n8k16 tensor core layout
//
// input is [m][k]
// output is [ceil(m / 16)][ceil(k / 16)][32][8]
//
// input m and k can be arbitrary, but the TC format is padded to m16 x k16
// with zeros
torch::Tensor convert_matrix_to_m16n8k16_A_layout(
    const torch::Tensor& in,
    int64_t innerKTiles) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(innerKTiles == 1);
  TORCH_CHECK(in.dtype() == torch::kBFloat16 || in.dtype() == torch::kFloat16);
  TORCH_CHECK(in.is_contiguous());

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto m = in.size(0);
  auto k = in.size(1);

  auto mTiles = divUp(in.size(0), kMTileSize);
  auto kTiles = divUp(in.size(1), kKTileSize);

  auto out = torch::empty(
      {mTiles, kTiles, 32, 8},
      torch::TensorOptions().dtype(in.dtype()).device(in.device()));

  constexpr int kWarps = 2;

  // k is innermost in data so should be in execution
  auto grid = dim3(divUp(kTiles, kWarps), mTiles);
  auto block = dim3(kWarpSize, kWarps);

  if (in.dtype() == torch::kFloat16) {
    if (isEvenDivisor(k, 8)) {
      matrix_to_m16n8k16_A_layout<__half, kWarps, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(in.data_ptr()),
              reinterpret_cast<__half*>(out.data_ptr()),
              in.size(0),
              in.size(1),
              mTiles,
              kTiles);
    } else {
      matrix_to_m16n8k16_A_layout<__half, kWarps, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(in.data_ptr()),
              reinterpret_cast<__half*>(out.data_ptr()),
              in.size(0),
              in.size(1),
              mTiles,
              kTiles);
    }
  } else if (in.dtype() == torch::kBFloat16) {
    if (isEvenDivisor(k, 8)) {
      matrix_to_m16n8k16_A_layout<__nv_bfloat16, kWarps, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
              in.size(0),
              in.size(1),
              mTiles,
              kTiles);
    } else {
      matrix_to_m16n8k16_A_layout<__nv_bfloat16, kWarps, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
              in.size(0),
              in.size(1),
              mTiles,
              kTiles);
    }
  }

  return out;
}

// FIXME: parallelize better, smem staging etc?
template <int InnerKTiles>
__global__ void matrix_to_m16n8k16_Aint4_layout(
    // size [m][k]
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> in,
    // A layout presents 8 scalar values. 8 x int4 = 32 bits, so there is
    // 1 int32 per k-tile
    // size [ceil(m / 16)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles]
    at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> out) {
  static_assert(InnerKTiles == 1 || InnerKTiles == 2 || InnerKTiles == 4, "");

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  // gridDim.x corresponds to the number of k-tiles divided by InnerKTiles
  auto kOuterTile = blockIdx.x;
  auto mTile = blockIdx.y;
  auto t = threadIdx.x;

  auto m = in.size(0);
  auto k = in.size(1);

  auto m0 = (t / 4);
  auto m1 = m0 + 8;

  auto k0 = (t % 4) * 2;
  auto k1 = k0 + 1;
  auto k2 = k0 + 8;
  auto k3 = k0 + 8 + 1;

  // One k-tile is packed into an int32 at a time
  auto mBase = mTile * kMTileSize;
#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; ++innerKTile) {
    auto kBase = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize;

    auto m0Cur = mBase + m0;
    auto m1Cur = mBase + m1;

    auto k0Cur = kBase + k0;
    auto k1Cur = kBase + k1;
    auto k2Cur = kBase + k2;
    auto k3Cur = kBase + k3;

    uint32_t v[8];
    v[0] = (m0Cur < m && k0Cur < k) ? in[m0Cur][k0Cur] : 0;
    v[1] = (m0Cur < m && k1Cur < k) ? in[m0Cur][k1Cur] : 0;
    v[2] = (m1Cur < m && k0Cur < k) ? in[m1Cur][k0Cur] : 0;
    v[3] = (m1Cur < m && k1Cur < k) ? in[m1Cur][k1Cur] : 0;
    v[4] = (m0Cur < m && k2Cur < k) ? in[m0Cur][k2Cur] : 0;
    v[5] = (m0Cur < m && k3Cur < k) ? in[m0Cur][k3Cur] : 0;
    v[6] = (m1Cur < m && k2Cur < k) ? in[m1Cur][k2Cur] : 0;
    v[7] = (m1Cur < m && k3Cur < k) ? in[m1Cur][k3Cur] : 0;

    uint32_t pack = (v[7] << 28) | (v[5] << 24) | (v[3] << 20) | (v[1] << 16) |
        (v[6] << 12) | (v[4] << 8) | (v[2] << 4) | v[0];

    out[mTile][kOuterTile][t][innerKTile] = pack;
  }
}

// input is [m][k] (int32 dtype)
// output is [ceil(m / 16)][ceil(k / (InnerKTiles * 16))][32][innerKTiles]
torch::Tensor convert_matrix_to_m16n8k16_Aint4_layout(
    const torch::Tensor& in,
    int64_t innerKTiles = 1) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dim() == 2);
  TORCH_CHECK(in.dtype() == torch::kInt32);
  TORCH_CHECK(in.is_contiguous());

  TORCH_CHECK(innerKTiles == 1 || innerKTiles == 2 || innerKTiles == 4);

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto mTiles = divUp(in.size(0), kMTileSize);

  // kSuperTiles is the number of k-tiles assuming k is innerKTiles * kKTileSize
  auto kSuperTiles = divUp(in.size(1), innerKTiles * kKTileSize);

  // each block handles `innerKTiles` k-tiles.
  // 2 k-tiles are a single int32
  auto out = torch::empty(
      {mTiles, kSuperTiles, 32, innerKTiles},
      torch::TensorOptions().dtype(torch::kInt32).device(in.device()));

  dim3 grid(kSuperTiles, mTiles);

  if (innerKTiles == 1) {
    matrix_to_m16n8k16_Aint4_layout<1><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 2) {
    matrix_to_m16n8k16_Aint4_layout<2><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 4) {
    matrix_to_m16n8k16_Aint4_layout<4><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  }

  return out;
}

// FIXME: parallelize better, smem staging etc?
// input is [n][k] (int32 dtype)
// A layout presents 8 scalar values. 8 x int8 = 64 bits, so there are
// 2 int32s per k-tile
// outut is [ceil(m / 16)][ceil(k / (InnerKTiles * 16))][32][InnerKTiles * 2]
template <int InnerKTiles>
__global__ void matrix_to_m16n8k16_Aint8_layout(
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits> in,
    at::PackedTensorAccessor32<int32_t, 4, at::RestrictPtrTraits> out) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  // gridDim.x corresponds to each k-tile
  auto kTile = blockIdx.x;
  auto mTile = blockIdx.y;
  auto t = threadIdx.x;

  auto m = in.size(0);
  auto k = in.size(1);

  auto m0 = (t / 4);
  auto m1 = m0 + 8;

  auto k0 = (t % 4) * 2;
  auto k1 = k0 + 1;
  auto k2 = k0 + 8;
  auto k3 = k0 + 8 + 1;

  // One k-tile is packed into 2 x int32 at a time
  auto mBase = mTile * kMTileSize;
  auto kBase = kTile * kKTileSize;

  auto m0Cur = mBase + m0;
  auto m1Cur = mBase + m1;

  auto k0Cur = kBase + k0;
  auto k1Cur = kBase + k1;
  auto k2Cur = kBase + k2;
  auto k3Cur = kBase + k3;

  uint32_t v[8];
  v[0] = (m0Cur < m && k0Cur < k) ? in[m0Cur][k0Cur] : 0;
  v[1] = (m0Cur < m && k1Cur < k) ? in[m0Cur][k1Cur] : 0;
  v[2] = (m1Cur < m && k0Cur < k) ? in[m1Cur][k0Cur] : 0;
  v[3] = (m1Cur < m && k1Cur < k) ? in[m1Cur][k1Cur] : 0;
  v[4] = (m0Cur < m && k2Cur < k) ? in[m0Cur][k2Cur] : 0;
  v[5] = (m0Cur < m && k3Cur < k) ? in[m0Cur][k3Cur] : 0;
  v[6] = (m1Cur < m && k2Cur < k) ? in[m1Cur][k2Cur] : 0;
  v[7] = (m1Cur < m && k3Cur < k) ? in[m1Cur][k3Cur] : 0;

  using V2 = VectorType<uint32_t, 2>;
  V2 vOut;

  vOut.vals[0] = (v[3] << 24) | (v[1] << 16) | (v[2] << 8) | v[0];
  vOut.vals[1] = (v[7] << 24) | (v[5] << 16) | (v[6] << 8) | v[4];

  auto outerKTile = kTile / InnerKTiles;
  auto innerKTile = kTile % InnerKTiles;

  *reinterpret_cast<V2*>(&out[mTile][outerKTile][t][innerKTile * 2]) = vOut;
}

// input is [m][k] (int32 dtype)
// output is [ceil(m / 16)][ceil(k / (innerKTiles * 16))][32][innerKTiles * 2]
torch::Tensor convert_matrix_to_m16n8k16_Aint8_layout(
    const torch::Tensor& in,
    int64_t innerKTiles = 1) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(in.dim() == 2);
  TORCH_CHECK(in.dtype() == torch::kInt32);
  TORCH_CHECK(in.is_contiguous());

  // 1 inner k-tile is an 8 byte load
  // 2 inner k-tiles is a 16 byte load
  TORCH_CHECK(innerKTiles == 1 || innerKTiles == 2);

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto mTiles = divUp(in.size(0), kMTileSize);
  auto kTiles = divUp(in.size(1), kKTileSize);
  auto outerKTiles = divUp(kTiles, innerKTiles);

  // each block handles `innerKTiles` k-tiles.
  auto out = torch::empty(
      {mTiles, outerKTiles, 32, 2 * innerKTiles},
      torch::TensorOptions().dtype(torch::kInt32).device(in.device()));

  dim3 grid(kTiles, mTiles);

  if (innerKTiles == 1) {
    matrix_to_m16n8k16_Aint8_layout<1><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  } else if (innerKTiles == 2) {
    matrix_to_m16n8k16_Aint8_layout<2><<<grid, kWarpSize, 0, stream>>>(
        in.packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
        out.packed_accessor32<int32_t, 4, at::RestrictPtrTraits>());
  }

  return out;
}

template <typename T, int Warps, bool KAligned16Byte>
__global__ void matrix_from_m16n8k16_A_layout(
    // size [m / 16][k / 16][32][8]
    const T* __restrict__ in,
    // size [m][k]
    T* __restrict__ out,
    int32_t m,
    int32_t k,
    int32_t mTiles,
    int32_t kTiles) {
  // This implementation only works for 2 byte types at present
  static_assert(sizeof(T) == 2, "");

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto warp = threadIdx.y;
  auto t = threadIdx.x;

  auto kTile = blockIdx.x * Warps + warp;
  auto mTile = blockIdx.y;

  if (kTile >= kTiles) {
    return;
  }

  auto mBase = mTile * kMTileSize;
  auto kBase = kTile * kKTileSize;

  auto m0 = (t / 4);
  auto m1 = m0 + 8;

  auto k0 = (t % 4) * 2;
  auto k1 = k0 + 1;
  auto k2 = k0 + 8;
  auto k3 = k2 + 1;

  using Vec8 = VectorType<T, 8>;
  static_assert(sizeof(Vec8) == sizeof(T) * 8, "");

  auto pIn = in + (((mTile * kTiles) + kTile) * kWarpSize + t) * 8;
  auto inV = *reinterpret_cast<const Vec8*>(pIn);

  constexpr int kSmemRowSize = kKTileSize;

  __shared__ T outTile[Warps][kMTileSize][kSmemRowSize];

  T* m0Tile = &outTile[warp][m0][0];
  T* m1Tile = &outTile[warp][m1][0];

  m0Tile[k0] = inV.vals[0];
  m0Tile[k1] = inV.vals[1];
  m1Tile[k0] = inV.vals[2];
  m1Tile[k1] = inV.vals[3];
  m0Tile[k2] = inV.vals[4];
  m0Tile[k3] = inV.vals[5];
  m1Tile[k2] = inV.vals[6];
  m1Tile[k3] = inV.vals[7];

  __syncwarp();

  if constexpr (KAligned16Byte) {
    // The output matrix is row-aligned to 16 bytes
    static_assert(sizeof(uint4) / sizeof(T) == 8, "");

    // each 2 contiguous lanes handles a tile row
    auto mOutTile = t / 2;
    auto kOutTile = (t % 2) * (sizeof(uint4) / sizeof(T));

    auto mOut = mBase + mOutTile;
    auto kOut = kBase + kOutTile;

    auto pTile = &outTile[warp][mOutTile][kOutTile];
    auto pOut = out + mOut * k + kOut;

    if (mOut < m && kOut < k) {
      *reinterpret_cast<uint4*>(pOut) = *reinterpret_cast<uint4*>(pTile);
    }
  } else {
    // The output matrix is not row-aligned to 16 bytes, instead we just assume
    // word (2 byte) alignment

    // each half warp handles a tile row
    constexpr auto kHalfWarpSize = kWarpSize / 2;
    static_assert(kKTileSize == kHalfWarpSize, "");
    auto mOutTile = t / kHalfWarpSize;
    auto kOutTile = t % kHalfWarpSize;

    auto mOut = mBase + mOutTile;
    auto kOut = kBase + kOutTile;

    auto pTile = &outTile[warp][mOutTile][kOutTile];
    auto pOut = out + mOut * k + kOut;

    if (kOut < k) {
#pragma unroll
      for (int i = 0; i < kMTileSize; i += 2) {
        if (mOut + i < m) {
          pOut[i * k] = pTile[i * kSmemRowSize];
        }
      }
    }
  }
}

// Reverse conversion of convert_matrix_to_m16n8k16_A_layout
//
// input is [ceil(m / 16)][ceil(k / 16)][32][8]
// output is [m][k]
//
// m and k can be arbitrary
torch::Tensor convert_matrix_from_m16n8k16_A_layout(
    const torch::Tensor& in,
    int64_t m,
    int64_t k) {
  c10::cuda::CUDAGuard g(in.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  TORCH_CHECK(in.dtype() == torch::kBFloat16 || in.dtype() == torch::kFloat16);
  TORCH_CHECK(in.is_contiguous());
  TORCH_CHECK(divUp(m, kMTileSize) == in.size(0));
  TORCH_CHECK(divUp(k, kKTileSize) == in.size(1));
  TORCH_CHECK(in.size(2) == kWarpSize);
  TORCH_CHECK(in.size(3) == 8);

  auto mTiles = in.size(0);
  auto kTiles = in.size(1);

  constexpr int kWarps = 2;

  auto out = torch::empty(
      {m, k}, torch::TensorOptions().dtype(in.dtype()).device(in.device()));

  auto grid = dim3(divUp(kTiles, kWarps), mTiles);
  auto block = dim3(kWarpSize, kWarps);

  if (in.dtype() == torch::kFloat16) {
    if (isEvenDivisor(k, 8)) {
      matrix_from_m16n8k16_A_layout<__half, kWarps, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(in.data_ptr()),
              reinterpret_cast<__half*>(out.data_ptr()),
              m,
              k,
              mTiles,
              kTiles);
    } else {
      matrix_from_m16n8k16_A_layout<__half, kWarps, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half*>(in.data_ptr()),
              reinterpret_cast<__half*>(out.data_ptr()),
              m,
              k,
              mTiles,
              kTiles);
    }
  } else if (in.dtype() == torch::kBFloat16) {
    if (isEvenDivisor(k, 8)) {
      matrix_from_m16n8k16_A_layout<__nv_bfloat16, kWarps, true>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
              m,
              k,
              mTiles,
              kTiles);
    } else {
      matrix_from_m16n8k16_A_layout<__nv_bfloat16, kWarps, false>
          <<<grid, block, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
              m,
              k,
              mTiles,
              kTiles);
    }
  }

  return out;
}

} // namespace tinygemm
