// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "StaticUtils.h"
#include "TinyGemm.h"
#include "TinyGemmUtils.cuh"

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

// #define PRINT_PERF

namespace tinygemm {

template <
    FloatType FT,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    int Warps,
    int KTilesPerIteration>
__global__
__launch_bounds__(Warps* kWarpSize) void tinygemm_m16n8k16_chunk_kernel(
    // Data for the A matrix, loaded as per ALayout
    const void* __restrict__ A,

    // Data for the B matrix, loaded as per BLayout
    const void* __restrict__ B,

    // Dequantization information (used for either ALayout or BLayout)
    DequantInfo dqInfo,

    // Output data for the C matrix, stored as per CLayout
    void* __restrict__ C,

    // The size of the matrix multiplication
    int32_t m,
    int32_t n,
    int32_t k,

    // The size of the matrix multiplication, in multiples of our TC tile size
    int32_t mTiles,
    int32_t nTiles,
    int32_t kTiles) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  static_assert(
      ALayout::kMTileSize == kMTileSize && ALayout::kNTileSize == kNTileSize &&
          ALayout::kKTileSize == kKTileSize,
      "");
  static_assert(FT == ALayout::kFT, "");

  static_assert(
      BLayout::kMTileSize == kMTileSize && BLayout::kNTileSize == kNTileSize &&
          BLayout::kKTileSize == kKTileSize,
      "");
  static_assert(FT == BLayout::kFT, "");

  static_assert(
      CLayout::kMTileSize == kMTileSize && CLayout::kNTileSize == kNTileSize &&
          CLayout::kKTileSize == kKTileSize,
      "");
  static_assert(FT == CLayout::kFT, "");

  // The smallest granularity of k-tiles we can process is the maximum of
  // the inner k-tiles for A and B
  constexpr int kMaxInnerKTiles =
      std::max(ALayout::kInnerKTiles, BLayout::kInnerKTiles);

  // We always process at least kInnerKTiles k-tiles back to back in a warp
  static_assert(isEvenDivisor(KTilesPerIteration, kMaxInnerKTiles), "");

  __shared__ char
      smemAC[std::max({ALayout::kSharedMemory, CLayout::kSharedMemory, 16})];
  __shared__ char smemB[std::max({BLayout::kSharedMemory, 16})];

  auto warpId = threadIdx.y;
  auto laneId = threadIdx.x;

  int32_t mTile = blockIdx.z;
  int32_t nTile = blockIdx.y;

  //
  // Accumulators
  // We don't simply accumulate into a single float4 accumulator as this
  // is a too-strong execution dependency. Instead, we use two accumulators
  // with one last combination when done. c[0] is the final accumulated value
  // within the warp before cross-warp reduction.
  //
  float4 c[2];

#pragma unroll
  for (int j = 0; j < 2; ++j) {
    c[j] = float4{0.0f, 0.0f, 0.0f, 0.0f};
  }

  // Global dequantization for layout objects, if there is any
  // (otherwise a no-op)
  auto aLayoutInit = ALayout::init(warpId, laneId, mTile, nTile, dqInfo);
  auto bLayoutInit = BLayout::init(warpId, laneId, mTile, nTile, dqInfo);

  // This warp starts processing this k-tile
  // gridDim.x > 1 means we are performing cross-block split-k reduction
  int32_t kTileBase = (blockIdx.x * Warps + warpId) * KTilesPerIteration;

  // First, handle whole multiples of KTilesPerIteration
  int32_t kTilesLimit = roundDown(kTiles, KTilesPerIteration);

  // Adjust A + B data pointers to avoid unnecessary pointer arithmetic
  // in the inner loop
  auto curA = ALayout::getMatrixTile(
      warpId, laneId, smemAC, A, m, k, mTiles, mTile, kTiles, kTileBase);
  auto curB = BLayout::getMatrixTile(
      warpId, laneId, smemB, B, n, k, nTiles, nTile, kTiles, kTileBase);

  // Each warp handles a set of KTilesPerIteration under the above limit
  for (; kTileBase < kTilesLimit; kTileBase += Warps * KTilesPerIteration) {
    //
    // Load data from A
    //
    typename ALayout::template LoadT<KTilesPerIteration> aLoad;
    ALayout::template load<KTilesPerIteration>(
        warpId,
        laneId,
        smemAC,
        curA,
        dqInfo,
        m,
        k,
        mTiles,
        mTile,
        kTiles,
        kTileBase,
        aLoad);

    //
    // Load data from B
    //
    typename BLayout::template LoadT<KTilesPerIteration> bLoad;
    BLayout::template load<KTilesPerIteration>(
        warpId,
        laneId,
        smemB,
        curB,
        dqInfo,
        n,
        k,
        nTiles,
        nTile,
        kTiles,
        kTileBase,
        bLoad);

    curA = ALayout::incrementMatrixTile(
        warpId,
        laneId,
        smemAC,
        curA,
        m,
        k,
        mTiles,
        mTile,
        kTiles,
        Warps * KTilesPerIteration);
    curB = BLayout::incrementMatrixTile(
        warpId,
        laneId,
        smemB,
        curB,
        n,
        k,
        nTiles,
        nTile,
        kTiles,
        Warps * KTilesPerIteration);

    // Prevent nvcc/ptxas from moving the memory loads above
    // asm volatile("fence.cta;");
    __syncwarp();

    u32x4 a[KTilesPerIteration];
    ALayout::template dequant<KTilesPerIteration>(
        warpId, laneId, smemAC, aLayoutInit, aLoad, a);

    u32x2 b[KTilesPerIteration];
    BLayout::template dequant<KTilesPerIteration>(
        warpId, laneId, smemB, bLayoutInit, bLoad, b);

    //
    // Now, perform the matrix multiplication
    //

    // We accumulate across k-tiles here
    static_assert(isEvenDivisor(KTilesPerIteration, 2), "");
#pragma unroll
    for (int i = 0; i < KTilesPerIteration / 2; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        FloatDefs<FT>::mma(c[j], a[i * 2 + j], b[i * 2 + j], c[j]);
      }
    }
  } // for all tiles under kTilesLimit

  // There could be a remainder of tiles (we are guaranteed to have a multiple
  // of kMaxInnerKTiles present, which is not guaranteed to be a multiple of
  // KTilesPerIteration. Handle the remainder tiles
  static_assert(Warps >= KTilesPerIteration / kMaxInnerKTiles, "");

  auto kTileBaseRemaining = kTilesLimit + warpId * kMaxInnerKTiles;

  // If we have any remainder k-tiles, some warps will handle them, processing
  // kInnerKTiles k-tiles at a time
  if (kTileBaseRemaining < kTiles) {
    curA = ALayout::getMatrixTile(
        warpId,
        laneId,
        smemAC,
        A,
        m,
        k,
        mTiles,
        mTile,
        kTiles,
        kTileBaseRemaining);
    curB = BLayout::getMatrixTile(
        warpId,
        laneId,
        smemB,
        B,
        n,
        k,
        nTiles,
        nTile,
        kTiles,
        kTileBaseRemaining);

    typename ALayout::template LoadT<kMaxInnerKTiles> aLoad;
    ALayout::template load<kMaxInnerKTiles>(
        warpId,
        laneId,
        smemAC,
        curA,
        dqInfo,
        m,
        k,
        mTiles,
        mTile,
        kTiles,
        kTileBaseRemaining,
        aLoad);

    u32x4 a[kMaxInnerKTiles];
    ALayout::template dequant<kMaxInnerKTiles>(
        warpId, laneId, smemAC, aLayoutInit, aLoad, a);

    typename BLayout::template LoadT<kMaxInnerKTiles> bLoad;
    BLayout::template load<kMaxInnerKTiles>(
        warpId,
        laneId,
        smemB,
        curB,
        dqInfo,
        n,
        k,
        nTiles,
        nTile,
        kTiles,
        kTileBaseRemaining,
        bLoad);

    u32x2 b[kMaxInnerKTiles];
    BLayout::template dequant<kMaxInnerKTiles>(
        warpId, laneId, smemB, bLayoutInit, bLoad, b);

    if constexpr (kMaxInnerKTiles == 1) {
      // just accumulate directly into c
      FloatDefs<FT>::mma(c[0], a[0], b[0], c[0]);
    } else {
      // FIXME: what about kMaxInnerKTiles == 2?
      static_assert(isEvenDivisor(kMaxInnerKTiles, 2), "");
#pragma unroll
      for (int i = 0; i < kMaxInnerKTiles / 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          FloatDefs<FT>::mma(c[j], a[i * 2 + j], b[i * 2 + j], c[j]);
        }
      }
    }
  }

  // Final warp sum into c[0]
  c[0].x += c[1].x;
  c[0].y += c[1].y;
  c[0].z += c[1].z;
  c[0].w += c[1].w;

  //
  // Reduce independent k-tiles (same m/n) across warps
  //
  __shared__ float4 smem_sum[Warps][kWarpSize];
  smem_sum[warpId][laneId] = c[0];

  __syncthreads();

  // only the first warp sums across warps
  if (warpId != 0) {
    return;
  }

  // the first warp can just preserve its prior value;
  // the rest we need to load
  float4 sum_f32 = c[0];

  // offset [0][laneId]
  auto smemLane = &smem_sum[0][laneId];

  // Reduce across the block in the first warp
#pragma unroll
  for (int i = 1; i < Warps; ++i) {
    float4 v = *(smemLane + i * kWarpSize);
    sum_f32.x += v.x;
    sum_f32.y += v.y;
    sum_f32.z += v.z;
    sum_f32.w += v.w;
  }

  // Write the reduced result (in the first warp) into the output
  CLayout::store(
      warpId, laneId, smemAC, C, m, n, mTiles, mTile, nTiles, nTile, sum_f32);
}

template <
    FloatType FT,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    int Warps,
    int KTilesPerWarp>
void launch_tinygemm_kernel(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const DequantInfo& dqInfo,
    torch::Tensor& C_final,
    int32_t mTiles,
    int32_t nTiles,
    int32_t kTiles,
    int32_t m,
    int32_t n,
    int32_t k,
    cudaStream_t stream) {
  // The chunking kernel requires that kTiles is a multiple of the larger of
  // A / B innerKTiles, which is the smallest number of tiles along the k
  // dimension that we can process
  constexpr int kMaxInnerKTiles =
      std::max(ALayout::kInnerKTiles, BLayout::kInnerKTiles);

  TORCH_CHECK(isEvenDivisor(kTiles, kMaxInnerKTiles));
  TORCH_CHECK(isEvenDivisor(KTilesPerWarp, kMaxInnerKTiles));

  // The k dimension must always be a multiple of 32
  TORCH_CHECK(isEvenDivisor(k, 32));

  // After intra-block reduction across the k dimension, we are left with this
  // many tiles
  //  int32_t postKernelKTiles = kTiles / (Warps * KTilesPerWarp);
  int32_t postKernelKTiles = 1; // we loop

  auto grid = dim3(postKernelKTiles, nTiles, mTiles);
  auto block = dim3(kWarpSize, Warps);

#ifdef PRINT_PERF
  auto startTimer = CudaEvent(stream, true);
#endif

  auto func = tinygemm_m16n8k16_chunk_kernel<
      FT,
      ALayout,
      BLayout,
      CLayout,
      Warps,
      KTilesPerWarp>;

  func<<<grid, block, 0, stream>>>(
      A.data_ptr(),
      B.data_ptr(),
      dqInfo,
      C_final.data_ptr(),
      m,
      n,
      k,
      mTiles,
      nTiles,
      kTiles);

#ifdef PRINT_PERF
  auto endTimer = CudaEvent(stream, true);
  auto time = endTimer.timeFrom(startTimer) * 1e-3f;

  cudaFuncAttributes funcAttr;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&funcAttr, func));

  printf(
      "tinygemm (m %d n %d k %d tilesPerWarp %d) time %.5f us %.5f tflop/s (grid %d %d %d regs %d final split-k %d)\n",
      m,
      n,
      k,
      KTilesPerWarp,
      time * 1e6,
      (float(m) * n * k * 2) / (time * 1e12),
      grid.x,
      grid.y,
      grid.z,
      funcAttr.numRegs,
      postKernelKTiles);
#endif
}

} // namespace tinygemm
