// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "Dequantization.cuh"
#include "FloatDefs.cuh"
#include "StaticUtils.h"

namespace tinygemm {

#define USE_ITER_B

//
// B matrix layouts
//

// Loads the B matrix in 16-bit standard n x k row major layout, and writes
// the C matrix in 16-bit standard m x n row major layout:
//
// load size [n][k]
// FIXME: assumes that k is a multiple of kKTileSize
// store size [m][n]
// FIXME: assumes n is a multiple of kKTileSize
//
// This version does not use smem for transposition (original version)
template <int Warps, FloatType FT>
struct BLayout_RM_NoSmem {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = 1;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  static constexpr int32_t kSharedMemory = 0;
  static constexpr bool kSyncAfterInit = false;

  // Dequantization information needed for all loads, if any
  struct InitT {};

  // Raw data type of matrix (before dequantization) and
  // data needed for dequantization
  template <int KTilesToLoad>
  struct LoadT {
    u32x2 data[KTilesToLoad];
    // no q info
  };

  static __device__ InitT init(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      int32_t mTile,
      int32_t nTile,
      const DequantInfo& dqInfo) {
    return InitT();
  };

  static __device__ const void* getMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_B
    auto nLane = nTile * kNTileSize + (laneId / 4);
    auto kLane = kTileStart * kKTileSize + (laneId % 4) * 2;

    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) + nLane * k +
        kLane;
#else
    return B;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_B
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        kTileIncrement * kKTileSize;
#else
    return B;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      // unused
      const DequantInfo& dqInfo,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    auto nLane = nTile * kNTileSize + (laneId / 4);
    auto kLane = kTileStart * kKTileSize + (laneId % 4) * 2;

#ifdef USE_ITER_B
    auto bPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(B);
#else
    auto bPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        nLane * k + kLane;
#endif

    // Lane has all values along the same n dimension
    bool nInBounds = nLane < n;

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out.data[i].vals[0] = nInBounds
          ? *reinterpret_cast<const uint32_t*>(bPtr + i * kKTileSize)
          : uint32_t(0);
      out.data[i].vals[1] = nInBounds
          ? *reinterpret_cast<const uint32_t*>(bPtr + 8 + i * kKTileSize)
          : uint32_t(0);
    }
  }

  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x2 out[KTilesToLoad]) {
    // should be a no-op
#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out[i] = in.data[i];
    }
  }

  static __device__ void store(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      void* __restrict__ C,
      // B storage output is transposed (as we are computing (W x^t)^t)
      // so `m` is the number of cols, `n` is the number of rows
      int32_t m,
      int32_t n,
      int32_t mOutTiles,
      int32_t mTile,
      int32_t nOutTiles,
      int32_t nTile,
      const float4& out) {
    auto laneId_div4 = laneId / 4;
    auto laneId_mod4 = laneId % 4;

    // tNc0 goes at row (N % 4) * 2 col N / 4
    // tNc1 goes at row (N % 4) * 2 + 1 col N / 4
    // tNc2 goes at row (N % 4) * 2 col (N / 4) + 8
    // tNc3 goes at row (N % 4) * 2 + 1 col (N / 4) + 8

    auto out0 = FloatDefs<FT>::floatToT(out.x);
    auto out1 = FloatDefs<FT>::floatToT(out.y);
    auto out2 = FloatDefs<FT>::floatToT(out.z);
    auto out3 = FloatDefs<FT>::floatToT(out.w);

    auto outRow = nTile * kNTileSize;
    auto outCol = mTile * kMTileSize;

    auto cPtr =
        reinterpret_cast<typename FloatDefs<FT>::T*>(C) + outRow * m + outCol;

    auto tileRow0 = (laneId_mod4 * 2);
    auto tileRow1 = (tileRow0 + 1);

    bool tileRow0_valid = (outRow + tileRow0) < n;
    bool tileRow1_valid = (outRow + tileRow1) < n;

    tileRow0 *= m;
    tileRow1 *= m;

    auto tileCol0 = laneId_div4;
    auto tileCol1 = tileCol0 + 8;

    if (tileRow0_valid) {
      *(cPtr + tileRow0 + tileCol0) = out0;
      *(cPtr + tileRow0 + tileCol1) = out2;
    }

    if (tileRow1_valid) {
      *(cPtr + tileRow1 + tileCol0) = out1;
      *(cPtr + tileRow1 + tileCol1) = out3;
    }
  }
};

// Loads the B matrix in 16-bit standard n x k row major layout, and writes
// the C matrix in 16-bit standard m x n row major layout:
//
// load size [n][k]
// FIXME: assumes that k is a multiple of kKTileSize
// store size [m][n]
// FIXME: assumes n is a multiple of kKTileSize
//
// This version uses smem for transposition from RM to TC layout
template <int Warps, FloatType FT>
struct BLayout_RM {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = 1;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  static constexpr int32_t kSharedMemory =
      sizeof(typename FloatDefs<FT>::T) * Warps * kNTileSize * kKTileSize;
  static constexpr bool kSyncAfterInit = false;

  // Dequantization information needed for all loads, if any
  struct InitT {};

  // Raw data type of matrix (before dequantization) and
  // data needed for dequantization
  template <int KTilesToLoad>
  struct LoadT {
    // Single tile is 8 x 16 (8 x 32 bytes)
    // Multiple tiles are 8 x 16 x KTilesToLoad
    u32x2 data[KTilesToLoad];
    // no q info
  };

  static __device__ InitT init(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      int32_t mTile,
      int32_t nTile,
      const DequantInfo& dqInfo) {
    return InitT();
  };

  static __device__ const void* getMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart) {
    auto nLane = nTile * kNTileSize + (laneId / 4);

    // we load 8 byte words
    auto kLane = kTileStart * kKTileSize + (laneId % 4) * 4;

    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) + nLane * k +
        kLane;
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        kTileIncrement * kKTileSize;
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      // unused
      const DequantInfo& dqInfo,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    // FIXME: we can precompute this and know whether we are always valid or not
    auto nLane = nTile * kNTileSize + (laneId / 4);

    // // Lane has all values along the same n dimension
    // // FIXME: load garbage instead? same index?
    bool nInBounds = nLane < n;

    auto bPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(B);

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out.data[i] = nInBounds
          ? *reinterpret_cast<const u32x2*>(bPtr + i * kKTileSize)
          : u32x2{0, 0};
    }
  }

  // There is no dequantization here, but dequant happens after all memorry
  // loads are waited upon
  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x2 out[KTilesToLoad]) {
    auto nLane = laneId / 4;
    auto kLane = (laneId % 4);

    // array is [Warps][kNTileSize][kKTileSize]
    // this lane accesses [warpId][nLane][0]
    auto smemBase = reinterpret_cast<typename FloatDefs<FT>::T*>(smem) +
        ((warpId * kNTileSize) + nLane) * kKTileSize;

    auto smemWrite = smemBase + kLane * 4;
    auto smemRead = smemBase + kLane * 2;

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      *reinterpret_cast<u32x2*>(smemWrite) = in.data[i];
      __syncwarp();

      // b0b1
      out[i].vals[0] = *reinterpret_cast<const uint32_t*>(smemRead);
      // b2b3
      out[i].vals[1] = *reinterpret_cast<const uint32_t*>(smemRead + 8);
      __syncwarp();
    }
  }

  static __device__ void store(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      void* __restrict__ C,
      // B storage output is transposed (as we are computing (W x^t)^t)
      // so `m` is the number of cols, `n` is the number of rows
      int32_t m,
      int32_t n,
      int32_t mOutTiles,
      int32_t mTile,
      int32_t nOutTiles,
      int32_t nTile,
      const float4& out) {
    auto laneId_div4 = laneId / 4;
    auto laneId_mod4 = laneId % 4;

    // tNc0 goes at row (N % 4) * 2 col N / 4
    // tNc1 goes at row (N % 4) * 2 + 1 col N / 4
    // tNc2 goes at row (N % 4) * 2 col (N / 4) + 8
    // tNc3 goes at row (N % 4) * 2 + 1 col (N / 4) + 8

    auto out0 = FloatDefs<FT>::floatToT(out.x);
    auto out1 = FloatDefs<FT>::floatToT(out.y);
    auto out2 = FloatDefs<FT>::floatToT(out.z);
    auto out3 = FloatDefs<FT>::floatToT(out.w);

    auto outRow = nTile * kNTileSize;
    auto outCol = mTile * kMTileSize;

    auto cPtr =
        reinterpret_cast<typename FloatDefs<FT>::T*>(C) + outRow * m + outCol;

    auto tileRow0 = (laneId_mod4 * 2);
    auto tileRow1 = (tileRow0 + 1);

    bool tileRow0_valid = (outRow + tileRow0) < n;
    bool tileRow1_valid = (outRow + tileRow1) < n;

    tileRow0 *= m;
    tileRow1 *= m;

    auto tileCol0 = laneId_div4;
    auto tileCol1 = tileCol0 + 8;

    if (tileRow0_valid) {
      *(cPtr + tileRow0 + tileCol0) = out0;
      *(cPtr + tileRow0 + tileCol1) = out2;
    }

    if (tileRow1_valid) {
      *(cPtr + tileRow1 + tileCol0) = out1;
      *(cPtr + tileRow1 + tileCol1) = out3;
    }

    /*
    // FIXME: to what degree are bank conflicts a problem (no +1)
    // vs minimizing the number of gmem writes and/or using vectorized
    // gmem writes (4/8/16 byte aligned rows)?
    // FIXME: move to dynamic smem
    __shared__ typename FloatDefs<FT>::T smemTile[kNTileSize][kMTileSize];


    {
      typename FloatDefs<FT>::T2 out01 = __float22bfloat162_rn(float2{out.x,
    out.y}); typename FloatDefs<FT>::T2 out23 =
    __float22bfloat162_rn(float2{out.z, out.w});

      auto tileRow0 = laneId_mod4 * 2;
      auto tileRow1 = tileRow0 + 1;
      auto tileCol0 = laneId_div4;
      auto tileCol1 = tileCol0 + 8;

      smemTile[tileRow0][tileCol0] = out01.x;
      smemTile[tileRow1][tileCol0] = out01.y;
      smemTile[tileRow0][tileCol1] = out23.x;
      smemTile[tileRow1][tileCol1] = out23.y;
    }

    __syncwarp();

    {
      auto tileRow = laneId_div4;
      auto tileCol = laneId_mod4 * 4;

      auto outRow = nTile * kNTileSize + tileRow;
      auto outCol = mTile * kMTileSize + tileCol;

      if (outRow < n) {
        auto cPtr = reinterpret_cast<typename FloatDefs<FT>::T*>(C) +
          outRow * m + outCol;

        // Each lane writes out 8 bytes (4 values) at a time
        // 16 x 8
        auto v = *reinterpret_cast<const
    bf16x2x2*>(&smemTile[tileRow][tileCol]); *reinterpret_cast<bf16x2x2*>(cPtr)
    = v;
      }
    }

    */
  }
};

template <int Warps, FloatType FT, int InnerKTiles>
struct BLayout_TC {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = InnerKTiles;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  static constexpr int32_t kSharedMemory =
      sizeof(typename FloatDefs<FT>::T) * kMTileSize * kNTileSize;
  static constexpr bool kSyncAfterInit = false;

  // Dequantization information needed for all loads, if any
  struct InitT {};

  // Raw data type of matrix (before dequantization) and
  // data needed for dequantization
  template <int KTilesToLoad>
  struct LoadT {
    u32x2 data[KTilesToLoad];
    // no q info
  };

  static __device__ InitT init(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      int32_t mTile,
      int32_t nTile,
      const DequantInfo& dqInfo) {
    return InitT();
  };

  static __device__ const void* getMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_B
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        // dim0 index
        ((nTile * (kTiles / InnerKTiles) +
          // dim1 index
          (kTileStart / InnerKTiles)) *
             kWarpSize +
         // dim2 index
         laneId) *
        (InnerKTiles * 4);
#else
    return B;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_B
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        (kTileIncrement / InnerKTiles) * kWarpSize * (InnerKTiles * 4);
#else
    return B;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles * 4]
      // n / 8: n-tiles (n8)
      // k / (InnerKTiles * 16): k "super-tiles"
      // 32: value per warp lane
      // (InnerKTiles * 4): 4 bf16/fp16 values per k-tile innermost
      const void* __restrict__ B,
      const DequantInfo& dqInfo,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    // Each lane presents 4 values in the B layout
    // InnerKTiles == 1 is 8 bytes, == 2 is 16 bytes
    static_assert(InnerKTiles == 1 || InnerKTiles == 2, "");

#ifdef USE_ITER_B
    auto bPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(B);
#else
    auto bPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(B) +
        // dim0 index
        ((nTile * (kTiles / InnerKTiles) +
          // dim1 index
          (kTileStart / InnerKTiles)) *
             kWarpSize +
         // dim2 index
         laneId) *
            (InnerKTiles * 4);
#endif

    if constexpr (InnerKTiles == 1) {
#pragma unroll
      for (int i = 0; i < KTilesToLoad; ++i) {
        auto v = *reinterpret_cast<const u32x2*>(
            bPtr + i * (kWarpSize * InnerKTiles * 4));

        out.data[i].vals[0] = v.vals[0];
        out.data[i].vals[1] = v.vals[1];
      }
    } else if constexpr (InnerKTiles == 2) {
#pragma unroll
      for (int i = 0; i < KTilesToLoad / kInnerKTiles; ++i) {
        auto v = *reinterpret_cast<const u32x4*>(
            bPtr + i * (kWarpSize * InnerKTiles * 4));

        out.data[i * kInnerKTiles + 0].vals[0] = v.vals[0];
        out.data[i * kInnerKTiles + 0].vals[1] = v.vals[1];
        out.data[i * kInnerKTiles + 1].vals[0] = v.vals[2];
        out.data[i * kInnerKTiles + 1].vals[1] = v.vals[3];
      }
    }
  }

  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x2 out[KTilesToLoad]) {
    // should be a no-op
#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out[i] = in.data[i];
    }
  }

  static __device__ void store(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles * 4]
      void* __restrict__ C,
      int32_t m, // B k dimension
      int32_t n, // B n dimension
      int32_t mOutTiles,
      int32_t mTile,
      int32_t nOutTiles,
      int32_t nTile,
      const float4& out) {
    auto row = laneId / 4;
    auto col = (laneId % 4) * 2;

    // FIXME: to what degree are bank conflicts a problem (no +1)
    // vs minimizing the number of gmem writes and/or using vectorized
    // gmem writes (4/8/16 byte aligned rows)?

    // smem is [kMTileSize][kNTIleSize]
    // [row][col]
    auto smemLaneIn = reinterpret_cast<typename FloatDefs<FT>::T*>(smem) +
        (row * kNTileSize) + col;

    // (row, col)
    *reinterpret_cast<typename FloatDefs<FT>::T2*>(smemLaneIn) =
        FloatDefs<FT>::float2ToT2(float2{out.x, out.y});
    // (row + 8, col)
    *reinterpret_cast<typename FloatDefs<FT>::T2*>(
        smemLaneIn + 8 * kNTileSize) =
        FloatDefs<FT>::float2ToT2(float2{out.z, out.w});

    __syncwarp();

    auto smemLaneOut = reinterpret_cast<typename FloatDefs<FT>::T*>(smem) +
        (col * kNTileSize) + row;

    auto c0c1c2c3 = typename FloatDefs<FT>::T4{// (col + 0, row)
                                               *smemLaneOut,
                                               // (col + 1, row)
                                               *(smemLaneOut + 1 * kNTileSize),
                                               // (col + 8, row)
                                               *(smemLaneOut + 8 * kNTileSize),
                                               // (col + 9, row)
                                               *(smemLaneOut + 9 * kNTileSize)};

    // C is presented as m16n8
    // In B format:
    // the `n` dimension in the C output is the `n` dimension of the B output
    // the `m` dimension in the C output is the `k` dimension of the B output
    // so the m-tiles become k-tiles, which are 16 in both layouts.
    // We pack k16 into each InnerKTile
    auto outerKTiles = divUp(mOutTiles, InnerKTiles);
    auto outerKTile = mTile / InnerKTiles;
    auto innerKTile = mTile % InnerKTiles;

    auto bPtr = reinterpret_cast<typename FloatDefs<FT>::T*>(C) +
        // dim0 index
        ((nTile * outerKTiles +
          // dim1 index
          outerKTile) *
             kWarpSize +
         // dim2 index
         laneId) *
            (InnerKTiles * 4) +
        // dim3 index
        innerKTile * 4;

    *reinterpret_cast<typename FloatDefs<FT>::T4*>(bPtr) = c0c1c2c3;
  }
};

template <
    int Warps,
    FloatType FT,
    int InnerKTiles,
    int QGroupSize,
    Int4_QType QType>
struct BLayout_TC_int4 {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = InnerKTiles;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  // 8 x 16 mma tile => need 8 rows of any4 data
  static constexpr int32_t kSharedMemory =
      QType == Int4_QType::Any4_RowWise_Grouped
      ? sizeof(typename FloatDefs<FT>::T) * 8 * 16
      : 0;
  // need sync for the any4 smem LUTs
  static constexpr bool kSyncAfterInit =
      (QType == Int4_QType::Any4_RowWise_Grouped);

  // Dequantization information needed for all loads, if any
  struct InitT {
    // If QType == Any4_Global_Grouped, then across the warp this
    // holds the 16 float dequantized values that each int4 value
    // maps to, prior to group dequantization
    // The 16 values are replicated across both half warps.
    //
    // FIXME: nvcc should remove these if they are not used for the given QType
    typename FloatDefs<FT>::T any4LUT;

    // If QType == MX4_Grouped, then across the warp this holds the
    // 16 float dequantized values for the fp4 MX4 values.
    // It seems faster to perform a lookup via register shuffle
    // than to do the arithmetic to convert MX4 fp4 values to f16.
    // The dequant values are the same for all matrix entries.
    // The 16 values are replicated across both half warps.
    //
    // FIXME: nvcc should remove these if they are not used for the given QType
    typename FloatDefs<FT>::T dequantMX4;
  };

  // Raw data type of matrix (before dequantization) and
  // data needed for dequantization
  template <int KTilesToLoad>
  struct LoadT {
    static constexpr int kKTilesPerQGroup = (QGroupSize / kKTileSize);
    // a q-group could be larger than what we are handling in a single warp
    static constexpr int kNumQGroups = (KTilesToLoad / kKTilesPerQGroup) < 1
        ? 1
        : (KTilesToLoad / kKTilesPerQGroup);

    // 2 k-tiles per int32 (B TC layout has 4 int4 words or 16 bits per k-tile)
    uint32_t data[KTilesToLoad / 2];

    // Group quantization scale/offset data for
    // Int4_Grouped / Any4_Global_Grouped / Any4_RowWise_Grouped
    //
    // The B layout only has values at row m, hence a single scale/offset.
    //
    // FIXME: nvcc should remove this if they are not used for the given QType
    typename FloatDefs<FT>::T2 qScaleAndZero[kNumQGroups];

    // Row-wise exponent for MX4_QType
    //
    // The B layout only has values at row m, hence a single value.
    // This is loaded as uint8_t and dequantized later into a float multiplier
    //
    // FIXME: nvcc should remove this if they are not used for the given QType
    uint8_t mx4Exponent[kNumQGroups];
  };

  static __device__ InitT init(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      int32_t mTile,
      int32_t nTile,
      const DequantInfo& dqInfo) {
    // qInfo2 is [16] : float representing what each int4 value dequantizes to
    InitT out;

    if constexpr (QType == Int4_QType::Any4_Global_Grouped) {
      auto any4LUT =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo2);

      // Just a single table of 16
      out.any4LUT = any4LUT[laneId % 16];

    } else if constexpr (QType == Int4_QType::Any4_RowWise_Grouped) {
      // We need to load 8 rows of data starting at this offset
      int32_t nStart = nTile * kNTileSize;

      auto any4LUT =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo2) +
          nStart * 16; // dqInfo.iInfo1;

      auto tid = threadIdx.y * kWarpSize + threadIdx.x;
      auto smemT = reinterpret_cast<typename FloatDefs<FT>::T*>(smem);

      // FIXME: might be better to have an
      // 8 x 256 -> (b)f16x2 LUT with dequantization
      // codes arranged as 76543210, can dequantize 2 values per smem lookup
      if (tid < 8 * 16) {
        smemT[tid] = any4LUT[tid];
      }

      // __syncthreads() is called by main kernel after all inits
    } else if constexpr (QType == Int4_QType::MX4_Grouped) {
      // We simply store the MX4 fp4 dequant values in device memory, is faster
      // than doing the bit manipulation to convert to a float
      // duplicate values across the lower and upper warp
      out.dequantMX4 = FloatDefs<FT>::floatToT(kMX4_Values[laneId % 16]);
    }

    return out;
  }

  static __device__ const void* getMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_B
    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
    return reinterpret_cast<const int32_t*>(B) +
        (((nTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
        (InnerKTiles / 2);
#else
    return B;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_B
    return reinterpret_cast<const int32_t*>(B) +
        kTileIncrement / InnerKTiles * kWarpSize * (InnerKTiles / 2);
#else
    return B;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // type uint32, size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles / 2]
      // n / 8: n-tiles (n8)
      // k / (InnerKTiles * 16): TC size per k-tile is 16 (m16n8k16)
      // 32: value per warp lane
      // (InnerKTiles / 2): B layout has 4 values per lane (16 bits) per k-tile.
      // 2 k-tiles packed is a uint32x1
      // (4 bits x 4 B-loads for m16n8k16 x 2 k-tiles = 32 bits)
      // 4 k-tiles packed is a uint32x2 (64 bits)
      // 8 k-tiles packed is a uint32x4 (128 bits)
      const void* __restrict__ B,
      // size [k / qGroupSize][n][2]
      // Contains the scale and zero point of each of the quantized int4 values
      // within B
      // v_reconstructed = (bf16/fp16(B_int4_val) * scale) - zero
      const DequantInfo& dqInfo,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    static_assert(InnerKTiles == 2 || InnerKTiles == 4 || InnerKTiles == 8);

#ifdef USE_ITER_B
    auto bPtr = reinterpret_cast<const int32_t*>(B);
#else
    auto bPtr = reinterpret_cast<const int32_t*>(B) +
        (((nTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
            (InnerKTiles / 2);
#endif

    //
    // Load k-tiles of quantized data from the B matrix
    //

    static_assert(KTilesToLoad >= 2 && isPowerOf2(KTilesToLoad), "");

#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
      auto bPtrCur = bPtr + i * kWarpSize * (InnerKTiles / 2);

      if constexpr (InnerKTiles == 2) {
        //         auto v = *reinterpret_cast<const i32x1*>(bPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles / 2; ++j) {
        //           out.data[i * (InnerKTiles / 2) + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.u32 {%0}, [%1];"
                     : "=r"(out.data[i * (InnerKTiles / 2)])
                     : "l"(bPtrCur));

      } else if constexpr (InnerKTiles == 4) {
        //         auto v = *reinterpret_cast<const i32x2*>(bPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles / 2; ++j) {
        //           out.data[i * (InnerKTiles / 2) + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
                     : "=r"(out.data[i * (InnerKTiles / 2) + 0]),
                       "=r"(out.data[i * (InnerKTiles / 2) + 1])
                     : "l"(bPtrCur));
      } else if constexpr (InnerKTiles == 8) {
        //         auto v = *reinterpret_cast<const i32x4*>(bPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles / 2; ++j) {
        //           out.data[i * (InnerKTiles / 2) + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(out.data[i * (InnerKTiles / 2) + 0]),
                       "=r"(out.data[i * (InnerKTiles / 2) + 1]),
                       "=r"(out.data[i * (InnerKTiles / 2) + 2]),
                       "=r"(out.data[i * (InnerKTiles / 2) + 3])
                     : "l"(bPtrCur));
      }
    }

    //
    // Load needed info for dequantization
    //

    // each lane has values only at row laneN
    int32_t laneN = nTile * kNTileSize + (laneId / 4);

    if constexpr (
        QType == Int4_QType::Int4_Grouped ||
        QType == Int4_QType::Any4_Global_Grouped ||
        QType == Int4_QType::Any4_RowWise_Grouped ||
        QType == Int4_QType::MX4_Grouped) {
      static_assert(isPowerOf2(QGroupSize), "");
      static_assert(isEvenDivisor(QGroupSize, kKTileSize), "");
      // smallest quantization group size is 32 (2 k-tiles are packed in an
      // int32)
      static_assert(QGroupSize >= kKTileSize * 2, "");

      int32_t groupStart = (kTileStart * kKTileSize) / QGroupSize;

      if constexpr (QType != Int4_QType::MX4_Grouped) {
        // int4/any4 have float scale + offset
        // offset [qScale_kGroup][qScale_n][0]

        auto qInfoPtr =
            reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo1) +
            (groupStart * n + laneN) * 2;

#pragma unroll
        for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
          out.qScaleAndZero[i] =
              *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                  qInfoPtr + i * n * 2);
        }
      } else if constexpr (QType == Int4_QType::MX4_Grouped) {
        // mx4 has a uint8 exponent

        // all values for the lane share the same row
        auto expPtr = reinterpret_cast<const uint8_t*>(dqInfo.qInfo1) +
            laneN * dqInfo.iInfo1 + groupStart;

#pragma unroll
        for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
          // FIXME: transpose exponents?
          out.mx4Exponent[i] = expPtr[i];
        }
      }
    }
  }

  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x2 out[KTilesToLoad]) {
    static_assert(isEvenDivisor(KTilesToLoad, 2), "");

    if constexpr (
        QType == Int4_QType::Int4_Grouped ||
        QType == Int4_QType::Any4_Global_Grouped ||
        QType == Int4_QType::Any4_RowWise_Grouped) {
      //
      // De-quantize int4 values to bf16/fp16. Values are dequantized as truly
      // int4
      // [-8, 7] range; dequant = (bf16/fp16(int4_value) * scale) + zero
      //

      // FIXME: does this negatively affect register counts, or will nvcc
      // move this expansion (and data loads above) closer to the point of use?
      typename FloatDefs<FT>::T2 qScale[LoadT<KTilesToLoad>::kNumQGroups];
      typename FloatDefs<FT>::T2 qZero[LoadT<KTilesToLoad>::kNumQGroups];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        qScale[i] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i].x);
        qZero[i] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i].y);
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad / 2; ++i) {
        typename FloatDefs<FT>::T2x4 v;

        // 8 x int4 -> 8 x bf16/fp16
        if constexpr (QType == Int4_QType::Any4_Global_Grouped) {
          convert_any4x8_global_to_f16x2x4<FT>(in.data[i], init.any4LUT, v);
        } else if constexpr (QType == Int4_QType::Any4_RowWise_Grouped) {
          convert_any4x8_rowwise_B_to_f16x2x4<FT>(
              laneId,
              in.data[i],
              reinterpret_cast<const typename FloatDefs<FT>::T*>(smem),
              v);
        } else {
          convert_i4x8_to_f16x2x4(in.data[i], v);
        }

        auto curKTile = i * 2;
        // q-group sizes are at least kKTileSize * 2, so this is ok
        // (won't be split across two different q-groups)
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // The dequantized values in `v` for a given lane have the same n
        // dimension (the B tensor core layout has all values in the same
        // thread along the same n) but different k dimension, but all are
        // guaranteed to occur within the same quantization group, so we need
        // only load a single scale + zero to cover what this lane has
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          // FIXME: verify that qScale/qZero are in regs not lmem
          v.vals[k] = FloatDefs<FT>::fma2(
              v.vals[k], qScale[curQGroup], qZero[curQGroup]);
        }

        // type pun, the typename FloatDefs<FT>::T2 value in T2x4 is a struct
        // and can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(out[0].vals[0]) == sizeof(v.vals[0]));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          std::memcpy(
              &out[i * 2 + (k / 2)].vals[k % 2], &v.vals[k], sizeof(v.vals[0]));
        }
      }
    } else if constexpr (QType == Int4_QType::MX4_Grouped) {
      // Convert a signed MX4 exponent (in the range [-127, 127] + NaN) to
      // a float value of our given float type
      typename FloatDefs<FT>::T2 fRowExp[LoadT<KTilesToLoad>::kNumQGroups];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        typename FloatDefs<FT>::T v;
        convert_mx4_exponent_to_f16(in.mx4Exponent[i], v);
        fRowExp[i] = FloatDefs<FT>::TToT2(v);
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad / 2; ++i) {
        typename FloatDefs<FT>::T2x4 v;

        // 8 x MX4 fp4 -> 8 x bf16/fp16
        convert_any4x8_global_to_f16x2x4<FT>(in.data[i], init.dequantMX4, v);

        auto curKTile = i * 2;
        // q-group sizes are at least kKTileSize * 2, so this is ok
        // (won't be split across two different q-groups)
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // The dequantized values in `v` for a given lane have the same n
        // dimension (the B tensor core layout has all values in the same
        // thread along the same n) but different k dimension. So we only
        // need the single row scaling exponent
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          v.vals[k] = FloatDefs<FT>::mul2(v.vals[k], fRowExp[curQGroup]);
        }

        // type pun, the typename FloatDefs<FT>::T2 value in T2x4 is a struct
        // and can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(out[0].vals[0]) == sizeof(v.vals[0]));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          std::memcpy(
              &out[i * 2 + (k / 2)].vals[k % 2], &v.vals[k], sizeof(v.vals[0]));
        }
      }
    }
  }
};

template <int Warps, FloatType FT, int InnerKTiles, int QGroupSize>
struct BLayout_TC_int8 {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = InnerKTiles;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  static constexpr int32_t kSharedMemory = 0;
  static constexpr bool kSyncAfterInit = false;

  // Dequantization information needed for all loads, if any
  struct InitT {};

  // Raw data type of matrix (before dequantization) and
  // data needed for dequantization
  template <int KTilesToLoad>
  struct LoadT {
    static constexpr int kKTilesPerQGroup = (QGroupSize / kKTileSize);
    // a q-group could be larger than what we are handling in a single warp
    static constexpr int kNumQGroups = (KTilesToLoad / kKTilesPerQGroup) < 1
        ? 1
        : (KTilesToLoad / kKTilesPerQGroup);

    // 1 k-tile per int32 (B TC layout has 4 int8 words or 32 bits per k-tile)
    uint32_t data[KTilesToLoad];

    // dequant info
    typename FloatDefs<FT>::T2 qScaleAndZero[kNumQGroups];
  };

  static __device__ InitT init(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      int32_t mTile,
      int32_t nTile,
      const DequantInfo& dqInfo) {
    return InitT();
  };

  static __device__ const void* getMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_B
    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
    return reinterpret_cast<const int32_t*>(B) +
        (((nTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
        InnerKTiles;
#else
    return B;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* B,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_B
    return reinterpret_cast<const int32_t*>(B) +
        (kTileIncrement / InnerKTiles) * kWarpSize * InnerKTiles;
#else
    return B;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // type uint32, size [n / 8][k / (InnerKTiles * 16)][32][InnerKTiles]
      // n / 8: n-tiles (n8)
      // k / (InnerKTiles * 16): TC size per k-tile is 16 (m16n8k16)
      // 32: value per warp lane
      // InnerKTiles: 4 int8 quantized values (4 per k-tile)
      // 2 k-tiles packed is a uint32x2 (64 bits)
      // 4 k-tiles packed is a uint32x4 (128 bits)
      const void* __restrict__ B,
      // size [k / qGroupSize][n][2]
      // Contains the scale and zero point of each of the quantized int4 values
      // within B
      // v_reconstructed = (bf16/fp16(B_int4_val) * scale) - zero
      const DequantInfo& dqInfo,
      int32_t n,
      int32_t k,
      int32_t nTiles,
      int32_t nTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    static_assert(InnerKTiles == 1 || InnerKTiles == 2 || InnerKTiles == 4);

#ifdef USE_ITER_B
    auto bPtr = reinterpret_cast<const int32_t*>(B);
#else
    auto bPtr = reinterpret_cast<const int32_t*>(B) +
        (((nTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
            InnerKTiles;
#endif

#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
      auto bPtrCur = bPtr + i * kWarpSize * InnerKTiles;

      if constexpr (InnerKTiles == 1) {
        auto v = *reinterpret_cast<const i32x1*>(bPtrCur);
#pragma unroll
        for (int j = 0; j < InnerKTiles; ++j) {
          out.data[i * InnerKTiles + j] = v.vals[j];
        }
      } else if constexpr (InnerKTiles == 2) {
        auto v = *reinterpret_cast<const i32x2*>(bPtrCur);
#pragma unroll
        for (int j = 0; j < InnerKTiles; ++j) {
          out.data[i * InnerKTiles + j] = v.vals[j];
        }
      } else if constexpr (InnerKTiles == 4) {
        auto v = *reinterpret_cast<const i32x4*>(bPtrCur);
#pragma unroll
        for (int j = 0; j < InnerKTiles; ++j) {
          out.data[i * InnerKTiles + j] = v.vals[j];
        }
      }
    }

    // Load needed info for dequantization

    static_assert(isPowerOf2(QGroupSize), "");
    static_assert(isEvenDivisor(QGroupSize, kKTileSize), "");
    // smallest quantization group size is 16 (1 k-tile is packed in an int32)
    static_assert(QGroupSize >= kKTileSize, "");

    {
      int32_t laneN = nTile * kNTileSize + (laneId / 4);
      int32_t groupStart = (kTileStart * kKTileSize) / QGroupSize;

      int32_t n = nTiles * kNTileSize;

      // offset [qScale_kGroup][qScale_n][0]
      auto qInfoPtr =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo1) +
          (groupStart * n + laneN) * 2;

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        out.qScaleAndZero[i] =
            *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                qInfoPtr + i * n * 2);
      }
    }
  }

  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x2 out[KTilesToLoad]) {
    //
    // De-quantize int8 values to bf16/fp16. Values are dequantized as truly
    // int8
    // [-128, 127] range; dequant = (bf16/fp16(int4_value) * scale) + zero
    //
    {
      // FIXME: does this negatively affect register counts, or will nvcc
      // move this expansion (and data loads above) closer to the point of use?
      typename FloatDefs<FT>::T2 qScale[LoadT<KTilesToLoad>::kNumQGroups];
      typename FloatDefs<FT>::T2 qZero[LoadT<KTilesToLoad>::kNumQGroups];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        qScale[i] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i].x);
        qZero[i] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i].y);
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad; ++i) {
        // 4 x int8 -> 4 x bf16/fp16
        typename FloatDefs<FT>::T2x2 v;

        convert_i8x4_to_f16x2x2(in.data[i], v);

        auto curKTile = i;
        // q-group sizes are at least kKTileSize
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // The dequantized values in `v` for a given lane have the same n
        // dimension (the B tensor core layout has all values in the same
        // thread along the same n) but different k dimension, but all are
        // guaranteed to occur within the same quantization group, so we need
        // only load a single scale + zero to cover what this lane has
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          v.vals[k] = FloatDefs<FT>::fma2(
              v.vals[k], qScale[curQGroup], qZero[curQGroup]);
        }

        // type pun, the T2 value in T2x2 is a struct and
        // can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(v) == sizeof(out[0]), "");
        std::memcpy(&out[i], &v, sizeof(u32x2));
      }
    }
  }
};

} // namespace tinygemm
