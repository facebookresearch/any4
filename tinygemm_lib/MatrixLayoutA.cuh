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

#define USE_ITER_A

//
// A matrix layouts
//

// Loads the A matrix in 16-bit standard m x k row major layout, and writes
// the C matrix in 16-bit standard m x n row major layout:
//
// load size [m][k]
// FIXME: assumes that k is a multiple of kKTileSize
// store size [m][n]
// FIXME: assumes n is a multiple of kKTileSize
template <int Warps, FloatType FT>
struct ALayout_RM {
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
    u32x4 data[KTilesToLoad];
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
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_A
    auto mLane = mTile * kMTileSize + (laneId / 4);
    auto kLane = kTileStart * kKTileSize + (laneId % 4) * 2;
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(A) + mLane * k +
        kLane;
#else
    return A;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_A
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(A) +
        kTileIncrement * kKTileSize;
#else
    return A;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ A,
      const DequantInfo& dqInfo,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    auto mLane = mTile * kMTileSize + (laneId / 4);
    auto kLane = kTileStart * kKTileSize + (laneId % 4) * 2;

#ifdef USE_ITER_A
    auto aPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(A);
#else
    auto aPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(A) +
        mLane * k + kLane;
#endif

    auto aPtrPlus8Rows = aPtr + 8 * k;

    bool m0InBounds = mLane < m;
    bool m1InBounds = (mLane + 8) < m;

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out.data[i].vals[0] = m0InBounds
          ? *reinterpret_cast<const uint32_t*>(aPtr + i * kKTileSize)
          : uint32_t(0);
      out.data[i].vals[1] = m1InBounds
          ? *reinterpret_cast<const uint32_t*>(aPtrPlus8Rows + i * kKTileSize)
          : uint32_t(0);

      out.data[i].vals[2] = m0InBounds
          ? *reinterpret_cast<const uint32_t*>(aPtr + i * kKTileSize + 8)
          : uint32_t(0);
      out.data[i].vals[3] = m1InBounds ? *reinterpret_cast<const uint32_t*>(
                                             aPtrPlus8Rows + i * kKTileSize + 8)
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
      u32x4 out[KTilesToLoad]) {
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
      int32_t m,
      int32_t n,
      int32_t mOutTiles,
      int32_t mTile,
      int32_t nOutTiles,
      int32_t nTile,
      const float4& out) {
    // sum.x / sum.y are written at
    // [laneId / 4], [(laneId % 4) * 2, (laneId % 4) * 2 + 1]
    // sum.z / sum.w are written at
    // [8 + (laneId / 4)], [(laneId % 4) * 2, (laneId % 4) * 2 + 1]
    // i.e., same columns, different row.
    int outRow = mTile * kMTileSize + (laneId / 4);
    int outCol = nTile * kNTileSize + (laneId % 4) * 2;

    // Pointer where sum.x / sum.y is written
    auto cPtr =
        reinterpret_cast<typename FloatDefs<FT>::T*>(C) + outRow * n + outCol;

    auto v01 = FloatDefs<FT>::float2ToT2(float2{out.x, out.y});
    auto v23 = FloatDefs<FT>::float2ToT2(float2{out.z, out.w});

    if (outRow < m) {
      *reinterpret_cast<typename FloatDefs<FT>::T2*>(cPtr) = v01;
    }

    // sum.z, sum.w at +8 rows from cPtr
    if (outRow + 8 < m) {
      *reinterpret_cast<typename FloatDefs<FT>::T2*>(cPtr + 8 * n) = v23;
    }
  }
};

// Loads the A matrix in 16-bit tensor core layout, and writes the C matrix in
// 16-bit tensor core layout (A format):
//
// size [m / 16][k / 16][32][8]
// m / 16: m-tiles (m16)
// k / 16: k-tiles (k16)
// 32: value per warp lane
// 8: 8 bf16/fp16 values per lane for tensor core packed contiguously
//
// KInnerTiles is not supported (always 1) because the A format requires
// 8 scalar values per lane, or 16 bytes for 2-byte words which is the
// largest vector load/store size
template <int Warps, FloatType FT>
struct ALayout_TC {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = 1;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  static constexpr int32_t kSharedMemory = 0;
  static constexpr bool kSyncAfterInit = false;

  // Dequantization information needed for all loads, if any
  struct InitT {};

  // Raw data type of matrix before dequantization (if any)
  template <int KTilesToLoad>
  struct LoadT {
    u32x4 data[KTilesToLoad];
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
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_A
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(A) +
        (((mTile * kTiles) + // mTiles
          kTileStart) *
             32 + // kTiles
         laneId) *
        8;
#else
    return A;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_A
    return reinterpret_cast<const typename FloatDefs<FT>::T*>(A) +
        kTileIncrement * 32 * 8;
#else
    return A;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ A,
      const DequantInfo& dqInfo,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    // access [mTile][kTileStart][laneId][0]

#ifdef USE_ITER_A
    auto aPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(A);
#else
    auto aPtr = reinterpret_cast<const typename FloatDefs<FT>::T*>(A) +
        (((mTile * kTiles) + // mTiles
          kTileStart) *
             32 + // kTiles
         laneId) *
            8;
#endif

#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out.data[i] = *reinterpret_cast<const u32x4*>(aPtr + i * (kWarpSize * 8));
    }
  }

  template <int KTilesToLoad>
  static __device__ void dequant(
      int32_t warpId,
      int32_t laneId,
      void* smem,
      const InitT& init,
      const LoadT<KTilesToLoad>& in,
      u32x4 out[KTilesToLoad]) {
    // should be a no-op
#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out[i] = in.data[i];
    }
  }

  // output: [mOutTiles][ceil(nOutTiles / 2)][32][8]
  // KInnerTiles for output is always 1 (16 byte load)
  static __device__ void store(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      void* __restrict__ C,
      int32_t m,
      int32_t n,
      int32_t mOutTiles,
      int32_t mTile,
      int32_t nOutTiles,
      int32_t nTile,
      const float4& out) {
    typename FloatDefs<FT>::T2x2 out_ft;
    out_ft.vals[0] = FloatDefs<FT>::float2ToT2(float2{out.x, out.y});
    out_ft.vals[1] = FloatDefs<FT>::float2ToT2(float2{out.z, out.w});

    // C is presented as m16n8
    // In A format:
    // the `m` dimension of the C output is the `m` dimension of the A output
    // the `n` dimension of the C output is the `k` dimension of the A output
    // so the n-tiles become k-tiles. However `out` is m16n8 not m16k16,
    // thus two n-tiles are needed per k-tile. InnerKTiles is always 1
    auto kTiles = divUp(nOutTiles, 2);
    auto outerKTile = nTile / 2;
    auto innerKTile = nTile % 2;

    auto pC = reinterpret_cast<typename FloatDefs<FT>::T*>(C) +
        // dim0 index
        ((mTile * kTiles +
          // dim1 index
          outerKTile) *
             kWarpSize +
         // dim2 index
         laneId) *
            8 +
        // dim3 index
        innerKTile * 4;

    *reinterpret_cast<typename FloatDefs<FT>::T2x2*>(pC) = out_ft;
  }
};

template <
    int Warps,
    FloatType FT,
    int InnerKTiles,
    int QGroupSize,
    Int4_QType QType>
struct ALayout_TC_int4 {
  static constexpr FloatType kFT = FT;
  static constexpr int32_t kInnerKTiles = InnerKTiles;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;
  // 16 x 16 mma tile => need 16 rows of any4 data
  static constexpr int32_t kSharedMemory =
      QType == Int4_QType::Any4_RowWise_Grouped
      ? sizeof(typename FloatDefs<FT>::T) * 16 * 16
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
    typename FloatDefs<FT>::T any4LUT = 0;

    // If QType == MX4, then across the warp this holds the
    // 16 float dequantized values for the fp4 MX4 values.
    // It seems faster to perform a lookup via register shuffle
    // than to do the arithmetic to convert MX4 fp4 values to f16.
    // The dequant values are the same for all matrix entries.
    // The 16 values are replicated across both half warps.
    //
    // FIXME: nvcc should remove these if they are not used for the given QType
    typename FloatDefs<FT>::T dequantMX4 = 0;
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

    uint32_t data[KTilesToLoad];

    // Group quantization scale/offset data for
    // Int4_Grouped / Any4_Global_Grouped / Any4_RowWise_Grouped
    //
    // Unlike the B layout where each lane only has values at row m,
    // for A layout each lane has values at row m and at row m + 8, hence the 2
    //
    // FIXME: nvcc should remove this if they are not used for the given QType
    typename FloatDefs<FT>::T2 qScaleAndZero[kNumQGroups][2];

    // Row-wise exponent for MX4_QType
    //
    // Unlike the B layout where each lane only has values at row m,
    // for A layout each lane has values at row m and at row m + 8, hence the 2
    // This is loaded as uint8_t and dequantized later into a float multiplier
    //
    // FIXME: nvcc should remove this if they are not used for the given QType
    uint8_t mx4Exponent[kNumQGroups][2];
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
      // We need to load 16 rows of data starting at this offset
      int32_t mStart = mTile * kMTileSize;

      auto any4LUT =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo2) +
          mStart * 16; // dqInfo.iInfo1;

      auto tid = threadIdx.y * kWarpSize + threadIdx.x;
      auto smemT = reinterpret_cast<typename FloatDefs<FT>::T*>(smem);

      // FIXME: might be better to have an
      // 16 x 256 -> (b)f16x2 LUT with dequantization
      // codes arranged as 76543210, can dequantize 2 values per smem lookup
      if (tid < 16 * 16) {
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
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_A
    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
    return reinterpret_cast<const uint32_t*>(A) +
        (((mTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
        InnerKTiles;
#else
    return A;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_A
    return reinterpret_cast<const uint32_t*>(A) +
        (kTileIncrement / InnerKTiles) * kWarpSize * InnerKTiles;
#else
    return A;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // type uint32, size [m / 16][k / (InnerKTiles * 16)][32][InnerKTiles]
      // m / 16: m-tiles (m16)
      // k / (InnerKTiles * 16): TC size per k-tile is 16 (m16n8k16)
      // 32: value per warp lane
      // InnerKTiles: A layout has 8 values per lane (32 bits) per k-tile.
      // 1 k-tile packed is a uint32x1
      // (4 bits x 8 B-loads for m16n8k16 = 32 bits)
      // 2 k-tiles packed is a uint32x2 (64 bits)
      // 4 k-tiles packed is a uint32x4 (128 bits)
      const void* __restrict__ A,
      // size [k / qGroupSize][m][2]
      // Contains the scale and zero point of each of the quantized int4 values
      // within A
      // v_reconstructed = (bf16/fp16(B_int4_val) * scale) - zero
      const DequantInfo& dqInfo,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    static_assert(InnerKTiles == 1 || InnerKTiles == 2 || InnerKTiles == 4, "");

    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
#ifdef USE_ITER_A
    auto aPtr = reinterpret_cast<const uint32_t*>(A);
#else
    auto aPtr = reinterpret_cast<const uint32_t*>(A) +
        (((mTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
            InnerKTiles;
#endif

    //
    // Load k-tiles of quantized data from the B matrix
    //

    // 1 k-tile per int32 (A TC layout has 8 int4 words or 32 bits per k-tile)
    static_assert(isEvenDivisor(KTilesToLoad, InnerKTiles), "");

#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
      auto aPtrCur = aPtr + i * kWarpSize * InnerKTiles;

      if constexpr (InnerKTiles == 1) {
        //         auto v = *reinterpret_cast<const u32x1*>(aPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles; ++j) {
        //           out.data[i * InnerKTiles + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.u32 {%0}, [%1];"
                     : "=r"(out.data[i * InnerKTiles + 0])
                     : "l"(aPtrCur));

      } else if constexpr (InnerKTiles == 2) {
        //         auto v = *reinterpret_cast<const u32x2*>(aPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles; ++j) {
        //           out.data[i * InnerKTiles + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
                     : "=r"(out.data[i * InnerKTiles + 0]),
                       "=r"(out.data[i * InnerKTiles + 1])
                     : "l"(aPtrCur));

      } else if constexpr (InnerKTiles == 4) {
        //         auto v = *reinterpret_cast<const u32x4*>(aPtrCur);
        // #pragma unroll
        //         for (int j = 0; j < InnerKTiles; ++j) {
        //           out.data[i * InnerKTiles + j] = v.vals[j];
        //         }

        asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(out.data[i * InnerKTiles + 0]),
                       "=r"(out.data[i * InnerKTiles + 1]),
                       "=r"(out.data[i * InnerKTiles + 2]),
                       "=r"(out.data[i * InnerKTiles + 3])
                     : "l"(aPtrCur));
      }
    }

    //
    // Load needed info for dequantization
    //

    // each lane handles values at row laneM0 and at laneM0 + 8 (kMTileSize / 2)
    int32_t laneM0 = mTile * kMTileSize + (laneId / 4);

    if constexpr (
        QType == Int4_QType::Int4_Grouped ||
        QType == Int4_QType::Any4_Global_Grouped ||
        QType == Int4_QType::Any4_RowWise_Grouped ||
        QType == Int4_QType::MX4_Grouped) {
      static_assert(isPowerOf2(QGroupSize), "");
      static_assert(isEvenDivisor(QGroupSize, kKTileSize), "");
      // smallest quantization group size is 16 (1 k-tile is packed in an int32)
      static_assert(QGroupSize >= kKTileSize, "");

      int32_t groupStart = (kTileStart * kKTileSize) / QGroupSize;

      if constexpr (QType != Int4_QType::MX4_Grouped) {
        // int4/any4 have float scale + offset
        // offset [qScale_kGroup][qScale_n][0]

        // info for m0
        auto qInfoPtr0 =
            reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo1) +
            (groupStart * m + laneM0) * 2;
        // info for m0 + 8
        auto qInfoPtr1 = qInfoPtr0 + (kMTileSize / 2) * 2;

#pragma unroll
        for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
          out.qScaleAndZero[i][0] =
              *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                  qInfoPtr0 + i * m * 2);
          out.qScaleAndZero[i][1] =
              *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                  qInfoPtr1 + i * m * 2);
        }

      } else if constexpr (QType == Int4_QType::MX4_Grouped) {
        // mx4 has a uint8 exponent

        // exp for row m0
        auto expPtr0 = reinterpret_cast<const uint8_t*>(dqInfo.qInfo1) +
            laneM0 * dqInfo.iInfo1 + groupStart;
        // exp for row m0 + 8
        auto expPtr0p8 = expPtr0 + (kMTileSize / 2) * dqInfo.iInfo1;

#pragma unroll
        for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
          // FIXME: transpose exponents?
          out.mx4Exponent[i][0] = expPtr0[i];
          out.mx4Exponent[i][1] = expPtr0p8[i];
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
      u32x4 out[KTilesToLoad]) {
    if constexpr (
        QType == Int4_QType::Int4_Grouped ||
        QType == Int4_QType::Any4_Global_Grouped ||
        QType == Int4_QType::Any4_RowWise_Grouped) {
      //
      // De-quantize int4 values to bf16/fp16. Values are dequantized as truly
      // int4
      // [-8, 7] range; dequant = (bf16/fp16(int4_value) * bf16_scale) +
      // bf16_zero
      //

      // FIXME: does this negatively affect register counts, or will nvcc
      // move this expansion (and data loads above) closer to the point of use?
      // the [2] dimension distinguishes the different m rows
      typename FloatDefs<FT>::T2 qScale[LoadT<KTilesToLoad>::kNumQGroups][2];
      typename FloatDefs<FT>::T2 qZero[LoadT<KTilesToLoad>::kNumQGroups][2];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          qScale[i][j] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i][j].x);
          qZero[i][j] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i][j].y);
        }
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad; ++i) {
        typename FloatDefs<FT>::T2x4 v;

        // 8 x int4 -> 8 x bf16/fp16
        if constexpr (QType == Int4_QType::Any4_Global_Grouped) {
          convert_any4x8_global_to_f16x2x4<FT>(in.data[i], init.any4LUT, v);
        } else if constexpr (QType == Int4_QType::Any4_RowWise_Grouped) {
          convert_any4x8_rowwise_A_to_f16x2x4<FT>(
              laneId,
              in.data[i],
              reinterpret_cast<const typename FloatDefs<FT>::T*>(smem),
              v);
        } else {
          convert_i4x8_to_f16x2x4(in.data[i], v);
        }

        auto curKTile = i;
        // q-group sizes are at least kKTileSize, so this is ok
        // (won't be split across two different q-groups)
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // There are two different m (row) indexes:
        // {v.vals[0], v.vals[2]}
        // {v.vals[1], v.vals[3]}
        // which require different row q-groups. However, all are in
        // the same k-tile.

        v.vals[0] = FloatDefs<FT>::fma2(
            v.vals[0], qScale[curQGroup][0], qZero[curQGroup][0]);
        v.vals[2] = FloatDefs<FT>::fma2(
            v.vals[2], qScale[curQGroup][0], qZero[curQGroup][0]);
        v.vals[1] = FloatDefs<FT>::fma2(
            v.vals[1], qScale[curQGroup][1], qZero[curQGroup][1]);
        v.vals[3] = FloatDefs<FT>::fma2(
            v.vals[3], qScale[curQGroup][1], qZero[curQGroup][1]);

        // type pun, the T2 value in T2x4 is a struct and
        // can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(out[0].vals[0]) == sizeof(v.vals[0]));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          std::memcpy(&out[i].vals[k], &v.vals[k], sizeof(v.vals[0]));
        }
      }
    } else if constexpr (QType == Int4_QType::MX4_Grouped) {
      // Convert a signed MX4 exponent (in the range [-127, 127] + NaN) to
      // a float value of our given float type
      typename FloatDefs<FT>::T2 fRowExp0[LoadT<KTilesToLoad>::kNumQGroups];
      typename FloatDefs<FT>::T2 fRowExp0p8[LoadT<KTilesToLoad>::kNumQGroups];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        {
          typename FloatDefs<FT>::T v;
          convert_mx4_exponent_to_f16(in.mx4Exponent[i][0], v);
          fRowExp0[i] = FloatDefs<FT>::TToT2(v);
        }

        {
          typename FloatDefs<FT>::T v;
          convert_mx4_exponent_to_f16(in.mx4Exponent[i][1], v);
          fRowExp0p8[i] = FloatDefs<FT>::TToT2(v);
        }
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad; ++i) {
        typename FloatDefs<FT>::T2x4 v;

        // 8 x MX4 fp4 -> 8 x bf16/fp16
        convert_any4x8_global_to_f16x2x4<FT>(in.data[i], init.dequantMX4, v);

        auto curKTile = i;
        // q-group sizes are at least kKTileSize, so this is ok
        // (won't be split across two different q-groups)
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // There are two different m (row) indexes:
        // {v.vals[0], v.vals[2]}
        // {v.vals[1], v.vals[3]}
        // which require different MX4 scaling exponents
        v.vals[0] = FloatDefs<FT>::mul2(v.vals[0], fRowExp0[curQGroup]);
        v.vals[2] = FloatDefs<FT>::mul2(v.vals[2], fRowExp0[curQGroup]);
        v.vals[1] = FloatDefs<FT>::mul2(v.vals[1], fRowExp0p8[curQGroup]);
        v.vals[3] = FloatDefs<FT>::mul2(v.vals[3], fRowExp0p8[curQGroup]);

        // type pun, the T2 value in T2x4 is a struct and
        // can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(out[0].vals[0]) == sizeof(v.vals[0]));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          std::memcpy(&out[i].vals[k], &v.vals[k], sizeof(v.vals[0]));
        }
      }
    }
  }
};

template <int Warps, FloatType FT, int InnerKTiles, int QGroupSize>
struct ALayout_TC_int8 {
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

    // 1 k-tile per 2 x int32 (B TC layout has 4 int8 words or
    // 32 bits per k-tile)
    u32x2 data[KTilesToLoad];

    // dequant info
    typename FloatDefs<FT>::T2 qScaleAndZero[kNumQGroups][2];
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
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart) {
#ifdef USE_ITER_A
    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
    return reinterpret_cast<const uint32_t*>(A) +
        (((mTile * divUp(kTiles, InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
        InnerKTiles * 2;
#else
    return A;
#endif
  }

  static __device__ const void* incrementMatrixTile(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      const void* __restrict__ A,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileIncrement) {
#ifdef USE_ITER_A
    return reinterpret_cast<const uint32_t*>(A) +
        (kTileIncrement / InnerKTiles) * kWarpSize * InnerKTiles * 2;
#else
    return A;
#endif
  }

  template <int KTilesToLoad>
  static __device__ void load(
      int32_t warpId,
      int32_t laneId,
      void* __restrict__ smem,
      // type uint32, size [ceil(m / 16)][ceil(k / (InnerKTiles *
      // 16))][32][InnerKTiles * 2] m / 16: m-tiles (m16) k / (InnerKTiles *
      // 16): TC size per k-tile is 16 (m16n8k16) 32: value per warp lane
      // InnerKTiles * 2: 8 int8 quantized values (2 x uint32) per k-tile
      // 1 k-tile packed is a uint32x2 (64 bits)
      // 2 k-tiles packed is a uint32x4 (128 bits)
      const void* __restrict__ A,
      // size [k / qGroupSize][m][2]
      // Contains the scale and zero point of each of the quantized int4 values
      // within B
      // v_reconstructed = (bf16/fp16(B_int4_val) * scale) - zero
      const DequantInfo& dqInfo,
      int32_t m,
      int32_t k,
      int32_t mTiles,
      int32_t mTile,
      int32_t kTiles,
      int32_t kTileStart,
      LoadT<KTilesToLoad>& out) {
    static_assert(InnerKTiles == 1 || InnerKTiles == 2);

    // offset [nTile][kTileStart / InnerKTiles][laneId][0]
#ifdef USE_ITER_A
    auto aPtr = reinterpret_cast<const uint32_t*>(A);
#else
    auto aPtr = reinterpret_cast<const uint32_t*>(A) +
        (((mTile * divUp(kTiles, InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
            InnerKTiles * 2;
#endif

#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
      auto aPtrCur = aPtr + i * kWarpSize * InnerKTiles * 2;

      if constexpr (InnerKTiles == 1) {
        out.data[i] = *reinterpret_cast<const u32x2*>(aPtrCur);
      } else if constexpr (InnerKTiles == 2) {
        auto v = *reinterpret_cast<const i32x4*>(aPtrCur);
        out.data[i * InnerKTiles + 0].vals[0] = v.vals[0];
        out.data[i * InnerKTiles + 0].vals[1] = v.vals[1];
        out.data[i * InnerKTiles + 1].vals[0] = v.vals[2];
        out.data[i * InnerKTiles + 1].vals[1] = v.vals[3];
      }
    }

    // Load needed info for dequantization

    static_assert(isPowerOf2(QGroupSize), "");
    static_assert(isEvenDivisor(QGroupSize, kKTileSize), "");
    // smallest quantization group size is 16 (1 k-tile is packed in an int32)
    static_assert(QGroupSize >= kKTileSize, "");

    // For the A layout, each lane has values at rows
    // (laneId / 4) and (laneId / 4) + 8
    {
      int32_t laneM0 = mTile * kMTileSize + (laneId / 4);
      int32_t laneM1 = laneM0 + 8;
      int32_t groupStart = (kTileStart * kKTileSize) / QGroupSize;

      int32_t m = mTiles * kMTileSize;

      // offset [qScale_kGroup][qScale_m][0]
      auto qInfoPtr0 =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo1) +
          (groupStart * m + laneM0) * 2;
      auto qInfoPtr1 =
          reinterpret_cast<const typename FloatDefs<FT>::T*>(dqInfo.qInfo1) +
          (groupStart * m + laneM1) * 2;

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
        out.qScaleAndZero[i][0] =
            *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                qInfoPtr0 + i * m * 2);
        out.qScaleAndZero[i][1] =
            *reinterpret_cast<const typename FloatDefs<FT>::T2*>(
                qInfoPtr1 + i * m * 2);
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
      u32x4 out[KTilesToLoad]) {
    //
    // De-quantize int8 values to bf16/fp16. Values are dequantized as truly
    // int8
    // [-128, 127] range; dequant = (bf16/fp16(int4_value) * scale) + zero
    //
    {
      // FIXME: does this negatively affect register counts, or will nvcc
      // move this expansion (and data loads above) closer to the point of use?
      typename FloatDefs<FT>::T2 qScale[LoadT<KTilesToLoad>::kNumQGroups][2];
      typename FloatDefs<FT>::T2 qZero[LoadT<KTilesToLoad>::kNumQGroups][2];

#pragma unroll
      for (int i = 0; i < LoadT<KTilesToLoad>::kNumQGroups; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          qScale[i][j] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i][j].x);
          qZero[i][j] = FloatDefs<FT>::TToT2(in.qScaleAndZero[i][j].y);
        }
      }

#pragma unroll
      for (int i = 0; i < KTilesToLoad; ++i) {
        // 8 x int8 -> 8 x bf16/fp16
        typename FloatDefs<FT>::T2x2 v0;
        typename FloatDefs<FT>::T2x2 v1;

        convert_i8x4_to_f16x2x2(in.data[i].vals[0], v0);
        convert_i8x4_to_f16x2x2(in.data[i].vals[1], v1);

        typename FloatDefs<FT>::T2x4 v;
        v.vals[0] = v0.vals[0];
        v.vals[1] = v0.vals[1];
        v.vals[2] = v1.vals[0];
        v.vals[3] = v1.vals[1];

        auto curKTile = i;
        // q-group sizes are at least kKTileSize
        auto curQGroup = (curKTile * kKTileSize) / QGroupSize;

        // The dequantized values in `v` for a given lane have the same n
        // dimension (the B tensor core layout has all values in the same
        // thread along the same n) but different k dimension, but all are
        // guaranteed to occur within the same quantization group, so we need
        // only load a single scale + zero to cover what this lane has
        // FIXME: v0 contains:
        //   (a0, a1) at row (laneId / 4), (a2, a3) at row (laneId / 4) + 8
        // FIXME: v1 contains:
        //   (a4, a5) at row (laneId / 4), (a6, a7) at row (laneId / 4) + 8

        v.vals[0] = FloatDefs<FT>::fma2(
            v.vals[0], qScale[curQGroup][0], qZero[curQGroup][0]);
        v.vals[1] = FloatDefs<FT>::fma2(
            v.vals[1], qScale[curQGroup][1], qZero[curQGroup][1]);
        v.vals[2] = FloatDefs<FT>::fma2(
            v.vals[2], qScale[curQGroup][0], qZero[curQGroup][0]);
        v.vals[3] = FloatDefs<FT>::fma2(
            v.vals[3], qScale[curQGroup][1], qZero[curQGroup][1]);

        // type pun, the T2 value in T2x4 is a struct and
        // can't be used as a 32-bit asm register argument for `mma`
        static_assert(sizeof(v) == sizeof(out[0]), "");
        std::memcpy(&out[i], &v, sizeof(u32x4));
      }
    }
  }
};

} // namespace tinygemm
