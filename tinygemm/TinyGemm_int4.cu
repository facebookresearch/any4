// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

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
#include <torch/types.h>

//
// Handles all int4 word quantization types
// (int4 grouped quantization,
//  any4 grouped quantization,
//  MX4 / e8m0 row-wise scale)
//
namespace tinygemm {

template <FloatType FT, Int4_QType QType>
torch::Tensor tinygemm_y_FT16TC_x_FT16TC_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    std::optional<int64_t> QGroup_Size,
    std::optional<torch::Tensor> QGroup_scaleAndZeros,
    std::optional<torch::Tensor> Any4_dequant,
    std::optional<torch::Tensor> MX4_exponents,
    bool weightOnRight) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(A.device() == B.device());

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
    B_innerKTiles = B.size(3) * 2;
  } else {
    TORCH_CHECK(A.size(3) == 1 || A.size(3) == 2 || A.size(3) == 4); // weights
    TORCH_CHECK(B.size(3) == 4 || B.size(3) == 8); // activations
    TORCH_CHECK(A.dtype() == torch::kInt32);

    A_innerKTiles = A.size(3);
    B_innerKTiles = B.size(3) / 4;
  }

  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();

  // tensor core layout
  auto mTiles = A.size(0);
  auto m = mTiles * kMTileSize;

  auto nTiles = B.size(0);
  auto n = nTiles * kNTileSize;

  auto W_rows = weightOnRight ? n : m;

  // validate k dimension equality
  auto kTileA = A.size(1) * A_innerKTiles;
  auto kTileB = B.size(1) * B_innerKTiles;
  TORCH_CHECK(kTileA == kTileB);
  auto kTiles = kTileA;

  auto k = kTiles * kKTileSize;

  // Validate quantization group information, if used
  int64_t qGroupSize = 32; // default if we don't use q-groups
  int64_t numQGroups = -1;

  if constexpr (
      QType == Int4_QType::Int4_Grouped || QType == Int4_QType::Any4_Grouped) {
    TORCH_CHECK(QGroup_Size);
    qGroupSize = *QGroup_Size;
    TORCH_CHECK(
        qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
        qGroupSize == 256);
    TORCH_CHECK(isEvenDivisor(k, qGroupSize));
    numQGroups = k / qGroupSize;

    TORCH_CHECK(QGroup_scaleAndZeros);
    TORCH_CHECK(QGroup_scaleAndZeros->device() == A.device());
    TORCH_CHECK(QGroup_scaleAndZeros->dim() == 3);

    TORCH_CHECK(QGroup_scaleAndZeros->size(0) == numQGroups);
    // Right now, the dequantization code assumes full n (or m) tiles for the
    // weights, so QGroup_scaleAndZeros must be sized as such (even if not all
    // rows are valid)
    TORCH_CHECK(QGroup_scaleAndZeros->size(1) == W_rows);
    TORCH_CHECK(QGroup_scaleAndZeros->size(2) == 2);
  } else if constexpr (QType == Int4_QType::MX4_Grouped) {
    TORCH_CHECK(QGroup_Size);
    qGroupSize = *QGroup_Size;
    TORCH_CHECK(
        qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
        qGroupSize == 256);
    TORCH_CHECK(isEvenDivisor(k, qGroupSize));
    numQGroups = k / qGroupSize;

    TORCH_CHECK(MX4_exponents);
    TORCH_CHECK(MX4_exponents->device() == A.device());
    TORCH_CHECK(MX4_exponents->dtype() == torch::kUInt8);
    TORCH_CHECK(MX4_exponents->dim() == 2);
    TORCH_CHECK(MX4_exponents->size(0) == W_rows);
    TORCH_CHECK(MX4_exponents->size(1) == numQGroups);
  }

  // Set up kernel quantization metadata
  auto dqInfo = DequantInfo::empty();

  if constexpr (QType == Int4_QType::Int4_Grouped) {
    dqInfo.qInfo1 = QGroup_scaleAndZeros->data_ptr();
  } else if constexpr (QType == Int4_QType::Any4_Grouped) {
    TORCH_CHECK(Any4_dequant);
    TORCH_CHECK(Any4_dequant->device() == A.device());
    TORCH_CHECK(Any4_dequant->dtype() == activationDtype);

    // 1-d is matrix-wise any4 quantization
    // 2-d is row-wise any4 quantization (must match rows of weight matrix
    // %in A or B TC format, which is a multiple of 16 or 8 respectively)
    TORCH_CHECK(
        (Any4_dequant->dim() == 1 && Any4_dequant->size(0) == 16) ||
        (Any4_dequant->dim() == 2 && Any4_dequant->size(0) == W_rows));

    dqInfo.qInfo1 = QGroup_scaleAndZeros->data_ptr();
    dqInfo.qInfo2 = Any4_dequant->data_ptr();
    // row stride for any4 (whether matrix-wise or row-wise)
    dqInfo.iInfo1 = Any4_dequant->dim() == 1
        ? 0 /* same LUT for all rows */
        : 16 /* different LUT for each row */;
  } else if constexpr (QType == Int4_QType::MX4_Grouped) {
    dqInfo.qInfo1 = MX4_exponents->data_ptr();
    // row stride for MX4 group exponents (numQGroups)
    dqInfo.iInfo1 = MX4_exponents->size(1);
  }

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

#define B_LAYOUT_INT4_QGROUP(B_INNER_K_TILES)                                  \
  switch (qGroupSize) {                                                        \
    case 32: {                                                                 \
      using BLayout = BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 32, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 64: {                                                                 \
      using BLayout = BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 64, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 128: {                                                                \
      using BLayout =                                                          \
          BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 128, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 256: {                                                                \
      using BLayout =                                                          \
          BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 256, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
  }

#define A_LAYOUT_INT4_QGROUP(A_INNER_K_TILES)                                  \
  switch (qGroupSize) {                                                        \
    case 32: {                                                                 \
      using ALayout = ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 32, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 64: {                                                                 \
      using ALayout = ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 64, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 128: {                                                                \
      using ALayout =                                                          \
          ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 128, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 256: {                                                                \
      using ALayout =                                                          \
          ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 256, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
  }

  if (weightOnRight) {
    using ALayout = ALayout_TC<kWarps, FT>;
    using CLayout = ALayout;
    if (B_innerKTiles == 2) {
      B_LAYOUT_INT4_QGROUP(2);
    } else if (B_innerKTiles == 4) {
      B_LAYOUT_INT4_QGROUP(4);
    } else if (B_innerKTiles == 8) {
      B_LAYOUT_INT4_QGROUP(8);
    }
  } else {
    if (B_innerKTiles == 1) {
      using BLayout = BLayout_TC<kWarps, FT, 1>;
      using CLayout = BLayout;

      if (A_innerKTiles == 1) {
        A_LAYOUT_INT4_QGROUP(1);
      } else if (A_innerKTiles == 2) {
        A_LAYOUT_INT4_QGROUP(2);
      } else if (A_innerKTiles == 4) {
        A_LAYOUT_INT4_QGROUP(4);
      }
    } else if (B_innerKTiles == 2) {
      using BLayout = BLayout_TC<kWarps, FT, 2>;
      using CLayout = BLayout;

      if (A_innerKTiles == 1) {
        A_LAYOUT_INT4_QGROUP(1);
      } else if (A_innerKTiles == 2) {
        A_LAYOUT_INT4_QGROUP(2);
      } else if (A_innerKTiles == 4) {
        A_LAYOUT_INT4_QGROUP(4);
      }
    }
  }

#undef A_LAYOUT_INT4_QGROUP
#undef B_LAYOUT_INT4_QGROUP
#undef RUN_GEMM

  return C_final;
}

template <FloatType FT, Int4_QType QType>
torch::Tensor tinygemm_y_FT16RM_x_FT16RM_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    std::optional<int64_t> QGroup_Size,
    std::optional<torch::Tensor> QGroup_scaleAndZeros,
    std::optional<torch::Tensor> Any4_dequant,
    std::optional<torch::Tensor> MX4_exponents,
    bool weightOnRight) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  c10::cuda::CUDAGuard g(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(A.device() == B.device());

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
    B_innerKTiles = B.size(3) * 2;

    // validation
    TORCH_CHECK(B.size(1) == divUp(kTiles, B_innerKTiles));
    TORCH_CHECK(B.size(2) == kWarpSize);
    TORCH_CHECK(B_innerKTiles == 2 || B_innerKTiles == 4 || B_innerKTiles == 8);
  } else {
    TORCH_CHECK(
        A.dim() == 4 && A.dtype() == torch::kInt32 && A.is_contiguous());
    TORCH_CHECK(B.dim() == 2 && B.is_contiguous());

    // A: tensor core layout
    mTiles = A.size(0);
    m = mTiles * kMTileSize;
    A_innerKTiles = A.size(3);

    // B: row major layout
    n = B.size(0);
    nTiles = divUp(n, kNTileSize);
    k = B.size(1);
    kTiles = divUp(k, kKTileSize);
    B_innerKTiles = 0; // unused

    // validation
    TORCH_CHECK(A.size(1) == divUp(kTiles, A_innerKTiles));
    TORCH_CHECK(A.size(2) == kWarpSize);
    TORCH_CHECK(A_innerKTiles == 1 || A_innerKTiles == 2 || A_innerKTiles == 4);
  }

  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();
  auto W_rows = weightOnRight ? n : m;

  // Validate quantization group information, if used
  int64_t qGroupSize = 32; // default if we don't use q-groups
  int64_t numQGroups = -1;

  if constexpr (
      QType == Int4_QType::Int4_Grouped || QType == Int4_QType::Any4_Grouped) {
    TORCH_CHECK(QGroup_Size);
    qGroupSize = *QGroup_Size;
    TORCH_CHECK(
        qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
        qGroupSize == 256);

    TORCH_CHECK(QGroup_scaleAndZeros);
    TORCH_CHECK(QGroup_scaleAndZeros->device() == A.device());
    TORCH_CHECK(QGroup_scaleAndZeros->dim() == 3);
    numQGroups = QGroup_scaleAndZeros->size(0);
    TORCH_CHECK(isEvenDivisor(k, numQGroups));

    // Right now, the dequantization code assumes full n (or m) tiles for the
    // weights, so QGroup_scaleAndZeros must be sized as such (even if not all
    // rows are valid)
    TORCH_CHECK(QGroup_scaleAndZeros->size(1) == W_rows);
    TORCH_CHECK(QGroup_scaleAndZeros->size(2) == 2);
  } else if constexpr (QType == Int4_QType::MX4_Grouped) {
    TORCH_CHECK(QGroup_Size);
    qGroupSize = *QGroup_Size;
    TORCH_CHECK(
        qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
        qGroupSize == 256);
    TORCH_CHECK(isEvenDivisor(k, qGroupSize));
    numQGroups = k / qGroupSize;

    TORCH_CHECK(MX4_exponents);
    TORCH_CHECK(MX4_exponents->device() == A.device());
    TORCH_CHECK(MX4_exponents->dtype() == torch::kUInt8);
    TORCH_CHECK(MX4_exponents->dim() == 2);
    TORCH_CHECK(MX4_exponents->size(0) == W_rows);
    TORCH_CHECK(MX4_exponents->size(1) == numQGroups);
  }

  // Set up kernel quantization metadata
  auto dqInfo = DequantInfo::empty();

  if constexpr (QType == Int4_QType::Int4_Grouped) {
    dqInfo.qInfo1 = QGroup_scaleAndZeros->data_ptr();

  } else if constexpr (QType == Int4_QType::Any4_Grouped) {
    TORCH_CHECK(Any4_dequant);
    TORCH_CHECK(Any4_dequant->device() == A.device());
    TORCH_CHECK(Any4_dequant->dtype() == activationDtype);

    // 1-d is matrix-wise any4 quantization
    // 2-d is row-wise any4 quantization (must match rows of weight matrix
    // %in A or B TC format, which is a multiple of 16 or 8 respectively)
    TORCH_CHECK(
        (Any4_dequant->dim() == 1 && Any4_dequant->size(0) == 16) ||
        (Any4_dequant->dim() == 2 && Any4_dequant->size(0) == W_rows));

    dqInfo.qInfo1 = QGroup_scaleAndZeros->data_ptr();
    dqInfo.qInfo2 = Any4_dequant->data_ptr();
    // row stride for any4 (whether matrix-wise or row-wise)
    dqInfo.iInfo1 = Any4_dequant->dim() == 1
        ? 0 /* same LUT for all rows */
        : 16 /* different LUT for each row */;
  } else if constexpr (QType == Int4_QType::MX4_Grouped) {
    dqInfo.qInfo1 = MX4_exponents->data_ptr();
    // row stride for MX4 group exponents (numQGroups)
    dqInfo.iInfo1 = MX4_exponents->size(1);
  }

  // Output is a standard row-major matrix
  torch::Tensor C_final;

  if (weightOnRight) {
    C_final = torch::empty(
        {m, n}, torch::TensorOptions().dtype(A.dtype()).device(A.device()));
  } else {
    C_final = torch::empty(
        {n, m}, torch::TensorOptions().dtype(B.dtype()).device(B.device()));
  }

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

#define B_LAYOUT_INT4_QGROUP(B_INNER_K_TILES)                                  \
  switch (qGroupSize) {                                                        \
    case 32: {                                                                 \
      using BLayout = BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 32, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 64: {                                                                 \
      using BLayout = BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 64, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 128: {                                                                \
      using BLayout =                                                          \
          BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 128, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 256: {                                                                \
      using BLayout =                                                          \
          BLayout_TC_int4<kWarps, FT, B_INNER_K_TILES, 256, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
  }

#define A_LAYOUT_INT4_QGROUP(A_INNER_K_TILES)                                  \
  switch (qGroupSize) {                                                        \
    case 32: {                                                                 \
      using ALayout = ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 32, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 64: {                                                                 \
      using ALayout = ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 64, QType>; \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 128: {                                                                \
      using ALayout =                                                          \
          ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 128, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
    case 256: {                                                                \
      using ALayout =                                                          \
          ALayout_TC_int4<kWarps, FT, A_INNER_K_TILES, 256, QType>;            \
      RUN_GEMM(kWarps, kKTilesPerWarp, ALayout, BLayout, CLayout);             \
    } break;                                                                   \
  }

  if (weightOnRight) {
    using ALayout = ALayout_RM<kWarps, FT>;
    using CLayout = ALayout;
    if (B_innerKTiles == 2) {
      B_LAYOUT_INT4_QGROUP(2);
    } else if (B_innerKTiles == 4) {
      B_LAYOUT_INT4_QGROUP(4);
    } else if (B_innerKTiles == 8) {
      B_LAYOUT_INT4_QGROUP(8);
    }
  } else {
    using BLayout = BLayout_RM<kWarps, FT>;
    using CLayout = BLayout;

    if (A_innerKTiles == 1) {
      A_LAYOUT_INT4_QGROUP(1);
    } else if (A_innerKTiles == 2) {
      A_LAYOUT_INT4_QGROUP(2);
    } else if (A_innerKTiles == 4) {
      A_LAYOUT_INT4_QGROUP(4);
    }
  }

#undef A_LAYOUT_INT4_QGROUP
#undef B_LAYOUT_INT4_QGROUP
#undef RUN_GEMM

  return C_final;
}

torch::Tensor tinygemm_y_f16TC_x_f16TC_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qGroupScaleAndZeros,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();
  TORCH_CHECK(
      activationDtype == torch::kBFloat16 ||
      activationDtype == torch::kFloat16);

  if (activationDtype == torch::kBFloat16) {
    return tinygemm_y_FT16TC_x_FT16TC_w_int4TC<
        FloatType::BFloat16,
        Int4_QType::Int4_Grouped>(
        A,
        B,
        qGroupSize,
        qGroupScaleAndZeros,
        std::nullopt,
        std::nullopt,
        weightOnRight);
  } else {
    return tinygemm_y_FT16TC_x_FT16TC_w_int4TC<
        FloatType::Float16,
        Int4_QType::Int4_Grouped>(
        A,
        B,
        qGroupSize,
        qGroupScaleAndZeros,
        std::nullopt,
        std::nullopt,
        weightOnRight);
  }
}

torch::Tensor tinygemm_y_f16RM_x_f16RM_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qGroupScaleAndZeros,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();
  TORCH_CHECK(
      activationDtype == torch::kBFloat16 ||
      activationDtype == torch::kFloat16);

  if (activationDtype == torch::kBFloat16) {
    return tinygemm_y_FT16RM_x_FT16RM_w_int4TC<
        FloatType::BFloat16,
        Int4_QType::Int4_Grouped>(
        A,
        B,
        qGroupSize,
        qGroupScaleAndZeros,
        std::nullopt,
        std::nullopt,
        weightOnRight);
  } else {
    return tinygemm_y_FT16RM_x_FT16RM_w_int4TC<
        FloatType::Float16,
        Int4_QType::Int4_Grouped>(
        A,
        B,
        qGroupSize,
        qGroupScaleAndZeros,
        std::nullopt,
        std::nullopt,
        weightOnRight);
  }
}

torch::Tensor tinygemm_y_f16TC_x_f16TC_w_any4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    torch::Tensor int4Dequant,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();
  TORCH_CHECK(
      activationDtype == torch::kBFloat16 ||
      activationDtype == torch::kFloat16);

  if (activationDtype == torch::kBFloat16) {
    return tinygemm_y_FT16TC_x_FT16TC_w_int4TC<
        FloatType::BFloat16,
        Int4_QType::Any4_Grouped>(
        A,
        B,
        qGroupSize,
        qScaleAndZeros,
        int4Dequant,
        std::nullopt,
        weightOnRight);
  } else {
    return tinygemm_y_FT16TC_x_FT16TC_w_int4TC<
        FloatType::Float16,
        Int4_QType::Any4_Grouped>(
        A,
        B,
        qGroupSize,
        qScaleAndZeros,
        int4Dequant,
        std::nullopt,
        weightOnRight);
  }
}

torch::Tensor tinygemm_y_f16RM_x_f16RM_w_any4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    torch::Tensor int4Dequant,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();
  TORCH_CHECK(
      activationDtype == torch::kBFloat16 ||
      activationDtype == torch::kFloat16);

  if (activationDtype == torch::kBFloat16) {
    return tinygemm_y_FT16RM_x_FT16RM_w_int4TC<
        FloatType::BFloat16,
        Int4_QType::Any4_Grouped>(
        A,
        B,
        qGroupSize,
        qScaleAndZeros,
        int4Dequant,
        std::nullopt,
        weightOnRight);
  } else {
    return tinygemm_y_FT16RM_x_FT16RM_w_int4TC<
        FloatType::Float16,
        Int4_QType::Any4_Grouped>(
        A,
        B,
        qGroupSize,
        qScaleAndZeros,
        int4Dequant,
        std::nullopt,
        weightOnRight);
  }
}

torch::Tensor tinygemm_y_f16TC_x_f16TC_w_mx4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor mx4Exponents,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();

  // Only bfloat16 is supported for mx4, as only it has an exponent range
  // compatible with 8 bit exponent scales
  TORCH_CHECK(activationDtype == torch::kBFloat16);

  return tinygemm_y_FT16TC_x_FT16TC_w_int4TC<
      FloatType::BFloat16,
      Int4_QType::MX4_Grouped>(
      A,
      B,
      qGroupSize,
      std::nullopt,
      std::nullopt,
      mx4Exponents,
      weightOnRight);
}

torch::Tensor tinygemm_y_f16RM_x_f16RM_w_mx4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor mx4Exponents,
    bool weightOnRight) {
  auto activationDtype = weightOnRight ? A.dtype() : B.dtype();

  // Only bfloat16 is supported for mx4, as only it has an exponent range
  // compatible with 8 bit exponent scales
  TORCH_CHECK(activationDtype == torch::kBFloat16);

  return tinygemm_y_FT16RM_x_FT16RM_w_int4TC<
      FloatType::BFloat16,
      Int4_QType::MX4_Grouped>(
      A,
      B,
      qGroupSize,
      std::nullopt,
      std::nullopt,
      mx4Exponents,
      weightOnRight);
}

} // namespace tinygemm
