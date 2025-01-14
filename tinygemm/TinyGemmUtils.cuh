// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#pragma once

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinygemm {

enum class Int4_QType {
  Int4_Grouped, // group-wise int4 quantization
  Any4_Grouped, // group-wise/row-wise any4 quantization
  MX4_Grouped, // mx4 (fp4 as s1e2m1) with group-wise exponent e8
};

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __floats2bfloat162_rn(val.x, val.y);
#else
  return __float22bfloat162_rn(val);
#endif
}

inline __device__ __nv_bfloat162
halves2bf162(const __nv_bfloat16 val1, const __nv_bfloat16 val2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 ret_val;
  ret_val.x = val1;
  ret_val.y = val2;
  return ret_val;
#else
  return __halves2bfloat162(val1, val2);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

constexpr int32_t kWarpSize = 32;

// i32 vector types
struct __align__(4) i32x1 {
  int32_t vals[1];
};

struct __align__(8) i32x2 {
  int32_t vals[2];
};

struct __align__(16) i32x4 {
  int32_t vals[4];
};

// u32 vector types
struct __align__(4) u32x1 {
  uint32_t vals[1];
};

struct __align__(8) u32x2 {
  uint32_t vals[2];
};

struct __align__(16) u32x4 {
  uint32_t vals[4];
};

// f16 vector ypes
struct __align__(2) f16x1 {
  __half vals[1];
};

struct __align__(4) f16x2 {
  __half vals[2];
};

struct __align__(8) f16x4 {
  __half vals[4];
};

struct __align__(16) f16x8 {
  __half vals[8];
};

// f16x2 vector types
struct __align__(4) f16x2x1 {
  half2 vals[1];
};

struct __align__(8) f16x2x2 {
  half2 vals[2];
};

struct __align__(16) f16x2x4 {
  half2 vals[4];
};

// bf16 vector types
struct __align__(2) bf16x1 {
  __nv_bfloat16 vals[1];
};

struct __align__(4) bf16x2 {
  __nv_bfloat16 vals[2];
};

struct __align__(8) bf16x4 {
  __nv_bfloat16 vals[4];
};

struct __align__(16) bf16x8 {
  __nv_bfloat16 vals[8];
};

// bf162 vector types
struct __align__(4) bf16x2x1 {
  __nv_bfloat162 vals[1];
};

struct __align__(8) bf16x2x2 {
  __nv_bfloat162 vals[2];
};

struct __align__(16) bf16x2x4 {
  __nv_bfloat162 vals[4];
};

template <typename T, int N>
struct __align__(sizeof(T) * N) VectorType {
  T vals[N];
};

// Dequantization info passed to the kernel and to load/store objects
struct DequantInfo {
  const void* __restrict__ qInfo1;
  const void* __restrict__ qInfo2;
  int32_t iInfo1;
  int32_t iInfo2;

  static inline DequantInfo empty() {
    DequantInfo out;
    out.qInfo1 = nullptr;
    out.qInfo2 = nullptr;
    out.iInfo1 = 0;
    out.iInfo2 = 0;
    return out;
  }
};

} // namespace tinygemm
