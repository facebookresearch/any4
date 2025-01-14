// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "TinyGemmUtils.cuh"

namespace tinygemm {

// MX4 fp4 dequantization values in (u)int4 order
__device__ const float kMX4_Values[16] = {
    0.0f,
    0.5f,
    1.0f,
    1.5f,
    2.0f,
    3.0f,
    4.0f,
    6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f};

enum class FloatType {
  BFloat16,
  Float16,
};

template <FloatType ft>
struct FloatDefs;

template <>
struct FloatDefs<FloatType::BFloat16> {
  using T = __nv_bfloat16;
  using T2 = __nv_bfloat162;
  using T4 = bf16x4;
  using T2x2 = bf16x2x2;
  using T2x4 = bf16x2x4;

  static inline __device__ T floatToT(float v) {
    return __float2bfloat16_rn(v);
  }

  static inline __device__ T2 float2ToT2(const float2& v) {
    return float22bf162(v);
  }

  static inline __device__ T2 TToT2(const T& v) {
    return bf162bf162(v);
  }

  static inline __device__ T2 Tx2ToT2(const T& a, const T& b) {
    return halves2bf162(a, b);
  }

  static inline __device__ T2 mul2(const T2 a, const T2 b) {
    __nv_bfloat162 val;
    NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_90,
        asm("{ mul.bf16x2 %0,%1,%2; }\n"
            : "=r"(__BFLOAT162_TO_UI(val))
            : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)));
        ,
        asm("{.reg .b32 c;\n"
            "  mov.b32 c, 0x80008000U;\n"
            "  fma.rn.bf16x2 %0,%1,%2,c;}\n"
            : "=r"(__BFLOAT162_TO_UI(val))
            : "r"(__BFLOAT162_TO_CUI(a)), "r"(__BFLOAT162_TO_CUI(b)));)
    return val;

    // FIXME: why does this not work (thinks it's mapping to the half2 version)?
    // return __hmul2_rn(a, b);
  }

  static inline __device__ T2 fma2(const T2& a, const T2& b, const T2& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return float22bf162(__half22float2(__hfma2(
        __float22half2_rn(bf1622float2(a)),
        __float22half2_rn(bf1622float2(b)),
        __float22half2_rn(bf1622float2(c)))));
#else
    return __hfma2(a, b, c);
#endif
  }

  static inline __device__ void
  mma(float4& out, const u32x4& a, const u32x2& b, const float4& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // FIXME: mma m16n8k16 requires >= 800
    // Maybe reference implementation
    CUDA_KERNEL_ASSERT(false);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
        : "r"(a.vals[0]),
          "r"(a.vals[1]),
          "r"(a.vals[2]),
          "r"(a.vals[3]),
          "r"(b.vals[0]),
          "r"(b.vals[1]),
          "f"(c.x),
          "f"(c.y),
          "f"(c.z),
          "f"(c.w));
#endif
  }
};

template <>
struct FloatDefs<FloatType::Float16> {
  using T = half;
  using T2 = half2;
  using T4 = f16x4;
  using T2x2 = f16x2x2;
  using T2x4 = f16x2x4;

  static inline __device__ T floatToT(float v) {
    return __float2half_rn(v);
  }

  static inline __device__ T2 float2ToT2(const float2& v) {
    return __float22half2_rn(v);
  }

  static inline __device__ T2 TToT2(const T& v) {
    return __half2half2(v);
  }

  static inline __device__ T2 Tx2ToT2(const T& a, const T& b) {
    return __halves2half2(a, b);
  }

  static inline __device__ T2 mul2(const T2 a, const T2 b) {
    return __hmul2_rn(a, b);
  }

  static inline __device__ T2 fma2(const T2& a, const T2& b, const T2& c) {
    return __hfma2(a, b, c);
  }

  static inline __device__ void
  mma(float4& out, const u32x4& a, const u32x2& b, const float4& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // FIXME: mma m16n8k16 requires >= 800
    // Maybe reference implementation
    CUDA_KERNEL_ASSERT(false);
#else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
        : "r"(a.vals[0]),
          "r"(a.vals[1]),
          "r"(a.vals[2]),
          "r"(a.vals[3]),
          "r"(b.vals[0]),
          "r"(b.vals[1]),
          "f"(c.x),
          "f"(c.y),
          "f"(c.z),
          "f"(c.w));
#endif
  }
};

} // namespace tinygemm
