// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "TinyGemmUtils.cuh"

namespace tinygemm {

inline __device__ void convert_any4x8_to_f16x2x4(
    uint32_t source,
    __nv_bfloat16 dq01_45,
    __nv_bfloat16 dq23_67,
    bf16x2x4& out) {
  // source is interleaved as 75316420
  uint32_t x_even = source; // 6420
  uint32_t x_odd = source >> 16; // 7531

  // Note: Really the lane which we want to access is the 4 LSBs of
  // x_even / x_odd, but it appears that if you pass non-zero LSBs
  // beyond that, the ones beyond 5 are ignored. We can pass 0/1 for
  // the 5th LSB because we duplicate data in the upper half of the
  // warp. So we can omit the masking with 0xfU for lane selection.
  //
  // This is undocumented (?) and possibly not supported by subsequent
  // GPUs / CUDA versions though.
  __nv_bfloat16 v0 = __shfl_sync(0xffffffff, dq01_45, x_even);
  __nv_bfloat16 v1 = __shfl_sync(0xffffffff, dq01_45, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[0] = halves2bf162(v0, v1);

  __nv_bfloat16 v2 = __shfl_sync(0xffffffff, dq23_67, x_even);
  __nv_bfloat16 v3 = __shfl_sync(0xffffffff, dq23_67, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[1] = halves2bf162(v2, v3);

  __nv_bfloat16 v4 = __shfl_sync(0xffffffff, dq01_45, x_even);
  __nv_bfloat16 v5 = __shfl_sync(0xffffffff, dq01_45, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[2] = halves2bf162(v4, v5);

  __nv_bfloat16 v6 = __shfl_sync(0xffffffff, dq23_67, x_even);
  __nv_bfloat16 v7 = __shfl_sync(0xffffffff, dq23_67, x_odd);
  out.vals[3] = halves2bf162(v6, v7);
}

inline __device__ void convert_any4x8_to_f16x2x4(
    uint32_t source,
    half dq01_45,
    half dq23_67,
    f16x2x4& out) {
  // source is interleaved as 75316420
  uint32_t x_even = source; // 6420
  uint32_t x_odd = source >> 16; // 7531

  // Note: Really the lane which we want to access is the 4 LSBs of
  // x_even / x_odd, but it appears that if you pass non-zero LSBs
  // beyond that, the ones beyond 5 are ignored. We can pass 0/1 for
  // the 5th LSB because we duplicate data in the upper half of the
  // warp. So we can omit the masking with 0xfU for lane selection.
  //
  // This is undocumented (?) and possibly not supported by subsequent
  // GPUs / CUDA versions though.
  half v0 = __shfl_sync(0xffffffff, dq01_45, x_even);
  half v1 = __shfl_sync(0xffffffff, dq01_45, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[0] = __halves2half2(v0, v1);

  half v2 = __shfl_sync(0xffffffff, dq23_67, x_even);
  half v3 = __shfl_sync(0xffffffff, dq23_67, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[1] = __halves2half2(v2, v3);

  half v4 = __shfl_sync(0xffffffff, dq01_45, x_even);
  half v5 = __shfl_sync(0xffffffff, dq01_45, x_odd);
  x_even >>= 4;
  x_odd >>= 4;
  out.vals[2] = __halves2half2(v4, v5);

  half v6 = __shfl_sync(0xffffffff, dq23_67, x_even);
  half v7 = __shfl_sync(0xffffffff, dq23_67, x_odd);
  out.vals[3] = __halves2half2(v6, v7);
}

// int4 x 8 -> bf16 x 8 uniform dequantization
// based on
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline __device__ void convert_i4x8_to_f16x2x4(
    uint32_t source,
    bf16x2x4& result) {
  constexpr int kElements = 8;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const source_i4s = source;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  // We don't have enough mantissa to remove as much shift overhead as FP16, so
  // we must loop. No shift needed for first item.
  uint32_t i4s = source_i4s;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
#pragma unroll
  for (int ii = 1; ii < kElements / 2; ++ii) {
    i4s >>= 4;
    // (i4s & 0x000f000f) | 0x43004300
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[ii])
        : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  }

  // This is the BF16 {-136, -136} represented as an integer.
  static constexpr uint32_t BF16_BIAS = 0xC308C308;
  static constexpr uint32_t BF16_ONE = 0x3F803F80;

// Finally, we construct the output numbers.
#pragma unroll
  for (int ii = 0; ii < kElements / 2; ++ii) {
    // Since this section is for Ampere+, we use bf16 fma to do the bias
    // subtraction
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(h[ii])
        : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
  }
}

// int4 x 8 -> fp16 x 8 dequantization
// based on
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline __device__ void convert_i4x8_to_f16x2x4(
    uint32_t source,
    f16x2x4& result) {
  constexpr int kElements = 8;

  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const source_i4s = source;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is
  // thanks to the register packing format and the fact that we force our
  // integers to be unsigned, and account for this in the fp16 subtractions. In
  // addition, I exploit the fact that sub and fma have the same throughput in
  // order to convert elt_23 and elt_67 to fp16 without having to shift them to
  // the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW
  // dependency if we issue immediately before required.
  const uint32_t top_i4s = source_i4s >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(source_i4s),
                 "n"(BOTTOM_MASK),
                 "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(source_i4s),
                 "n"(TOP_MASK),
                 "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s),
                 "n"(BOTTOM_MASK),
                 "n"(I4s_TO_F16s_MAGIC_NUM),
                 "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile(
      "lop3.b32 %0, %1, %2, %3, %4;\n"
      : "=r"(h[3])
      : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  // I use inline PTX below because I am not sure if the compiler will emit
  // float2half instructions if I use the half2 ctor. In this case, I chose
  // performance reliability over code readability.

  // This is the half2 {1032, 1032} represented as an integer.
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-72, -72} represented as an integer.
  static constexpr uint32_t NEG_72 = 0xd480d480;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
}

// int8 x 4 -> bf16 x 4 dequantization
// based on
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline __device__ void convert_i8x4_to_f16x2x2(
    uint32_t source,
    bf16x2x2& result) {
  uint32_t v_u32[4];

  constexpr uint32_t kFP32Base = 0x4b000000;
  v_u32[0] = __byte_perm(source, kFP32Base, 0x7650);
  v_u32[1] = __byte_perm(source, kFP32Base, 0x7652);
  v_u32[2] = __byte_perm(source, kFP32Base, 0x7651);
  v_u32[3] = __byte_perm(source, kFP32Base, 0x7653);

  // Subtract out kFP32Base + 128 to make the unsigned integer signed.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v_f32;
    // type pun
    static_assert(sizeof(v_u32[0]) == sizeof(v_f32), "");
    std::memcpy(&v_f32, &v_u32[i], sizeof(v_f32));

    v_f32 -= 8388736.f;

    std::memcpy(&v_u32[i], &v_f32, sizeof(v_f32));
  }

  // Truncate the fp32 representation and pack up as bfloat16s.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    auto bf16_u32 = __byte_perm(v_u32[2 * i + 0], v_u32[2 * i + 1], 0x7632);

    // type pun
    static_assert(sizeof(result.vals[0]) == sizeof(bf16_u32), "");
    std::memcpy(&result.vals[i], &bf16_u32, sizeof(bf16_u32));
  }
}

// int8 x 4 -> f16 x 4 dequantization
// based on
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
inline __device__ void convert_i8x4_to_f16x2x2(
    uint32_t source,
    f16x2x2& result) {
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  // Lastly, we subtract 1152 from our constructed number using fp16 math to get
  // our signed integer as fp16.
  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}

// MX4 e8m0 exponent -> fp32
inline __device__ float convert_mx4_exponent_to_fp32(uint8_t exp) {
  // MX4 e8m0 exponent is in the range [-127, 127] + NaN
  // e.g., exp = 0 -> -127, exp = 254 -> +127, exp = 255 -> NaN
  // FIXME: is this ok or is bit manipulation faster?
  float exp32 = ldexpf(1.0f, int(exp) - 127);
  exp32 = (exp == 0xffU) ? std::numeric_limits<float>::quiet_NaN() : exp32;

  return exp32;
}

// MX4 e8m0 exponent -> bf16
inline __device__ void convert_mx4_exponent_to_f16(
    uint8_t exp,
    __nv_bfloat16& out) {
  out = __float2bfloat16_rn(convert_mx4_exponent_to_fp32(exp));
}

// MX4 e8m0 exponent -> fp16
inline __device__ void convert_mx4_exponent_to_f16(uint8_t exp, half& out) {
  out = __float2half_rn(convert_mx4_exponent_to_fp32(exp));
}

} // namespace tinygemm
