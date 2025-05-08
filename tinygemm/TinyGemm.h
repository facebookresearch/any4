// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <torch/types.h>

namespace tinygemm {

//
// Conversions to/from "A" tensor core format
//

// [m][k] -> [ceil(m / 16)][ceil(k / 16)][32][8]
// supports float16 and bfloat16
//
// valid innerKTiles: 1
torch::Tensor convert_matrix_to_m16n8k16_A_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

// [ceil(m / 16)][ceil(k / 16)][32][8] -> [m][k]
// supports float16 and bfloat16
torch::Tensor convert_matrix_from_m16n8k16_A_layout(
    const torch::Tensor& in,
    int64_t m,
    int64_t k);

// [m][k] -> [ceil(m / 16)][ceil(k / innerKTiles * 16)][32][innerKTiles]
// input and output are *int32* dtype; output contains packed int4 values
// (thus is 8x smaller than the input)
//
// valid innerKTiles: 1, 2, 4
//
// NOTE:
// input is pre-quantized to *uint4* (range [0, 15]) stored as int32
// output is packed as 8 x *int4* (range [-8, 7]) into a single int32 word
torch::Tensor convert_matrix_to_m16n8k16_Aint4_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

// [m][k] -> [ceil(m / 16)][ceil(k / (innerKTiles * 16))][32][innerKTiles * 2]
// input and output are *int32* dtype; output contains packed int8 values
// (thus is 4x smaller than the input)
//
// valid innerKTiles: 1, 2
//
// NOTE:
// input is pre-quantized to *uint8* (range [0, 255]) stored as int32
// output is packed as 8 x *int8* (range [-128, 127]) into a single int32 word
torch::Tensor convert_matrix_to_m16n8k16_Aint8_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

//
// Conversions to/from "B" tensor core format
//

// [n][k] -> [ceil(n / 8)][ceil(k / innerKTiles * 16)][32][innerKTiles * 4]
// supports float16 and bfloat16
//
// valid innerKTiles: 1, 2
torch::Tensor convert_matrix_to_m16n8k16_B_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

// [ceil(n / 8)][ceil(k / innerKTiles * 16)][32][innerKTiles * 4] -> [n][k]
// supports float16 and bfloat16
torch::Tensor convert_matrix_from_m16n8k16_B_layout(
    const torch::Tensor& in,
    int64_t n,
    int64_t k);

// [n][k] -> [ceil(n / 8)][ceil(k / innerKTiles * 16)][32][innerKTiles / 2]
// input and output are *int32* dtype; output contains packed int4 values
// (thus is 8x smaller than the input)
//
// valid innerKTiles: 2, 4, 8
//
// NOTE:
// input is pre-quantized to *uint4* (range [0, 15]) stored as int32
// output is packed as 8 x *int4* (range [-8, 7]) into a single int32 word
torch::Tensor convert_matrix_to_m16n8k16_Bint4_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

// [n][k] -> [ceil(n / 8)][ceil(k / innerKTiles * 16)][32][innerKTiles]
// input and output are *int32* dtype; output contains packed int8 values
// (thus is 4x smaller than the input)
//
// valid innerKTiles: 1, 2, 4
//
// NOTE:
// input is pre-quantized to *uint8* (range [0, 255]) stored as int32
// output is packed as 8 x *int4* (range [-128, 127]) into a single int32 word
torch::Tensor convert_matrix_to_m16n8k16_Bint8_layout(
    const torch::Tensor& in,
    int64_t innerKTiles);

//
// weights as 4-bit uniform int4 (quantization groups)
//

// tinygemm bf16/fp16 (TC layout) x int4 (q-group, TC layout) = bf16/fp16 (TC
// layout)
torch::Tensor tinygemm_y_f16TC_x_f16TC_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight);

// tinygemm bf16/fp16 (RM layout) x int4 (q-group, TC layout) = bf16/fp16 (RM
// layout)
torch::Tensor tinygemm_y_f16RM_x_f16RM_w_int4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight);

//
// weights as 4-bit any4 floating point (quantization groups + arbitrary
// mapping)
//

// tinygemm bf16/fp16 (TC layout) x int4 (q-group, TC layout) = bf16/fp16 (TC
// layout) with arbitrary dequantization
torch::Tensor tinygemm_y_f16TC_x_f16TC_w_any4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    torch::Tensor int4DequantValues,
    bool weightOnRight);

// tinygemm bf16/fp16 (RM layout) x int4 (q-group, TC layout) = bf16/fp16 (RM
// layout) with arbitrary dequantization
torch::Tensor tinygemm_y_f16RM_x_f16RM_w_any4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    torch::Tensor int4DequantValues,
    bool weightOnRight);

//
// weights as 4-bit MX4 (fp4 + row-wise 8 bit exponent)
//

// tinygemm bf16/fp16 (TC layout) x int4 (q-group, TC layout) = bf16/fp16 (TC
// layout)
torch::Tensor tinygemm_y_f16TC_x_f16TC_w_mx4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor mx4Exponents,
    bool weightOnRight);

// tinygemm bf16/fp16 (RM layout) x int4 (q-group, TC layout) = bf16/fp16 (RM
// layout)
torch::Tensor tinygemm_y_f16RM_x_f16RM_w_mx4TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor mx4Exponents,
    bool weightOnRight);

//
// weights as int8 (quantization groups)
//

// tinygemm bf16/fp16 (TC layout) x int8 (q-group, TC layout) = bf16/fp16 (TC
// layout)
torch::Tensor tinygemm_y_f16TC_x_f16TC_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight);

// tinygemm bf16/fp16 (RM layout) x int8 (q-group, TC layout) = bf16/fp16 (RM
// layout)
torch::Tensor tinygemm_y_f16RM_x_f16RM_w_int8TC(
    torch::Tensor A,
    torch::Tensor B,
    int64_t qGroupSize,
    torch::Tensor qScaleAndZeros,
    bool weightOnRight);

//
// weights as fp16 / bf16
//

// tinygemm bf16/fp16 (TC layout) x bf16/fp16 (TC layout) = bf16/fp16 (TC
// layout) weightOnRight: y = A B^t !weightOnRight: y = (A B^t)^t A and B must
// both be fp16 or bf16; retval is the same dtype as input
torch::Tensor tinygemm_y_f16TC_x_f16TC_w_f16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight);

// tinygemm bf16/fp16 (RM layout) x bf16/fp16 (TC layout) = bf16/fp16 (RM
// layout) weightOnRight: y = A B^t !weightOnRight: y = (A B^t)^t A and B must
// both be fp16 or bf16; retval is the same dtype as input
torch::Tensor tinygemm_y_f16RM_x_f16RM_w_f16TC(
    torch::Tensor A,
    torch::Tensor B,
    bool weightOnRight);

torch::Tensor tinygemm_dequant_int4(torch::Tensor in);

// RAII object to manage a cudaEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(cudaStream_t stream, bool timer = false);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

  inline cudaEvent_t get() {
    return event_;
  }

  /// Wait on this event in this stream
  void streamWaitOnEvent(cudaStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  /// Returns the elapsed time from the other event
  float timeFrom(CudaEvent& from);

 private:
  cudaEvent_t event_;
};

} // namespace tinygemm
