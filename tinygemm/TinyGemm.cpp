// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "TinyGemm.h"

#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(tinygemm, m) {
    m.doc() = "tinygemm: low-bit CUDA GEMM library";
}

TORCH_LIBRARY_FRAGMENT(tinygemm, m) {
  // convert to A
  m.def(
      "convert_matrix_to_m16n8k16_A_layout(Tensor t, int innerKTiles) -> Tensor");
  m.def(
      "convert_matrix_to_m16n8k16_Aint4_layout(Tensor t, int innerKTiles) -> Tensor");
  m.def(
      "convert_matrix_to_m16n8k16_Aint8_layout(Tensor t, int innerKTiles) -> Tensor");

  // convert from A
  m.def(
      "convert_matrix_from_m16n8k16_A_layout(Tensor t, int m, int k) -> Tensor");

  // convert to B
  m.def(
      "convert_matrix_to_m16n8k16_B_layout(Tensor t, int innerKTiles) -> Tensor");
  m.def(
      "convert_matrix_to_m16n8k16_Bint4_layout(Tensor t, int innerKTiles) -> Tensor");
  m.def(
      "convert_matrix_to_m16n8k16_Bint8_layout(Tensor t, int innerKTiles) -> Tensor");

  // convert from B
  m.def(
      "convert_matrix_from_m16n8k16_B_layout(Tensor t, int n, int k) -> Tensor");

  //
  // int4 weights
  //

  // tinygemm bf16/fp16 (TC layout) x int4 (q-group, TC layout) = bf16/fp16 (TC
  // layout)
  m.def(
      "tinygemm_y_f16TC_x_f16TC_w_int4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, bool weightOnRight) -> Tensor");

  // tinygemm bf16/fp16 (RM layout) x int4 (q-group, TC layout) = bf16/fp16 (RM
  // layout)
  m.def(
      "tinygemm_y_f16RM_x_f16RM_w_int4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, bool weightOnRight) -> Tensor");

  //
  // any4 weights
  //

  // tinygemm bf16/fp16 (TC layout) x int4 (q-group, TC layout) = bf16/fp16 (TC
  // layout) with arbitrary dequantization
  m.def(
      "tinygemm_y_f16TC_x_f16TC_w_any4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, Tensor int4DequantValues, bool weightOnRight) -> Tensor");

  // tinygemm bf16/fp16 (RM layout) x int4 (q-group, TC layout) = bf16/fp16 (RM
  // layout) with arbitrary dequantization
  m.def(
      "tinygemm_y_f16RM_x_f16RM_w_any4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, Tensor int4DequantValues, bool weightOnRight) -> Tensor");

  //
  // mx4 weights
  //

  // tinygemm bf16/fp16 (TC layout) x mx4 (e8 row scale) = bf16/fp16 (TC
  // layout) with arbitrary dequantization
  m.def(
      "tinygemm_y_f16TC_x_f16TC_w_mx4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor mx4Exponents, bool weightOnRight) -> Tensor");

  // tinygemm bf16/fp16 (RM layout) x mx4 (e8 row scale) = bf16/fp16 (RM
  // layout) with arbitrary dequantization
  m.def(
      "tinygemm_y_f16RM_x_f16RM_w_mx4TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor mx4Exponents, bool weightOnRight) -> Tensor");

  //
  // int8 weights
  //

  // tinygemm bf16/fp16 (TC layout) x int8 (q-group, TC layout) = bf16/fp16 (TC
  // layout)
  m.def(
      "tinygemm_y_f16TC_x_f16TC_w_int8TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, bool weightOnRight) -> Tensor");

  // tinygemm bf16/fp16 (RM layout) x int8 (q-group, TC layout) = bf16/fp16 (RM
  // layout)
  m.def(
      "tinygemm_y_f16RM_x_f16RM_w_int8TC(Tensor A, Tensor B, "
      "int qGroupSize, Tensor qScaleAndZeros, bool weightOnRight) -> Tensor");

  //
  // bf16 / fp16 weights
  //

  // tinygemm bf16/fp16 (TC layout) x bf16/fp16 (TC layout) = bf16/fp16 (TC
  // layout)
  m.def(
      "tinygemm_y_f16TC_x_f16TC_w_f16TC(Tensor A, Tensor B, bool weightOnRight) -> Tensor");

  // tinygemm bf16/fp16 (RM layout) x bf16/fp16 (TC layout) = bf16/fp16 (RM
  // layout)
  m.def(
      "tinygemm_y_f16RM_x_f16RM_w_f16TC(Tensor A, Tensor B, bool weightOnRight) -> Tensor");

  // debug only
  m.def("tinygemm_dequant_int4(Tensor t) -> Tensor");
}

TORCH_LIBRARY(tinygemm, m) {
  // to A
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_A_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_A_layout));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_Aint4_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_Aint4_layout));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_Aint8_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_Aint8_layout));

  // from A
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_from_m16n8k16_A_layout"),
      TORCH_FN(tinygemm::convert_matrix_from_m16n8k16_A_layout));

  // to B
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_B_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_B_layout));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_Bint4_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_Bint4_layout));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_to_m16n8k16_Bint8_layout"),
      TORCH_FN(tinygemm::convert_matrix_to_m16n8k16_Bint8_layout));

  // from B
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::convert_matrix_from_m16n8k16_B_layout"),
      TORCH_FN(tinygemm::convert_matrix_from_m16n8k16_B_layout));

  // bf16 x int4 = bf16 or f16 x int4 = f16
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16TC_x_f16TC_w_int4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16TC_x_f16TC_w_int4TC));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16RM_x_f16RM_w_int4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16RM_x_f16RM_w_int4TC));

  // bf16 x any4 = bf16 or f16 x any4 = f16 (arbitrary dequantization)
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16TC_x_f16TC_w_any4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16TC_x_f16TC_w_any4TC));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16RM_x_f16RM_w_any4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16RM_x_f16RM_w_any4TC));

  // bf16 x mx4 = bf16 or f16 x mx4 = f16 (MX4 fp4 + e8 row-wise scale)
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16TC_x_f16TC_w_mx4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16TC_x_f16TC_w_mx4TC));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16RM_x_f16RM_w_mx4TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16RM_x_f16RM_w_mx4TC));

  // bf16 x int8 = bf16 or f16 x int8 = f16
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16TC_x_f16TC_w_int8TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16TC_x_f16TC_w_int8TC));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16RM_x_f16RM_w_int8TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16RM_x_f16RM_w_int8TC));

  // bf16 x bf16 = bf16 or f16 x f16 = f16
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16TC_x_f16TC_w_f16TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16TC_x_f16TC_w_f16TC));
  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_y_f16RM_x_f16RM_w_f16TC"),
      TORCH_FN(tinygemm::tinygemm_y_f16RM_x_f16RM_w_f16TC));

  m.impl(
      TORCH_SELECTIVE_NAME("tinygemm::tinygemm_dequant_int4"),
      TORCH_FN(tinygemm::tinygemm_dequant_int4));
}

namespace tinygemm {

CudaEvent::CudaEvent(cudaStream_t stream, bool timer) : event_(nullptr) {
  C10_CUDA_CHECK(cudaEventCreateWithFlags(
      &event_, timer ? cudaEventDefault : cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event_, stream));
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept
    : event_(std::move(event.event_)) {
  event.event_ = nullptr;
}

CudaEvent::~CudaEvent() {
  if (event_) {
    C10_CUDA_CHECK(cudaEventDestroy(event_));
  }
}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
  event_ = std::move(event.event_);
  event.event_ = nullptr;

  return *this;
}

void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaEvent::cpuWaitOnEvent() {
  C10_CUDA_CHECK(cudaEventSynchronize(event_));
}

float CudaEvent::timeFrom(CudaEvent& from) {
  cpuWaitOnEvent();
  float ms = 0;
  C10_CUDA_CHECK(cudaEventElapsedTime(&ms, from.event_, event_));

  return ms;
}

} // namespace tinygemm
