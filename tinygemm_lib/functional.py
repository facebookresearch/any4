# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import tinygemm

def valid_tinygemm_kernel_call(functional_api, w_inner_k):
    if functional_api=="linear_y_f16RM_x_f16RM_W_any4TC" and w_inner_k in [2, 4, 8]:
        return True
    if functional_api=="linear_y_f16TC_x_f16TC_W_any4TC" and w_inner_k in [2, 4, 8]:
        return True
    if functional_api=="linear_y_f16TC_W_any4TC_x_f16TC" and w_inner_k in [1, 2, 4]:
        return True
    if functional_api=="linear_y_f16RM_W_any4TC_x_f16RM" and w_inner_k in [1, 2, 4]:
        return True

def linear_y_f16TC_x_f16TC_W_int4TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_int4TC(
        x2, w2, q_group, w_scales_and_zeros, True
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
        y2, x.size(0), w_int32.size(0)
    )
    return y

def linear_y_f16TC_W_int4TC_x_f16TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_int4TC(
        w2, x2, q_group, w_scales_and_zeros, False
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
        y2, x.size(0), w_int32.size(0)
    )
    return y

def linear_y_f16RM_x_f16RM_W_int4TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
        x, w2, q_group, w_scales_and_zeros, True
    )
    return y

def linear_y_f16RM_W_int4TC_x_f16RM(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
        w2, x, q_group, w_scales_and_zeros, False
    )
    return y

def linear_y_f16TC_x_f16TC_W_int8TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, 1)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint8_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_int8TC(x2, w2, q_group, w_scales_and_zeros, True)
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(y2, x.shape[0], w_int32.shape[0])
    return y

def linear_y_f16TC_W_int8TC_x_f16TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint8_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_int8TC(w2, x2, q_group, w_scales_and_zeros, False)
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(y2, x.shape[0], x.shape[1])
    return y

def linear_y_f16RM_x_f16RM_W_int8TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint8_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int8TC(x, w2, q_group, w_scales_and_zeros, True)
    return y

def linear_y_f16RM_W_int8TC_x_f16RM(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint8_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int8TC(w2, x, q_group, w_scales_and_zeros, False)
    return y

def linear_y_f16TC_x_f16TC_W_any4TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_lut: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_any4TC(
        x2, w2, q_group, w_scales_and_zeros, w_lut, True
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
        y2, x.size(0), w_int32.size(0)
    )

    return y

def linear_y_f16TC_W_any4TC_x_f16TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_lut: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_any4TC(
        w2, x2, q_group, w_scales_and_zeros, w_lut, False
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
        y2, x.size(0), w_int32.size(0)
    )

    return y

def linear_y_f16RM_x_f16RM_W_any4TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_lut: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_any4TC(
        x, w2, q_group, w_scales_and_zeros, w_lut, True
    )

    return y

def linear_y_f16RM_W_any4TC_x_f16RM(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_lut: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k) if reshape_weight else w_int32
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_any4TC(
        w2, x, q_group, w_scales_and_zeros, w_lut, False
    )

    return y

def linear_y_f16TC_x_f16TC_W_f16TC(
        x: torch.Tensor,
        w: torch.Tensor,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, 1)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(w, w_inner_k) if reshape_weight else w
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(x2, w2, True)
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(y2, x.shape[0], w.shape[0])

    return y

def linear_y_f16TC_W_f16TC_x_f16TC(
        x: torch.Tensor,
        w: torch.Tensor,
        x_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(w, 1) if reshape_weight else w
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(w2, x2, False)
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(y2, x.shape[0], w.shape[0])

    return y

def linear_y_f16RM_x_f16RM_W_f16TC(
        x: torch.Tensor,
        w: torch.Tensor,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(w, w_inner_k) if reshape_weight else w
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(x, w2, True)

    return y

def linear_y_f16RM_W_f16TC_x_f16RM(
        x: torch.Tensor,
        w: torch.Tensor,
        w_inner_k: int = 4,
        reshape_weight: bool = True,
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(w, w_inner_k) if reshape_weight else w
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(w2, x, False)

    return y
