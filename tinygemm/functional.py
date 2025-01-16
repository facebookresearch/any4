import torch
import tinygemm

def linear_y_f16TC_x_f16TC_W_int4TC(
        x: torch.Tensor,
        w_int32: torch.Tensor,
        w_scales_and_zeros: torch.Tensor,
        q_group: int,
        w_inner_k: int = 4,
        x_inner_k: int = 1,
    ) -> torch.Tensor:
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k)
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
        x_inner_k: int = 1
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k)
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
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k)
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
    ) -> torch.Tensor:
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k)
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_int4TC(
        w2, x, q_group, w_scales_and_zeros, False
    )
    return y
