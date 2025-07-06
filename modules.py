# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import tinygemm_lib.functional

# TODO: add FP4Linear, NF4Linear, MX4Linear

class Int4Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        group_size: int = 128,
        kernel: str = "linear_y_f16RM_W_int4TC_x_f16RM",
        w_inner_k: int = 4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = torch.nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=torch.int32),
            requires_grad = False,
        )
        self.scales_and_zeros = torch.nn.Parameter(
            torch.zeros((in_features // group_size, out_features, 2), device=device, dtype=dtype)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.kernel = kernel
        self.w_inner_k = w_inner_k
        self.weight_reshaped = False

    # TODO: add `set_weight()` function that will automatically reshape?
    def reshape_weight(self, w_inner_k: int = 4):
        if self.kernel == "linear_y_f16RM_x_f16RM_W_int4TC":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(self.weight, w_inner_k)
        elif self.kernel == "linear_y_f16RM_W_int4TC_x_f16RM":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(self.weight, w_inner_k)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")
        self.weight_reshaped = True
        self.w_inner_k = w_inner_k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        # Reshape input to 2D
        input = input.view(-1, orig_shape[-1])

        # Apply GEMM
        if self.kernel == "linear_y_f16RM_x_f16RM_W_int4TC":
            y = tinygemm_lib.functional.linear_y_f16RM_x_f16RM_W_int4TC(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        elif self.kernel == "linear_y_f16RM_W_int4TC_x_f16RM":
            y = tinygemm_lib.functional.linear_y_f16RM_W_int4TC_x_f16RM(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        elif self.kernel == "linear_y_f16TC_W_int4TC_x_f16TC":
            y = tinygemm_lib.functional.linear_y_f16TC_W_int4TC_x_f16TC(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")

        # Apply bias
        if self.bias is not None:
            y = y + self.bias

        # Resshape output to input's original shape
        y = y.view(*orig_shape[:-1], y.shape[-1])

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}"

class Int8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        group_size: int = 128,
        kernel: str = "linear_y_f16RM_W_int8TC_x_f16RM",
        w_inner_k: int = 2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = torch.nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=torch.int32),
            requires_grad = False,
        )
        self.scales_and_zeros = torch.nn.Parameter(
            torch.zeros((in_features // group_size, out_features, 2), device=device, dtype=dtype)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.kernel = kernel
        self.w_inner_k = w_inner_k
        self.weight_reshaped = False

    # TODO: add `set_weight()` function that will automatically reshape?
    def reshape_weight(self, w_inner_k: int = 2):
        if self.kernel == "linear_y_f16RM_x_f16RM_W_int8TC":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint8_layout(self.weight, w_inner_k)
        elif self.kernel == "linear_y_f16RM_W_int8TC_x_f16RM":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint8_layout(self.weight, w_inner_k)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")
        self.weight_reshaped = True
        self.w_inner_k = w_inner_k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        # Reshape input to 2D
        input = input.view(-1, orig_shape[-1])

        # Apply GEMM
        if self.kernel == "linear_y_f16RM_x_f16RM_W_int8TC":
            y = tinygemm_lib.functional.linear_y_f16RM_x_f16RM_W_int8TC(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        elif self.kernel == "linear_y_f16RM_W_int8TC_x_f16RM":
            y = tinygemm_lib.functional.linear_y_f16RM_W_int8TC_x_f16RM(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        elif self.kernel == "linear_y_f16TC_W_int8TC_x_f16TC":
            y = tinygemm_lib.functional.linear_y_f16TC_W_int8TC_x_f16TC(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")

        # Apply bias
        if self.bias is not None:
            y = y + self.bias

        # Resshape output to input's original shape
        y = y.view(*orig_shape[:-1], y.shape[-1])

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}"

class Any4Linear(torch.nn.Module):
    _N_BIT = 4
    @property
    def N_BIT(self):
        return self._N_BIT
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        group_size: int = 128,
        kernel: str = "linear_y_f16RM_x_f16RM_W_any4TC",
        w_inner_k: int = 4,
        per_row: bool = True,
    ) -> None:
        super().__init__()
        self.n_bit = 4
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.int32),
            requires_grad = False,
        )
        self.scales_and_zeros = torch.nn.Parameter(
            torch.empty((in_features // group_size, out_features, 2), device=device, dtype=dtype)
        )
        self.per_row = per_row
        if self.per_row:
            self.lut = torch.nn.Parameter(torch.empty(out_features, 2**self.N_BIT, device=device, dtype=dtype))
        else:
            self.lut = torch.nn.Parameter(torch.empty(2**self.N_BIT, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.kernel = kernel
        self.w_inner_k = w_inner_k
        self.weight_reshaped = False

    # TODO: add `set_weight()` function that will automatically reshape?
    def reshape_weight(self, w_inner_k: int = 4):
        if self.kernel == "linear_y_f16RM_x_f16RM_W_any4TC":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(self.weight, w_inner_k)
        elif self.kernel == "linear_y_f16RM_W_any4TC_x_f16RM":
            self.weight.data = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(self.weight, w_inner_k)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")
        self.weight_reshaped = True
        self.w_inner_k = w_inner_k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        # Reshape input to 2D
        input = input.view(-1, orig_shape[-1])

        # Apply GEMM
        if self.kernel == "linear_y_f16RM_x_f16RM_W_any4TC":
            y = tinygemm_lib.functional.linear_y_f16RM_x_f16RM_W_any4TC(input, self.weight, self.lut, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        elif self.kernel == "linear_y_f16RM_W_any4TC_x_f16RM":
            y = tinygemm_lib.functional.linear_y_f16RM_W_any4TC_x_f16RM(input, self.weight, self.lut, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel}")

        # Apply bias
        if self.bias is not None:
            y = y + self.bias

        # Reshape output to input's original shape
        y = y.view(*orig_shape[:-1], y.shape[-1])

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}, per_row={self.per_row}"
