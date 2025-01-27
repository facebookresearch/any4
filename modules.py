import torch
import tinygemm.functional

class QLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        qtype = None,
        dtype = None,
        group_size: int = 32,
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
        self.qtype = qtype
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
        # Reshape input to 2D
        input = input.view(-1, input.shape[-1])

        # Apply GEMM
        if self.qtype == "int4":
            if self.kernel == "linear_y_f16RM_x_f16RM_W_int4TC":
                y = tinygemm.functional.linear_y_f16RM_x_f16RM_W_int4TC(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
            elif self.kernel == "linear_y_f16RM_W_int4TC_x_f16RM":
                y = tinygemm.functional.linear_y_f16RM_W_int4TC_x_f16RM(input, self.weight, self.scales_and_zeros, self.group_size, w_inner_k=self.w_inner_k, reshape_weight=not self.weight_reshaped)
            else:
                raise ValueError(f"Unsupported kernel type {self.kernel}")
        else:
            raise ValueError(f"Unsupported quantization type {self.qtype}")

        # Apply bias
        if self.bias is not None:
            y = y + self.bias

        # Resshape output to input's original shape
        y = y.view(*input.shape[:-1], y.shape[-1])

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}"
