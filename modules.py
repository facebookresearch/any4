import torch
import tinygemm

class INT4Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        group_size: int = 32,
        x_inner_k: int = 1,
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
        self.x_inner_k = x_inner_k
        self.w_inner_k = w_inner_k

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(input, self.x_inner_k)
        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(self.weight, self.w_inner_k)
        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_int4TC(
            x2, w2, self.group_size, self.scales_and_zeros, True
        )
        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
            y2, input.size(0), self.weight.size(0)
        )
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}"