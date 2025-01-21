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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.qtype == "int4":
            y = tinygemm.functional.linear_y_f16RM_x_f16RM_W_int4TC(input, self.weight, self.scales_and_zeros, self.group_size)
        else:
            raise ValueError(f"Unsupported quantization type {self.qtype}")

        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, group_size={self.group_size}"
