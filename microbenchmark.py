import torch

from utils import benchmark_in_ms
from modules import QLinear

# TODO: support int4, int8
def microbenchmark_module(bs=1, seqlen=1, input_dim=16384, output_dim=16384, dtype=torch.bfloat16, group_size=128):
    device = "cuda"
    qtype="int4"
    bias=False
    n_bit = 4
    w_inner_k = 4
    n_warmup=50
    n_iters=100

    x = torch.randn(bs * seqlen, input_dim, dtype=dtype, device=device)

    linear = torch.nn.Linear(
        input_dim,
        output_dim,
        dtype=dtype,
        device=device,
        bias=bias,
    )
    linear_time = benchmark_in_ms(linear, n_warmup, n_iters, x)
    print(f"{dtype}:\t {linear_time}")

    linear_quant = QLinear(
        in_features=input_dim,
        out_features=output_dim,
        bias=linear.bias is not None,
        device=device,
        qtype=qtype,
        dtype=dtype,
        group_size=group_size
    )
    linear_quant.reshape_weight(w_inner_k=w_inner_k)
    linear_quant_time = benchmark_in_ms(linear_quant, n_warmup, n_iters, x)
    print(f"{qtype}:\t\t {linear_quant_time}")


# TODO: add argument parsing
microbenchmark_module()