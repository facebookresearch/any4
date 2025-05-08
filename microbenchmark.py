# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional
import argparse
import os
import torch

from lm_eval.utils import simple_parse_args_string

from utils import benchmark_in_ms, benchmark_cuda_only_in_ms
from any4 import quant_methods

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: support int4, int8
@torch.no_grad()
def microbenchmark_module(
    bs: int = 1,
    seqlen: int = 1,
    input_dim: int = 16384,
    output_dim: int = 16384,
    n_warmup: int = 50,
    n_iters: int = 100,
    device: str = default_device,
    dtype=torch.bfloat16,
    quant_method: Optional[Callable] = None,
    quant_args: Optional[Dict] = {},
):
    device = "cuda"
    bias=False

    x = torch.randn(bs * seqlen, input_dim, dtype=dtype, device=device)

    linear = torch.nn.Linear(
        input_dim,
        output_dim,
        dtype=dtype,
        device=device,
        bias=bias,
    )
    linear_time = benchmark_in_ms(linear, n_warmup, n_iters, x)
    linear_cuda_time = benchmark_cuda_only_in_ms(linear, n_warmup, n_iters, x)

    if quant_method:
        # Quantize
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        qlinear = quant_method(linear, pseudo=False, **quant_args)
        qlinear_time = benchmark_in_ms(qlinear, n_warmup, n_iters, x)
        qlinear_cuda_time = benchmark_cuda_only_in_ms(qlinear, n_warmup, n_iters, x)

    print("Baseline:")
    print(f"\tTotal: {linear_time:.4f} ms\tCUDA: {linear_cuda_time:.4f} ms")
    if quant_method:
            print("Quantized:")
            print(f"\tTotal: {qlinear_time:.4f} ms\tCUDA: {qlinear_cuda_time:.4f} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark quantization on a linear layer.")

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size of sample input.")
    parser.add_argument("--seqlen", type=int, default=1, help="Sequence length of sample input.")
    parser.add_argument("--input-dim", type=int, default=4096, help="Input dimension of linear layer.")
    parser.add_argument("--output-dim", type=int, default=4096, help="Output dimension of linear layer.")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations for benchmarking.")
    parser.add_argument("--quantize", type=str, default="intq", choices=quant_methods.keys(), help="Quantization method.")
    parser.add_argument("--quantize-args", type=str, help="Comma separated string args to pass to quantization method.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Input tensor data type.")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use.")

    args = parser.parse_args()

    # Pre-process some args
    torch_dtype = getattr(torch, args.dtype)
    quant_method = None if not args.quantize else quant_methods[args.quantize]
    quant_args = {} if not args.quantize_args else simple_parse_args_string(args.quantize_args)

    # Run Evaluation
    microbenchmark_module(
        bs=args.batch_size,
        seqlen=args.seqlen,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        n_warmup=args.warmup,
        n_iters=args.iters,
        device=args.device,
        dtype=torch_dtype,
        quant_method=quant_method,
        quant_args=quant_args,
    )
