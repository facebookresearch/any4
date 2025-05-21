# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from parameterized import parameterized
import itertools
import numpy as np

import any4

class TestIntQ(unittest.TestCase):
    @parameterized.expand([
        (M, N, group_size, n_bit, unsigned)
        for M in [1, 4, 8, 24]
        for N in [256, 1024, 2048]
        for group_size in [32, 64, 128]
        for n_bit in [2, 3, 4, 6, 8]
        for unsigned in [True, False]
        if group_size % 2**n_bit == 0 and N % group_size == 0  # Conditions to filter combinations
    ])
    def test_quantize_dequantize(self, M=4096, N=4096, group_size=64, n_bit=4, unsigned=True, dtype=torch.float32):
        device = "cuda"
        new_grouping = False
        zero_point = False
        assert group_size % 2**n_bit == 0, f"This test case assumes that group_size is a multiple of 2**n_bit, but instead we got group_size={group_size}, n_bit={n_bit}."
        assert N % group_size == 0, f"This test case assumes that number of elements per row is a multiple of group_size, but instead we got N={N}, group_size={group_size}."

        # if the weights are equally spaced and have the same the number of samples as 2**n_bit, quantizing and dequantizing should be exact
        a, b = np.random.normal(), np.random.normal()
        w_min, w_max = min(a, b), max(a, b)
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(M * N // 2**n_bit)]).view(M, N)
        w = w_vals[w_indices]

        wq, _, scales_and_zeros = any4.intq_quantize(w, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping, unsigned=unsigned, zero_point=zero_point)
        wdeq = any4.intq_dequantize(wq, scales_and_zeros=scales_and_zeros, n_bit=n_bit, q_group_size=group_size, dtype=dtype, new_grouping=new_grouping, unsigned=unsigned)

        torch.testing.assert_close(w, wdeq)

    @parameterized.expand([
        (M, N, group_size, n_bit, dtype)
        for M in [1, 4, 8, 24]
        for N in [256, 1024, 2048]
        for group_size in [32, 64, 128]
        for n_bit in [4, 8]
        for dtype in [torch.float16, torch.bfloat16]
    ])
    def test_tinygemm_quantize(self, M=4096, N=4096, group_size=64, n_bit=4, dtype=torch.bfloat16):
        new_grouping = False
        zero_point = True
        w = torch.randn(M, N, dtype=dtype, device="cuda")

        import tinygemm
        import tinygemm.utils
        wq1, scales_and_zeros1 = tinygemm.utils.group_quantize_tensor(w, n_bit, group_size)
        wq2, _, scales_and_zeros2 = any4.intq_quantize(w, n_bit, group_size, new_grouping=new_grouping, zero_point=zero_point)

        torch.testing.assert_close(wq1, wq2)
        torch.testing.assert_close(scales_and_zeros1, scales_and_zeros2)

        # TODO: add dequantize check?

    # TODO: support int8
    @parameterized.expand([
        (bs, input_dim, output_dim, dtype, n_bit, group_size, functional_api, w_inner_k)
        for bs in [1, 2, 3, 29, 64]
        for input_dim in [64, 256] # TODO: support 1024, 2048
        for output_dim in [64] # TODO: support 128
        for dtype in [torch.float16, torch.bfloat16]
        for n_bit in [4]
        for group_size in [32, 64, 128]
        for functional_api in ["linear_y_f16RM_x_f16RM_W_int4TC", "linear_y_f16TC_W_int4TC_x_f16TC", "linear_y_f16RM_x_f16RM_W_int4TC", "linear_y_f16RM_W_int4TC_x_f16RM", "linear_y_f16TC_W_int4TC_x_f16TC"]
        for w_inner_k in [1, 2, 4] # TODO: support 8
        if group_size % 2**n_bit == 0 and input_dim % group_size == 0 and not (functional_api=="linear_y_f16RM_x_f16RM_W_int4TC" and w_inner_k==1) # Conditions to filter combinations
    ])
    def test_tinygemm_int4_functional(self, bs=64, input_dim=64, output_dim=64, dtype=torch.bfloat16, n_bit=4, group_size=64, functional_api="linear_y_f16RM_x_f16RM_W_int4TC", w_inner_k=2):
        device = "cuda"
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        w = w_vals[w_indices]

        x = torch.randn(bs, input_dim, dtype=dtype, device=device)
        y_ref = x @ w.t()

        import tinygemm
        import tinygemm.utils
        w_int32, w_scales_and_zeros = tinygemm.utils.group_quantize_tensor(
            w, n_bit=n_bit, q_group_size=group_size
        )

        match functional_api:
            case "linear_y_f16RM_x_f16RM_W_int4TC":
                y = tinygemm.functional.linear_y_f16RM_x_f16RM_W_int4TC(
                    x, w_int32, w_scales_and_zeros, group_size, w_inner_k
                )

            case "linear_y_f16TC_W_int4TC_x_f16TC":
                y = tinygemm.functional.linear_y_f16TC_W_int4TC_x_f16TC(
                    x, w_int32, w_scales_and_zeros, group_size, w_inner_k, x_inner_k=1
                )

            case "linear_y_f16RM_x_f16RM_W_int4TC":
                y = tinygemm.functional.linear_y_f16TC_W_int4TC_x_f16TC(
                    x, w_int32, w_scales_and_zeros, group_size, w_inner_k
                )

            case "linear_y_f16RM_W_int4TC_x_f16RM":
                y = tinygemm.functional.linear_y_f16TC_W_int4TC_x_f16TC(
                    x, w_int32, w_scales_and_zeros, group_size, w_inner_k
                )

            case _:
                raise ValueError(f"tinygemm.functional has no function {functional_api}.")

        torch.testing.assert_close(y, y_ref)

    # TODO: support int4, int8
    # TODO: sweep over parameters
    def test_tinygemm_module(self, bs=64, input_dim=64, output_dim=64, dtype=torch.bfloat16, n_bit=4, group_size=64, functional_api="linear_y_f16RM_x_f16RM_W_int4TC", w_inner_k=2):
        device = "cuda"

        linear = torch.nn.Linear(input_dim, output_dim, dtype=dtype, device=device)

        x = torch.randn(bs, input_dim, dtype=dtype, device=device)
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        linear.weight.data = w_vals[w_indices]

        y_ref = linear(x)

        from modules import Int4Linear
        import tinygemm
        import tinygemm.utils
        linear_quant = Int4Linear(
            in_features=input_dim,
            out_features=output_dim,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
            group_size=group_size,
            kernel=functional_api,
            w_inner_k=w_inner_k,
        )
        w_int32, scales_and_zeros = tinygemm.utils.group_quantize_tensor(linear.weight, n_bit, group_size)
        linear_quant.bias.data = linear.bias
        linear_quant.weight.data = w_int32
        linear_quant.scales_and_zeros.data = scales_and_zeros

        y = linear_quant(x)

        torch.testing.assert_close(y, y_ref)

    # TODO: support int4, int8
    # TODO: sweep over parameters
    def test_conversion_module(self, bs=64, input_dim=64, output_dim=64, dtype=torch.bfloat16, n_bit=4, group_size=64, functional_api="linear_y_f16RM_x_f16RM_W_int4TC", w_inner_k=2):
        device = "cuda"
        new_grouping = False
        zero_point = True

        linear = torch.nn.Linear(input_dim, output_dim, dtype=dtype, device=device)

        x = torch.randn(bs, input_dim, dtype=dtype, device=device)
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        linear.weight.data = w_vals[w_indices]

        y_ref = linear(x)

        linear_quant = any4.intq(
            module=linear,
            n_bit=n_bit,
            group_size=group_size,
            new_grouping=new_grouping,
            zero_point=zero_point,
            pseudo=False,
        )
        y = linear_quant(x)

        torch.testing.assert_close(y, y_ref)
