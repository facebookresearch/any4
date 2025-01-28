import unittest
import torch
from parameterized import parameterized
import itertools
import numpy as np

import any4
from modules import Any4Linear
import tinygemm
import tinygemm.functional
import tinygemm.utils

class TestAnyQ(unittest.TestCase):
    # TODO: sweep on more arguments
    # TODO: have separate test for some arguments to save time, e.g., separate test for parallelize=False
    @parameterized.expand([
        (M, N, group_size, n_bit, dtype)
        for M in [1, 4, 8, 24]
        for N in [256, 1024, 2048]
        for group_size in [64, 128] # TODO: support 32
        for n_bit in [4] # TODO: support 2, 3, 4, 6, 8
        for dtype in [torch.float32] # TODO: suport torch.float16, torch.bfloat16
        if group_size % 2**n_bit == 0 and N % group_size == 0  # Conditions to filter combinations
    ])
    def test_quantize_dequantize(self, M=4096, N=4096, group_size=64, n_bit=4, dtype=torch.float32):
        device = "cuda"
        new_grouping = False
        zero_point = True
        parallelize=True
        assert group_size % 2**n_bit == 0, f"This test case assumes that group_size is a multiple of 2**n_bit, but instead we got group_size={group_size}, n_bit={n_bit}."
        assert N % group_size == 0, f"This test case assumes that number of elements per row is a multiple of group_size, but instead we got N={N}, group_size={group_size}."

        # if we only have 2**n_bit values per row and there is no scaling, quantizing and dequantizing should be exact
        # TODO: have different w_vals for each row
        w_vals = torch.randn(2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(M * N // 2**n_bit)]).view(M, N)
        w = w_vals[w_indices]

        wq, lut, scales_and_zeros = any4.anyq_quantize(w, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping, parallelize=parallelize, zero_point=zero_point)
        #lut.sub_(2**(n_bit - 1))
        wdeq = any4.anyq_dequantize(wq, lut, scales_and_zeros=scales_and_zeros, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping)

        torch.testing.assert_close(w, wdeq)

    @parameterized.expand([
        (bs, input_dim, output_dim, dtype, group_size, functional_api, w_inner_k)
        for bs in [1, 2, 3, 29, 64]
        for input_dim in [64, 256] # TODO: support 1024, 2048
        for output_dim in [64] # TODO: support 128
        for dtype in [torch.float16, torch.bfloat16]
        for group_size in [32, 64, 128]
        for functional_api in ["linear_y_f16TC_x_f16TC_W_any4TC", "linear_y_f16TC_W_any4TC_x_f16TC", "linear_y_f16RM_x_f16RM_W_any4TC", "linear_y_f16RM_W_any4TC_x_f16RM"]
        for w_inner_k in [1, 2, 4] # TODO: support 8
        if group_size % 2**4 == 0 and input_dim % group_size == 0 and tinygemm.functional.valid_tinygemm_kernel_call(functional_api, w_inner_k) # Conditions to filter combinations
    ])
    def test_tinygemm_any4_functional(self, bs=64, input_dim=64, output_dim=64, dtype=torch.bfloat16, group_size=64, functional_api="linear_y_f16TC_x_f16TC_W_any4TC", w_inner_k=4):
        device = "cuda"
        n_bit=4
        new_grouping = False
        zero_point = True
        per_row=False

        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        w = w_vals[w_indices]

        x = torch.randn(bs, input_dim, dtype=dtype, device=device)
        y_ref = x @ w.t()

        w_int32, w_lut, w_scales_and_zeros = any4.anyq_quantize(w, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping, zero_point=zero_point, per_row=per_row)
        w_lut = w_lut - (2**(n_bit - 1))

        match functional_api:
            case "linear_y_f16TC_x_f16TC_W_any4TC":
                y = tinygemm.functional.linear_y_f16TC_x_f16TC_W_any4TC(
                    x, w_int32, w_lut, w_scales_and_zeros, group_size, w_inner_k, x_inner_k = 1
                )

            case "linear_y_f16TC_W_any4TC_x_f16TC":
                y = tinygemm.functional.linear_y_f16TC_W_any4TC_x_f16TC(
                    x, w_int32, w_lut, w_scales_and_zeros, group_size, w_inner_k, x_inner_k = 1
                )

            case "linear_y_f16RM_x_f16RM_W_any4TC":
                y = tinygemm.functional.linear_y_f16RM_x_f16RM_W_any4TC(
                    x, w_int32, w_lut, w_scales_and_zeros, group_size, w_inner_k
                )

            case "linear_y_f16RM_W_any4TC_x_f16RM":
                y = tinygemm.functional.linear_y_f16RM_W_any4TC_x_f16RM(
                    x, w_int32, w_lut, w_scales_and_zeros, group_size, w_inner_k
                )

            case _:
                raise ValueError(f"tinygemm.functional has no function {functional_api}.")

        torch.testing.assert_close(y, y_ref)

    # Temporary test case to debug kernel invocation
    def test_do_y_f16TC_x_f16TC_W_any4TC(self, bs=64, input_dim=64, output_dim=64, q_group=32, w_inner_k=2, x_inner_k=1, dt=torch.bfloat16):
        dev = torch.device("cuda:0")
        n_bit=4
        new_grouping = False
        zero_point = True
        per_row=False

        x = torch.randn((bs, input_dim), dtype=dt, device=dev)
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dt, device=dev)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        w = w_vals[w_indices]
        y_ref = x @ w.t()

        int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8
        w_int32, _, w_scales_and_zeros = any4.intq_quantize(w, n_bit, q_group, new_grouping=new_grouping, zero_point=zero_point)

        w_int32_b, int4_dequant_b, w_scales_and_zeros = any4.anyq_quantize(w, n_bit=n_bit, q_group_size=q_group, new_grouping=new_grouping, zero_point=zero_point, per_row=per_row)
        wdeq = any4.anyq_dequantize(w_int32_b, int4_dequant_b, scales_and_zeros=w_scales_and_zeros, n_bit=n_bit, q_group_size=q_group, new_grouping=new_grouping, per_row=per_row)
        torch.testing.assert_close(w, wdeq)

        int4_dequant_c = int4_dequant_b - (2**(n_bit - 1))
        w_int32_c = w_int32_b
        if per_row:
            for i in range(output_dim):
                torch.testing.assert_close(int4_dequant_c[i][w_int32_c[i]], int4_dequant[w_int32[i]])

        y = tinygemm.functional.linear_y_f16TC_x_f16TC_W_any4TC(x, w_int32, int4_dequant, w_scales_and_zeros, q_group, w_inner_k, x_inner_k)
        torch.testing.assert_close(y, y_ref)

        y_c = tinygemm.functional.linear_y_f16TC_x_f16TC_W_any4TC(x, w_int32_c, int4_dequant_c, w_scales_and_zeros, q_group, w_inner_k, x_inner_k)
        torch.testing.assert_close(y_c, y_ref)

    def test_tinygemm_module(self, bs=64, input_dim=64, output_dim=64, dtype=torch.bfloat16, n_bit=4, group_size=64, functional_api="linear_y_f16RM_x_f16RM_W_any4TC", w_inner_k=4):
        device = "cuda"
        per_row = False

        linear = torch.nn.Linear(input_dim, output_dim, dtype=dtype, device=device)

        x = torch.randn(bs, input_dim, dtype=dtype, device=device)
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        linear.weight.data = w_vals[w_indices]

        y_ref = linear(x)

        linear_quant = Any4Linear(
            in_features=input_dim,
            out_features=output_dim,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
            group_size=group_size,
            kernel=functional_api,
            w_inner_k=w_inner_k,
        )
        w_int32, w_lut, w_scales_and_zeros = any4.anyq_quantize(linear.weight, n_bit=n_bit, q_group_size=group_size, per_row=per_row)
        w_lut = w_lut - (2**(n_bit - 1))
        linear_quant.bias.data = linear.bias
        linear_quant.weight.data = w_int32
        linear_quant.scales_and_zeros.data = w_scales_and_zeros
        linear_quant.lut.data = w_lut

        y = linear_quant(x)

        torch.testing.assert_close(y, y_ref)
