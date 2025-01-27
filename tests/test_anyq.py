import unittest
import torch
from parameterized import parameterized
import itertools
import numpy as np

import any4
from modules import QLinear
import tinygemm
import tinygemm.utils

class TestAnyQ(unittest.TestCase):
    # TODO: sweep over the arguments of anyq()
    def test_quantize_dequantize(self, M=4096, N=4096, group_size=64, n_bit=4, parallelize=True, dtype=torch.float32):
        device = "cuda"
        new_grouping = False
        assert group_size % 2**n_bit == 0, f"This test case assumes that group_size is a multiple of 2**n_bit, but instead we got group_size={group_size}, n_bit={n_bit}."
        assert N % group_size == 0, f"This test case assumes that number of elements per row is a multiple of group_size, but instead we got N={N}, group_size={group_size}."

        # if we only have 2**n_bit values per row and there is no scaling, quantizing and dequantizing should be exact
        w_vals = torch.randn(2**n_bit, dtype=dtype, device=device)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(M * N // 2**n_bit)]).view(M, N)
        w = w_vals[w_indices]

        wq, lut, scales_and_zeros = any4.anyq_quantize(w, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping, parallelize=parallelize)
        wdeq = any4.anyq_dequantize(wq, lut, scales_and_zeros=scales_and_zeros, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping)

        torch.testing.assert_close(w, wdeq)

    def test_do_y_f16TC_x_f16TC_W_any4TC(self, bs=64, input_dim=64, output_dim=64, q_group=32, w_inner_k=2, x_inner_k=1, dt=torch.bfloat16):
        dev = torch.device("cuda:0")
        n_bit=4
        new_grouping = False
        zero_point = True

        x = torch.randn((bs, input_dim), dtype=dt, device=dev)
        # currently tinygemm kernels return expected results if `zeros` are zero. So ensure each block of size `group_size` to be symmetrical.
        w_min, w_max = -8, 7
        w_vals = torch.linspace(start=w_min, end=w_max, steps=2**n_bit, dtype=dt, device=dev)
        w_indices = torch.stack([torch.randperm(2**n_bit) for _ in range(output_dim * input_dim // 2**n_bit)]).view(output_dim, input_dim)
        w = w_vals[w_indices]
        y_ref = x @ w.t()

        int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8
        w_int32, _, w_scales_and_zeros = any4.intq_quantize(w, n_bit, q_group, new_grouping=new_grouping, zero_point=zero_point)

        w_int32_b, int4_dequant_b, w_scales_and_zeros = any4.anyq_quantize(w, n_bit=n_bit, q_group_size=q_group, new_grouping=new_grouping, zero_point=zero_point)
        wdeq = any4.anyq_dequantize(w_int32_b, int4_dequant_b, scales_and_zeros=w_scales_and_zeros, n_bit=n_bit, q_group_size=q_group, new_grouping=new_grouping)
        torch.testing.assert_close(w, wdeq)

        int4_dequant_c = int4_dequant_b - (2**(n_bit - 1))
        w_int32_c = w_int32_b
        for i in range(output_dim):
            torch.testing.assert_close(int4_dequant_c[i][w_int32_c[i]], int4_dequant[w_int32[i]])

        y = tinygemm.functional.linear_y_f16TC_x_f16TC_W_any4TC(x, w_int32, int4_dequant, w_scales_and_zeros, q_group, w_inner_k, x_inner_k)
        torch.testing.assert_close(y, y_ref)

        y_c = tinygemm.functional.linear_y_f16TC_x_f16TC_W_any4TC(x, w_int32_c, int4_dequant_c, w_scales_and_zeros, q_group, w_inner_k, x_inner_k)
        torch.testing.assert_close(y_c, y_ref)
