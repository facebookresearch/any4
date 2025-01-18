import unittest
import torch
from parameterized import parameterized
import itertools
import numpy as np

import any4
from modules import QLinear
import tinygemm
import tinygemm.utils

class TestIntQ(unittest.TestCase):
    @parameterized.expand([
        (M, N, group_size, n_bit, unsigned)
        for M in [1, 4, 8, 24]
        for N in [256, 1024, 2048]
        for group_size in [32, 64, 128]
        for n_bit in [2, 3, 4, 6, 8]
        for unsigned in [True, False]
        if group_size % 2**n_bit == 0 and N % group_size == 0  # Optional condition to filter combinations
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

        wq, scales_and_zeros = any4.intq_quantize(w, n_bit=n_bit, q_group_size=group_size, new_grouping=new_grouping, unsigned=unsigned, zero_point=zero_point)
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

        wq1, scales_and_zeros1 = tinygemm.utils.group_quantize_tensor(w, n_bit, group_size)
        wq2, scales_and_zeros2 = any4.intq_quantize(w, n_bit, group_size, new_grouping=new_grouping, zero_point=zero_point)

        torch.testing.assert_close(wq1, wq2)
        torch.testing.assert_close(scales_and_zeros1, scales_and_zeros2)

        # TODO: add dequantize check?

    def test_intq(self, bs=1, input_dim=64, output_dim=64, dtype=torch.bfloat16, n_bit=4, group_size=64):
        new_grouping = False
        unsigned = True
        w = torch.randn(output_dim, input_dim, dtype=dtype, device="cuda")
        x = torch.randn(bs, input_dim, dtype=dtype).to("cuda")
        wq1, scales_and_zeros1 = tinygemm.utils.group_quantize_tensor(w, n_bit, group_size)
        wq2, scales_and_zeros2 = any4.intq_quantize(w, n_bit, group_size, new_grouping=new_grouping, zero_point=False, unsigned=unsigned)
        wq3, scales_and_zeros3 = any4.intq_quantize(w, n_bit, group_size, new_grouping=new_grouping, zero_point=True, unsigned=unsigned)
        torch.testing.assert_close(wq1, wq2)
        torch.testing.assert_close(wq1, wq3)
        # torch.testing.assert_close(scales_and_zeros1, scales_and_zeros2)
        torch.testing.assert_close(scales_and_zeros1, scales_and_zeros3)

        wdeq1 = any4.intq_dequantize(intq=wq1, scales_and_zeros=scales_and_zeros1, n_bit=n_bit, q_group_size=group_size, dtype=dtype, new_grouping=new_grouping)
        wdeq2 = any4.intq_dequantize(intq=wq2, scales_and_zeros=scales_and_zeros2, n_bit=n_bit, q_group_size=group_size, dtype=dtype, new_grouping=new_grouping, unsigned=unsigned)
        wdeq3 = any4.intq_dequantize(intq=wq3, scales_and_zeros=scales_and_zeros3, n_bit=n_bit, q_group_size=group_size, dtype=dtype, new_grouping=new_grouping, unsigned=unsigned)
        # self.assertTrue(torch.allclose(wdeq1, wdeq2))
        self.assertTrue(torch.allclose(wdeq1, wdeq3))

        y1 = tinygemm.functional.linear_y_f16TC_x_f16TC_W_int4TC(x, wq1, scales_and_zeros1, group_size)
        y2 = torch.nn.functional.linear(x, wdeq2)
        y3 = torch.nn.functional.linear(x, wdeq3)
        torch.testing.assert_close(y1, y2)
        # torch.testing.assert_close(y1, y3)
