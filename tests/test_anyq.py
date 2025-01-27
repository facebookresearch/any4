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
