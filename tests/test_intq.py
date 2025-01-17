import unittest
import torch
import torch.nn as nn

import any4
from modules import QLinear
import tinygemm
import tinygemm.utils

class TestIntQ(unittest.TestCase):
    def test_intq(self, bs=1, input_dim=4096, output_dim=4096, dtype=torch.bfloat16, n_bit=4, group_size=64):
        w = torch.randn(input_dim, output_dim, dtype=dtype, device="cuda") * 0.001
        x = torch.randn(bs, input_dim, dtype=dtype).to("cuda") * 0.001
        wq1, scales_and_zeros1 = tinygemm.utils.group_quantize_tensor(w, n_bit, group_size)
        wq2, scales_and_zeros2 = any4.intq_quantize(w, n_bit, group_size)
        torch.allclose(wq1, wq2)
        torch.allclose(scales_and_zeros1, scales_and_zeros2)

        wdeq1 = any4.intq_dequantize(intq=wq1, scales_and_zeros=scales_and_zeros1, n_bit=n_bit, q_group_size=group_size, dtype=dtype)
        wdeq2 = any4.intq_dequantize(intq=wq2, scales_and_zeros=scales_and_zeros2, n_bit=n_bit, q_group_size=group_size, dtype=dtype)
        torch.allclose(wdeq1, wdeq2)

        y1 = tinygemm.functional.linear_y_f16RM_x_f16RM_W_int4TC(x, wq1, scales_and_zeros1, group_size)
        y2 = torch.nn.functional.linear(x, wdeq2)
        self.assertTrue(torch.allclose(y1, y2, atol=1e-4))
