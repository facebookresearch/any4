import unittest
import torch
import torch.nn as nn

import any4
from modules import QLinear
import tinygemm
import tinygemm.utils

class TestIntQ(unittest.TestCase):
    def test_group_quantize_tensor(self, bs=1, input_dim=4096, output_dim=4096, dtype=torch.float16, n_bit=4, group_size=64):
        w = torch.randn(input_dim, output_dim, dtype=dtype, device="cuda")
        x = torch.randn(bs, input_dim, dtype=dtype).to("cuda")
        wq1, scales_and_zeros1 = tinygemm.utils.group_quantize_tensor(w, n_bit, group_size)
        wq2, scales_and_zeros2 = any4.intq_quantize(w, n_bit, group_size)
        torch.allclose(wq1, wq2)
        torch.allclose(scales_and_zeros1, scales_and_zeros2)
