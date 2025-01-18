import unittest
import torch
import torch.nn as nn

import any4
from modules import QLinear
import tinygemm
import tinygemm.functional
import tinygemm.utils

class TestModules(unittest.TestCase):
    def test_intq(self, bs=1, input_dim=4096, output_dim=4096, dtype=torch.float16, n_bit=4, group_size=64):
        linear = nn.Linear(input_dim, output_dim, dtype=dtype).to("cuda")
        x = torch.randn(bs, input_dim, dtype=dtype).to("cuda") * 0.01

        linear_pseudoquant = any4.intq(linear, n_bit=n_bit, group_size=group_size, pseudo=True)

        # linear_quant = any4.intq(linear, n_bit=n_bit, group_size=group_size, pseudo=False)
        linear_quant = QLinear(
            in_features=input_dim,
            out_features=output_dim,
            bias=linear.bias is not None,
            device="cuda",
            qtype="int4",
            dtype=dtype,
            group_size=group_size
        )
        w_int32, scales_and_zeros = tinygemm.utils.group_quantize_tensor(linear.weight, n_bit, group_size)
        linear_quant.bias.data = linear.bias
        linear_quant.weight.data = w_int32
        linear_quant.scales_and_zeros.data = scales_and_zeros

        # Perform dequantize
        linear_quant_weight_dequantized = any4.intq_dequantize(intq=linear_quant.weight, scales_and_zeros=linear_quant.scales_and_zeros, n_bit=n_bit, q_group_size=group_size)
        self.assertTrue(torch.allclose(linear_quant_weight_dequantized, linear_pseudoquant.weight))

        # Perform forward pass using functional APIs
        y_pseudoquant = torch.nn.functional.linear(x, linear_pseudoquant.weight)
        y_quant = tinygemm.functional.linear_y_f16RM_x_f16RM_W_int4TC(x, linear_quant.weight, linear_quant.scales_and_zeros, group_size)
        self.assertTrue(torch.allclose(y_quant, y_pseudoquant, atol=1e-4))

        # Perform the forward pass using the Linear module
        y_pseudoquant = linear_pseudoquant(x)
        y_quant = linear_quant(x)

        # Assert that the output from the Linear module matches the expected output
        self.assertTrue(torch.allclose(y_quant, y_pseudoquant, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
