import unittest
import torch
import torch.nn as nn

import any4

class TestLinearMatmul(unittest.TestCase):
    def test_intq(self, bs=1, input_dim=4096, output_dim=4096, dtype=torch.float16):
        linear = nn.Linear(input_dim, output_dim, dtype=dtype).to("cuda")
        x = torch.randn(bs, input_dim, dtype=dtype).to("cuda")

        linear_quant = any4.intq(linear, pseudo=False)
        linear_pseudoquant = any4.intq(linear, pseudo=True)

        # Perform the forward pass using the Linear module
        y = linear_quant(x)
        y_pseudo = linear_pseudoquant(x)

        # Assert that the output from the Linear module matches the expected output
        self.assertTrue(torch.allclose(y, y_pseudo, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
