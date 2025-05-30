# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import random
import unittest
import torch

class TestConvert(unittest.TestCase):
    def setUp(self):
        try:
            import tinygemm
        except ImportError:
            self.skipTest("tinygemm is not installed")

    def test_convert_A_exact_tile(self):
        dev = torch.device("cuda:0")

        for dt in [torch.float16, torch.bfloat16]:
            for _ in range(10):
                m = random.randrange(1, 1000) * 16
                k = random.randrange(1, 1000) * 16

                t = torch.randn((m, k), dtype=dt, device=dev)
                t2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(t, 1)
                u = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(t2, m, k)

                assert torch.equal(t, u)

    def test_convert_A_arbitrary(self):
        dev = torch.device("cuda:0")

        for dt in [torch.float16, torch.bfloat16]:
            for _ in range(10):
                m = random.randrange(1, 16)
                k = random.randrange(1, 16)

                t = torch.randn((m, k), dtype=dt, device=dev)
                t2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(t, 1)
                u = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(t2, m, k)

                if not torch.equal(t, u):
                    print(t)
                    print(t2)
                    print(u)
                    print(t - u)

                assert torch.equal(t, u)

    def test_convert_B_exact_tile(self):
        dev = torch.device("cuda:0")

        for dt in [torch.float16, torch.bfloat16]:
            for inner_k_tiles in [1, 2]:
                for _ in range(10):
                    n = random.randrange(1, 1000) * 8
                    k = random.randrange(inner_k_tiles, 1000) * 16

                    t = torch.randn((n, k), dtype=dt, device=dev)
                    t2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                        t, inner_k_tiles
                    )
                    u = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
                        t2, n, k
                    )

                    assert torch.equal(t, u)

    def test_convert_B_arbitrary(self):
        dev = torch.device("cuda:0")

        for dt in [torch.float16, torch.bfloat16]:
            for inner_k_tiles in [1, 2]:
                for _ in range(10):
                    n = random.randrange(1, 10) * 8 + random.randrange(1, 8)
                    k = random.randrange(1, 10) * 16 * random.randrange(1, 16)

                    t = torch.randn((n, k), dtype=dt, device=dev)
                    t2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                        t, inner_k_tiles
                    )
                    u = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
                        t2, n, k
                    )

                    if not torch.equal(t, u):
                        print(t)
                        print(t2)
                        print(u)
                        print(t - u)

                    assert torch.equal(t, u)
