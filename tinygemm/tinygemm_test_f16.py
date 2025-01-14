# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-unsafe

import unittest

import torch

torch.ops.load_library("//tinygemm:tinygemm")


# W on right
class Test_y_f16TC_x_f16TC_W_f16TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [1, 5, 16, 32, 48, 50]:
                for k in [1024, 2048, 4096, 8192]:
                    for w_inner_k in [1, 2]:
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = x @ w.t()

                        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            x, 1
                        )
                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            w, w_inner_k
                        )
                        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                            x2, w2, True
                        )
                        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
                            y2, m, k
                        )

                        if not torch.equal(y_ref, y):
                            print("error", dt, m, k, w_inner_k)

                        assert torch.equal(y_ref, y)

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [1, 5, 16, 32, 48, 50]:
                # bf16 has 2 inner k-tiles, so any k that is a multiple of 32 should work
                for k in range(32, 1024, 32):
                    for w_inner_k in [1, 2]:
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = x @ w.t()

                        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            x, 1
                        )
                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            w, w_inner_k
                        )
                        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                            x2, w2, True
                        )
                        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
                            y2, m, k
                        )

                        assert torch.equal(y_ref, y)

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [5, 16, 32, 48]:
                for n in [16, 32, 48]:
                    # keep the reduction size small for better ability to test equivalency
                    for k in [64, 160, 1024, 2048]:
                        for w_inner_k in [1, 2]:
                            x = torch.randn((m, k), dtype=dt, device=dev) * 0.1
                            w = torch.randn((n, k), dtype=dt, device=dev) * 0.1
                            y_ref = x @ w.t()

                            x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                                x, 1
                            )
                            w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                                w, w_inner_k
                            )
                            y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                                x2, w2, True
                            )
                            y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
                                y2, m, n
                            )

                            torch.testing.assert_close(y_ref, y, atol=0.01, rtol=0.1)


# W on left
class Test_y_f16TC_W_f16TC_x_f16TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [7, 16, 32, 48]:
                for k in [1024, 2048, 4096, 8192]:
                    for x_inner_k in [1, 2]:
                        w = torch.eye(k, dtype=dt, device=dev)
                        x = torch.randn((n, k), dtype=dt, device=dev)
                        y_ref = (w @ x.t()).t()

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            w, 1
                        )
                        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            x, x_inner_k
                        )
                        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                            w2, x2, False
                        )
                        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
                            y2, n, k
                        )

                        assert torch.equal(y_ref, y)

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [7, 16, 21, 32, 48]:
                # bf16 has 2 inner k-tiles, so any k that is a multiple of 32 should work
                for k in range(32, 1024, 32):
                    for x_inner_k in [1, 2]:
                        w = torch.eye(k, dtype=dt, device=dev)
                        x = torch.randn((n, k), dtype=dt, device=dev)
                        y_ref = (w @ x.t()).t()

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            w, 1
                        )
                        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            x, x_inner_k
                        )
                        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                            w2, x2, False
                        )
                        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
                            y2, n, k
                        )

                        assert torch.equal(y_ref, y)

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [16, 32, 48]:
                for n in [5, 16, 21, 32, 48]:
                    for k in [64, 160, 1024, 2048]:
                        for x_inner_k in [1, 2]:
                            w = torch.randn((m, k), dtype=dt, device=dev) * 0.1
                            x = torch.randn((n, k), dtype=dt, device=dev) * 0.1
                            y_ref = (w @ x.t()).t()

                            w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                                w, 1
                            )
                            x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                                x, x_inner_k
                            )
                            y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_f16TC(
                                w2, x2, False
                            )
                            y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
                                y2, n, m
                            )

                            torch.testing.assert_close(y_ref, y, atol=0.01, rtol=0.1)


# W on right
class Test_y_f16RM_x_f16RM_W_f16TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [16, 32, 48]:
                for k in [64, 1024, 2048, 4096, 8192]:
                    for w_inner_k in [1, 2]:
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = x @ w

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            w, w_inner_k
                        )
                        y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                            x, w2, True
                        )

                        assert torch.equal(y_ref, y)

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [5, 16, 24, 29]:
                # bf16 has 2 inner k-tiles, so any k that is a multiple of 32 should work
                for k in range(32, 1024, 32):
                    for w_inner_k in [1, 2]:
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = x @ w

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                            w, w_inner_k
                        )
                        y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                            x, w2, True
                        )

                        assert torch.equal(y_ref, y)

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [16, 32, 48]:
                for n in [16, 32, 48]:
                    # keep the reduction size small for better ability to test equivalency
                    for k in [64, 1024, 2048]:
                        for w_inner_k in [1, 2]:
                            n = k
                            x = torch.randn((m, k), dtype=dt, device=dev) * 0.1
                            w = torch.randn((n, k), dtype=dt, device=dev) * 0.1
                            y_ref = x @ w.t()

                            w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(
                                w, w_inner_k
                            )
                            y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                                x, w2, True
                            )

                            torch.testing.assert_close(y_ref, y, atol=0.01, rtol=0.1)


# W on left
class Test_y_f16RM_W_f16TC_x_f16RM(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [5, 8, 16, 32]:
                for k in [64, 1024, 2048, 4096, 8192]:
                    for w_inner_k in [1]:
                        x = torch.randn((n, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = (w @ x.t()).t()

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            w, w_inner_k
                        )
                        y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                            w2, x, False
                        )

                        assert torch.equal(y_ref, y)

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [3, 8, 11, 16, 24]:
                # bf16 has 2 inner k-tiles, so any k that is a multiple of 32 should work
                for k in range(32, 1024, 32):
                    for w_inner_k in [1]:
                        x = torch.randn((n, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)
                        y_ref = (w @ x.t()).t()

                        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                            w, w_inner_k
                        )
                        y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                            w2, x, False
                        )

                        assert torch.equal(y_ref, y)

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [16, 32, 48]:
                for n in [8, 16, 32, 48]:
                    # keep the reduction size small for better ability to test equivalency
                    for k in [64, 1024, 2048]:
                        for w_inner_k in [1]:
                            w = torch.randn((m, k), dtype=dt, device=dev) * 0.1
                            x = torch.randn((n, k), dtype=dt, device=dev) * 0.1
                            y_ref = (w @ x.t()).t()

                            w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(
                                w, w_inner_k
                            )
                            y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_f16TC(
                                w2, x, False
                            )

                            torch.testing.assert_close(y_ref, y, atol=0.01, rtol=0.1)
