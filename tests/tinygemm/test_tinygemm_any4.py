# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
import torch

from tinygemm_lib.utils import group_quantize_tensor

def do_y_f16TC_x_f16TC_W_any4TC(x, w, q_group, w_inner_k, x_inner_k=1):
    y_ref = x @ w.t()

    # lookup table to represent int4 uniform quantization as any4
    int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8

    w_int32, w_scales_and_zeros = group_quantize_tensor(
        w, n_bit=4, q_group_size=q_group
    )

    # negate values to test that we are actually using any4, as opposed to int4
    int4_dequant *= -1.0
    w_scales_and_zeros[:, :, 0] *= -1.0

    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_any4TC(
        x2, w2, q_group, w_scales_and_zeros, int4_dequant, True
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
        y2, x.size(0), w_int32.size(0)
    )

    return y, y_ref


def do_y_f16TC_W_any4TC_x_f16TC(x, w, q_group, w_inner_k, x_inner_k):
    y_ref = (w @ x.t()).t()

    # lookup table to represent int4 uniform quantization as any4
    int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8

    w_int32, w_scales_and_zeros = group_quantize_tensor(
        w, n_bit=4, q_group_size=q_group
    )

    # negate values to test that we are actually using any4, as opposed to int4
    int4_dequant *= -1.0
    w_scales_and_zeros[:, :, 0] *= -1.0

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k)
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_any4TC(
        w2, x2, q_group, w_scales_and_zeros, int4_dequant, False
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
        y2, x.size(0), w_int32.size(0)
    )

    return y, y_ref


def do_y_f16RM_x_f16RM_W_any4TC(x, w, q_group, w_inner_k):
    y_ref = x @ w.t()

    # lookup table to represent int4 uniform quantization as any4
    int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8

    w_int32, w_scales_and_zeros = group_quantize_tensor(
        w, n_bit=4, q_group_size=q_group
    )

    # negate values to test that we are actually using any4, as opposed to int4
    int4_dequant *= -1.0
    w_scales_and_zeros[:, :, 0] *= -1.0

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_int32, w_inner_k)
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_any4TC(
        x, w2, q_group, w_scales_and_zeros, int4_dequant, True
    )

    return y, y_ref


def do_y_f16RM_W_any4TC_x_f16RM(x, w, q_group, w_inner_k):
    y_ref = (w @ x.t()).t()

    # lookup table to represent int4 uniform quantization as any4
    int4_dequant = torch.arange(16, dtype=x.dtype, device=x.device) - 8

    w_int32, w_scales_and_zeros = group_quantize_tensor(
        w, n_bit=4, q_group_size=q_group
    )

    # negate values to test that we are actually using any4, as opposed to int4
    int4_dequant *= -1.0
    w_scales_and_zeros[:, :, 0] *= -1.0

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_int32, w_inner_k)
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_any4TC(
        w2, x, q_group, w_scales_and_zeros, int4_dequant, False
    )

    return y, y_ref


class Test_y_f16TC_x_f16TC_W_any4TC(unittest.TestCase):
    def setUp(self):
        try:
            import tinygemm
        except ImportError:
            self.skipTest("tinygemm is not installed")

    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [5, 16, 19, 32, 48]:
                for k in [256, 1024, 2048, 4096, 8192]:
                    for q_group in [32, 64, 128, 256]:
                        for w_inner_k in [2, 4, 8]:
                            x = torch.randn((m, k), dtype=dt, device=dev)
                            w = torch.eye(k, dtype=dt, device=dev)

                            y, y_ref = do_y_f16TC_x_f16TC_W_any4TC(
                                x, w, q_group, w_inner_k, 1
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, m, k, q_group, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [5, 16, 19]:
                for w_inner_k in [2, 4, 8]:
                    q_group = w_inner_k * 16
                    for k in range(w_inner_k * 16, 1024, w_inner_k * 16):
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)

                        y, y_ref = do_y_f16TC_x_f16TC_W_any4TC(
                            x, w, q_group, w_inner_k, 1
                        )

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, m, k, q_group, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for _ in range(5):
                # m, being non-weight, can be non multiples of the tile size
                for m in [3, 16, 31, 32, 48]:
                    # n however must be a multiple of the tile size
                    for n in [8, 16, 24, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [2, 4, 8]:
                                for q_group in [32, 64, 128, 256]:
                                    x = torch.randn((m, k), dtype=dt, device=dev)
                                    # in order to test equivalency, restrict the int4 quantized
                                    # tensor to [0, 1], so at least this will test positionality
                                    w = torch.randint(
                                        0, 2, (n, k), dtype=dt, device=dev
                                    )

                                    y, y_ref = do_y_f16TC_x_f16TC_W_any4TC(
                                        x, w, q_group, w_inner_k, 1
                                    )

                                    diff = (y_ref.float() - y.float()).abs()
                                    avg_err = diff.sum() / (m * n)

                                    assert avg_err < 1e-1


class Test_y_f16TC_W_any4TC_x_f16TC(unittest.TestCase):
    def setUp(self):
        try:
            import tinygemm
        except ImportError:
            self.skipTest("tinygemm is not installed")

    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [5, 16, 19, 32, 48]:
                for k in [256, 1024, 2048, 4096, 8192]:
                    for w_inner_k in [1, 2, 4]:
                        for x_inner_k in [1, 2]:
                            q_group = w_inner_k * 32
                            w = torch.eye(k, dtype=dt, device=dev)
                            x = torch.randn((n, k), dtype=dt, device=dev)

                            y, y_ref = do_y_f16TC_W_any4TC_x_f16TC(
                                x, w, q_group, w_inner_k, x_inner_k
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, n, k, q_group, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [5, 16, 19, 32, 48]:
                for w_inner_k in [1, 2, 4]:
                    for x_inner_k in [1, 2]:
                        q_group = w_inner_k * 32
                        for k in range(w_inner_k * 32, 1024, w_inner_k * 32):
                            w = torch.eye(k, dtype=dt, device=dev)
                            x = torch.randn((n, k), dtype=dt, device=dev)

                            y, y_ref = do_y_f16TC_W_any4TC_x_f16TC(
                                x, w, q_group, w_inner_k, x_inner_k
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, n, k, q_group, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            for dt in [torch.bfloat16, torch.float16]:
                # m must be a multiple of the tile size
                for m in [16, 32, 48]:
                    # n, being non-weight, can be non multiples of the tile size
                    for n in [3, 8, 11, 16, 31, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [1, 2, 4]:
                                for x_inner_k in [1, 2]:
                                    for q_group in [32, 64, 128, 256]:
                                        # in order to test equivalency, restrict the int4 quantized
                                        # tensor to [0, 1], so at least this will test positionality
                                        w = torch.randint(
                                            0, 2, (m, k), dtype=dt, device=dev
                                        )
                                        x = torch.randn((n, k), dtype=dt, device=dev)

                                        y, y_ref = do_y_f16TC_W_any4TC_x_f16TC(
                                            x, w, q_group, w_inner_k, x_inner_k
                                        )

                                        diff = (y_ref.float() - y.float()).abs()
                                        avg_err = diff.sum() / (m * n)

                                        if avg_err >= 1e-1:
                                            print(
                                                "error",
                                                m,
                                                n,
                                                k,
                                                w_inner_k,
                                                x_inner_k,
                                                q_group,
                                                avg_err,
                                            )
                                            assert avg_err < 1e-1


class Test_y_f16RM_x_f16RM_W_any4TC(unittest.TestCase):
    def setUp(self):
        try:
            import tinygemm
        except ImportError:
            self.skipTest("tinygemm is not installed")

    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [5, 16, 32, 45, 48, 64, 80]:
                for k in [1024, 2048, 4096, 8192]:
                    for q_group in [32, 64, 128, 256]:
                        for w_inner_k in [2, 4, 8]:
                            x = torch.randn((m, k), dtype=dt, device=dev)
                            w = torch.eye(k, dtype=dt, device=dev)

                            y, y_ref = do_y_f16RM_x_f16RM_W_any4TC(
                                x, w, q_group, w_inner_k
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, m, k, q_group, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for m in [7, 16, 32, 33]:
                for w_inner_k in [2, 4, 8]:
                    q_group = w_inner_k * 16
                    for k in range(w_inner_k * 16, 1024, w_inner_k * 16):
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_x_f16RM_W_any4TC(x, w, q_group, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, m, k, q_group, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            for dt in [torch.bfloat16, torch.float16]:
                # m, being non-weight, can be non multiples of the tile size
                for m in [3, 16, 31, 32, 48]:
                    # n however must be a multiple of the tile size
                    for n in [8, 16, 24, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [2, 4, 8]:
                                for q_group in [32, 64, 128, 256]:
                                    x = torch.randn((m, k), dtype=dt, device=dev)
                                    # in order to test equivalency, restrict the int4 quantized
                                    # tensor to [0, 1], so at least this will test positionality
                                    w = torch.randint(
                                        0, 2, (n, k), dtype=dt, device=dev
                                    )

                                    y, y_ref = do_y_f16RM_x_f16RM_W_any4TC(
                                        x, w, q_group, w_inner_k
                                    )

                                    diff = (y_ref.float() - y.float()).abs()
                                    avg_err = diff.sum() / (m * n)
                                    assert avg_err < 1e-1


class Test_y_f16RM_W_any4TC_x_f16RM(unittest.TestCase):
    def setUp(self):
        try:
            import tinygemm
        except ImportError:
            self.skipTest("tinygemm is not installed")

    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [5, 16, 32, 45, 48, 64, 80]:
                for k in [1024, 2048, 4096, 8192]:
                    for q_group in [32, 64, 128, 256]:
                        for w_inner_k in [1, 2, 4]:
                            w = torch.eye(k, dtype=dt, device=dev)
                            x = torch.randn((n, k), dtype=dt, device=dev)

                            y, y_ref = do_y_f16RM_W_any4TC_x_f16RM(
                                x, w, q_group, w_inner_k
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, n, k, q_group, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        for dt in [torch.bfloat16, torch.float16]:
            for n in [7, 16, 32, 33]:
                for w_inner_k in [1, 2, 4]:
                    q_group = w_inner_k * 32
                    for k in range(q_group, 1024, q_group):
                        w = torch.eye(k, dtype=dt, device=dev)
                        x = torch.randn((n, k), dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_W_any4TC_x_f16RM(x, w, q_group, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, n, k, q_group, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            for dt in [torch.bfloat16, torch.float16]:
                # m, being weight, must be a multiple of the tile size
                for m in [16, 32, 48]:
                    # n, being non-weight, can be non multiples of the tile size
                    for n in [3, 16, 31, 32, 48]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [1, 2, 4]:
                                for q_group in [32, 64, 128, 256]:
                                    # in order to test equivalency, restrict the int4 quantized
                                    # tensor to [0, 1], so at least this will test positionality
                                    w = torch.randint(
                                        0, 2, (m, k), dtype=dt, device=dev
                                    )
                                    x = torch.randn((n, k), dtype=dt, device=dev)

                                    y, y_ref = do_y_f16RM_W_any4TC_x_f16RM(
                                        x, w, q_group, w_inner_k
                                    )

                                    diff = (y_ref.float() - y.float()).abs()
                                    avg_err = diff.sum() / (m * n)
                                    assert avg_err < 1e-1
