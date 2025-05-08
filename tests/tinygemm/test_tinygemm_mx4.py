# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
import torch

import tinygemm
from tinygemm.utils import quantize_mx4


def do_y_f16TC_x_f16TC_W_mx4TC(x, w, w_inner_k, x_inner_k, exp_scale=False):
    y_ref = x @ w.t()
    q_group_size = 32

    w_q, w_e = quantize_mx4(w, q_group_size)

    if exp_scale:
        for row in range(w.size(0)):
            # vary the scale per each row, so we test that we are getting
            # the right row-wise exponent
            scale = row % 4
            # weight rows become output columns
            y_ref[:, row] *= 2**scale
            w_e[row, :] += scale

    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_q, w_inner_k)

    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_mx4TC(
        x2, w2, q_group_size, w_e, True
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
        y2, x.size(0), w.size(0)
    )

    return y, y_ref


def do_y_f16TC_W_mx4TC_x_f16TC(x, w, w_inner_k, x_inner_k, exp_scale=False):
    y_ref = (w @ x.t()).t()
    q_group_size = 32

    w_q, w_e = quantize_mx4(w, q_group_size)

    if exp_scale:
        for row in range(w.size(0)):
            # vary the scale per each row, so we test that we are getting
            # the right row-wise exponent
            scale = row % 4
            # weight rows become output columns
            y_ref[:, row] *= 2**scale
            w_e[row, :] += scale

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_q, w_inner_k)
    x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_B_layout(x, x_inner_k)
    y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_mx4TC(
        w2, x2, q_group_size, w_e, False
    )
    y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_B_layout(
        y2, x.size(0), w.size(0)
    )

    return y, y_ref


def do_y_f16RM_x_f16RM_W_mx4TC(x, w, w_inner_k, exp_scale=0):
    y_ref = x @ w.t()
    y_ref *= 2**exp_scale
    q_group_size = 32

    w_q, w_e = quantize_mx4(w, q_group_size)
    w_e += exp_scale

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_q, w_inner_k)
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_mx4TC(
        x, w2, q_group_size, w_e, True
    )

    return y, y_ref


def do_y_f16RM_W_mx4TC_x_f16RM(x, w, w_inner_k, exp_scale=0):
    y_ref = (w @ x.t()).t()
    y_ref *= 2**exp_scale
    q_group_size = 32

    w_q, w_e = quantize_mx4(w, q_group_size)
    w_e += exp_scale

    w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Aint4_layout(w_q, w_inner_k)
    y = torch.ops.tinygemm.tinygemm_y_f16RM_x_f16RM_w_mx4TC(
        w2, x, q_group_size, w_e, False
    )

    return y, y_ref


class Test_y_f16TC_x_f16TC_W_mx4TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for m in [5, 16, 19, 32, 48]:
                for w_inner_k in [2, 4, 8]:
                    for k in range(w_inner_k * 16, 1024, w_inner_k * 16):
                        # We insert a test here to make sure that the row-wise
                        # MX4 exponent scale works by varying it
                        # (we need only do this once each for weight on right
                        # and weight on left)
                        for exp_scale in [False, True]:
                            x = torch.randn((m, k), dtype=dt, device=dev)
                            w = torch.eye(k, dtype=dt, device=dev)

                            y, y_ref = do_y_f16TC_x_f16TC_W_mx4TC(
                                x, w, w_inner_k, 1, exp_scale
                            )

                            assert torch.equal(y_ref, y)

    def test_general_k(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for m in [5, 16, 19]:
                for w_inner_k in [2, 4, 8]:
                    for k in range(w_inner_k * 16, 1024, w_inner_k * 16):
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)

                        y, y_ref = do_y_f16TC_x_f16TC_W_mx4TC(x, w, w_inner_k, 1)

                        assert torch.equal(y_ref, y)

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for _ in range(5):
                # m, being non-weight, can be non multiples of the tile size
                for m in [3, 16, 31, 32, 48]:
                    # n however must be a multiple of the tile size
                    for n in [8, 16, 24, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [2, 4, 8]:
                                x = torch.randn((m, k), dtype=dt, device=dev)
                                # in order to test equivalency, restrict the int4 quantized
                                # tensor to [0, 1], so at least this will test positionality
                                w = torch.randint(0, 2, (n, k), dtype=dt, device=dev)

                                y, y_ref = do_y_f16TC_x_f16TC_W_mx4TC(
                                    x, w, w_inner_k, 1
                                )

                                diff = (y_ref.float() - y.float()).abs()
                                avg_err = diff.sum() / (m * n)

                                assert avg_err < 1e-1


class Test_y_f16TC_W_mx4TC_x_f16TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for n in [5, 16, 19, 32, 48]:
                for k in [256, 1024, 2048, 4096, 8192]:
                    for w_inner_k in [1, 2, 4]:
                        for x_inner_k in [1, 2]:
                            # We insert a test here to make sure that the row-wise
                            # MX4 exponent scale works by varying it
                            # (we need only do this once each for weight on right
                            # and weight on left)
                            for exp_scale in [False, True]:
                                w = torch.eye(k, dtype=dt, device=dev)
                                x = torch.randn((n, k), dtype=dt, device=dev)

                                y, y_ref = do_y_f16TC_W_mx4TC_x_f16TC(
                                    x, w, w_inner_k, x_inner_k, exp_scale
                                )

                                # FIXME: why are there minor differences with float16?
                                if dt == torch.bfloat16:
                                    assert torch.equal(y_ref, y)
                                else:
                                    max_err = (y - y_ref).abs().max()
                                    if not max_err < 1e-4:
                                        print(dt, n, k, w_inner_k, max_err)
                                        assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for n in [5, 16, 19, 32, 48]:
                for w_inner_k in [1, 2, 4]:
                    for x_inner_k in [1, 2]:
                        for k in range(w_inner_k * 32, 1024, w_inner_k * 32):
                            w = torch.eye(k, dtype=dt, device=dev)
                            x = torch.randn((n, k), dtype=dt, device=dev)

                            y, y_ref = do_y_f16TC_W_mx4TC_x_f16TC(
                                x, w, w_inner_k, x_inner_k
                            )

                            # FIXME: why are there minor differences with float16?
                            if dt == torch.bfloat16:
                                assert torch.equal(y_ref, y)
                            else:
                                max_err = (y - y_ref).abs().max()
                                if not max_err < 1e-4:
                                    print(dt, n, k, w_inner_k, max_err)
                                    assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
            # range necessary for the 8-bit MX group scaling exponents
            for dt in [torch.bfloat16]:
                # m must be a multiple of the tile size
                for m in [16, 32, 48]:
                    # n, being non-weight, can be non multiples of the tile size
                    for n in [3, 8, 11, 16, 31, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [1, 2, 4]:
                                for x_inner_k in [1, 2]:
                                    # in order to test equivalency, restrict the mx4 quantized
                                    # tensor to [0, 1], so at least this will test positionality
                                    w = torch.randint(
                                        0, 2, (m, k), dtype=dt, device=dev
                                    )
                                    x = torch.randn((n, k), dtype=dt, device=dev)

                                    y, y_ref = do_y_f16TC_W_mx4TC_x_f16TC(
                                        x, w, w_inner_k, x_inner_k
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
                                            avg_err,
                                        )
                                        assert avg_err < 1e-1


class Test_y_f16RM_x_f16RM_W_mx4TC(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for m in [5, 16, 32, 45, 48, 64, 80]:
                for k in [1024, 2048, 4096, 8192]:
                    for w_inner_k in [2, 4, 8]:
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_x_f16RM_W_mx4TC(x, w, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, m, k, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for m in [7, 16, 32, 33]:
                for w_inner_k in [2, 4, 8]:
                    for k in range(w_inner_k * 16, 1024, w_inner_k * 16):
                        x = torch.randn((m, k), dtype=dt, device=dev)
                        w = torch.eye(k, dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_x_f16RM_W_mx4TC(x, w, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, m, k, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
            # range necessary for the 8-bit MX group scaling exponents
            for dt in [torch.bfloat16]:
                # m, being non-weight, can be non multiples of the tile size
                for m in [3, 16, 31, 32, 48]:
                    # n however must be a multiple of the tile size
                    for n in [8, 16, 24, 32]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [2, 4, 8]:
                                x = torch.randn((m, k), dtype=dt, device=dev)
                                # in order to test equivalency, restrict the mx4 quantized
                                # tensor to [0, 1], so at least this will test positionality
                                w = torch.randint(0, 2, (n, k), dtype=dt, device=dev)

                                y, y_ref = do_y_f16RM_x_f16RM_W_mx4TC(x, w, w_inner_k)

                                diff = (y_ref.float() - y.float()).abs()
                                avg_err = diff.sum() / (m * n)
                                assert avg_err < 1e-1


class Test_y_f16RM_W_mx4TC_x_f16RM(unittest.TestCase):
    def test_identity_mul(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for n in [5, 16, 32, 45, 48, 64, 80]:
                for k in [1024, 2048, 4096, 8192]:
                    for w_inner_k in [1, 2, 4]:
                        w = torch.eye(k, dtype=dt, device=dev)
                        x = torch.randn((n, k), dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_W_mx4TC_x_f16RM(x, w, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, n, k, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_k(self):
        dev = torch.device("cuda:0")

        # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
        # range necessary for the 8-bit MX group scaling exponents
        for dt in [torch.bfloat16]:
            for n in [7, 16, 32, 33]:
                for w_inner_k in [1, 2, 4]:
                    for k in range(w_inner_k * 32, 1024, w_inner_k * 32):
                        w = torch.eye(k, dtype=dt, device=dev)
                        x = torch.randn((n, k), dtype=dt, device=dev)

                        y, y_ref = do_y_f16RM_W_mx4TC_x_f16RM(x, w, w_inner_k)

                        # FIXME: why are there minor differences with float16?
                        if dt == torch.bfloat16:
                            assert torch.equal(y_ref, y)
                        else:
                            max_err = (y - y_ref).abs().max()
                            if not max_err < 1e-4:
                                print(dt, n, k, w_inner_k, max_err)
                                assert max_err < 1e-4

    def test_general_mul(self):
        dev = torch.device("cuda:0")

        for _ in range(5):
            # Equivalent to the int4/any4 test, but only bfloat16 has the exponent
            # range necessary for the 8-bit MX group scaling exponents
            for dt in [torch.bfloat16]:
                # m, being weight, must be a multiple of the tile size
                for m in [16, 32, 48]:
                    # n, being non-weight, can be non multiples of the tile size
                    for n in [3, 16, 31, 32, 48]:
                        # limit the reduction size to test equivalency
                        for k in [1024, 2048]:
                            for w_inner_k in [1, 2, 4]:
                                # in order to test equivalency, restrict the mx4 quantized
                                # tensor to [0, 1], so at least this will test positionality
                                w = torch.randint(0, 2, (m, k), dtype=dt, device=dev)
                                x = torch.randn((n, k), dtype=dt, device=dev)

                                y, y_ref = do_y_f16RM_W_mx4TC_x_f16RM(x, w, w_inner_k)

                                diff = (y_ref.float() - y.float()).abs()
                                avg_err = diff.sum() / (m * n)
                                assert avg_err < 1e-1


# Test MX exponents mapping to NaN
# (not really needed because we do not support re-quantization of outputs,
# and all weights are checked before quantization, but it's here)
class Test_NaN_exponent(unittest.TestCase):
    def test_nan(self):
        dev = torch.device("cuda:0")

        # Only bfloat16 has the exponent range necessary for 8-bit MX scaling
        # exponents
        dt = torch.bfloat16
        m = 5
        w_inner_k = 2
        x_inner_k = 1
        k = 32
        q_group_size = 32

        x = torch.randn((m, k), dtype=dt, device=dev)
        w = torch.eye(k, dtype=dt, device=dev)

        y_ref = x @ w.t()

        w_q, w_e = quantize_mx4(w, q_group_size)

        x2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_A_layout(x, x_inner_k)
        w2 = torch.ops.tinygemm.convert_matrix_to_m16n8k16_Bint4_layout(w_q, w_inner_k)

        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_mx4TC(
            x2, w2, q_group_size, w_e, True
        )
        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
            y2, x.size(0), w.size(0)
        )

        # Validate that the baseline MM works
        assert torch.equal(y_ref, y)

        assert not torch.isnan(y).any()

        # Introduce a MX4 exponent right below NaN
        w_e[0][0] = 254

        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_mx4TC(
            x2, w2, q_group_size, w_e, True
        )
        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
            y2, x.size(0), w.size(0)
        )

        assert not torch.isnan(y).any()

        # Introduce a MX4 exponent that maps to NaN
        w_e[0][0] = 255

        y2 = torch.ops.tinygemm.tinygemm_y_f16TC_x_f16TC_w_mx4TC(
            x2, w2, q_group_size, w_e, True
        )
        y = torch.ops.tinygemm.convert_matrix_from_m16n8k16_A_layout(
            y2, x.size(0), w.size(0)
        )

        assert torch.isnan(y).any()
