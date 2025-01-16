# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-unsafe

import math

import torch

if torch.version.cuda:
    from mx.elemwise_ops import _quantize_elemwise_core
    from mx.mx_ops import (
        _get_format_params,
        _reshape_to_blocks,
        _shared_exponents,
        FP32_EXPONENT_BIAS,
    )


# Performs row-wise (k/reduction dimension-wise) int4 group
# quantization on the m x k input tensor. Returns a tensor of the same
# size with values quantized to [0, 2^n_bit - 1], along with scale and zero point
# Reconstruction is bf16(int4_value) * scale + zero_point
def group_quantize_tensor(w_orig, n_bit, q_group_size=128):
    w = w_orig.float()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0

    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.int32).reshape(w.shape)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out, scales_and_zeros.to(w_orig.dtype)


# Produces a float32 matrix containing `q` rounded to the nearest mx4 value
# and the assoicated group exponent scale
def round_to_mx4(x, q_group_size):
    if x.numel() <= 0 or torch.isnan(x).any():
        return x

    # do all arithmetic in float
    x = x.float()

    shared_exp_axes = [-1]
    x_grouped, axes, orig_shape, padded_shape = _reshape_to_blocks(
        x, axes=shared_exp_axes, block_size=q_group_size
    )

    x_exponents = _shared_exponents(x_grouped, method="max", axes=shared_exp_axes)

    # Flush subnormal FP32 inputs to zero
    x_grouped = x_grouped * (x_exponents > -FP32_EXPONENT_BIAS).type(x_grouped.dtype)
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp4_e2m1")

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    x_exponents = x_exponents - emax
    scale_bits = 8  # E8M0 for the shared exponents
    scale_emax = 2 ** (scale_bits - 1) - 1
    if (x_exponents > scale_emax).any():
        print(f"{x_exponents.max()=} {emax=} {scale_emax=} ")
    x_exponents[x_exponents > scale_emax] = float("NaN")
    x_exponents[x_exponents < -scale_emax] = -scale_emax

    x_q = _quantize_elemwise_core(
        x_grouped / (2**x_exponents),
        mbits,
        ebits,
        max_norm,
        round="nearest",
        allow_denorm=True,
        saturate_normals=True,
        custom_cuda=False,
    )

    dequantized_x_q = x_q * (2**x_exponents)

    if torch.isnan(dequantized_x_q).any():
        raise RuntimeError(
            f"NaN encountered with {x.abs().max()=} {x.abs().min()=} {torch.isnan(x_exponents).any()} {torch.isnan(x).any()=} {torch.isnan(x_q).any()=} {torch.isnan(x_exponents).any()=} {torch.isnan(x_exponents).any()=}"
        )

    # Reshape as matrices
    return x_q.reshape(x.size()), x_exponents.reshape(
        x_exponents.size(0), x_exponents.size(1)
    )


def quantize_mx4(x, q_group_size):
    # Round the values to their nearest mx4 representation
    x_q, x_e = round_to_mx4(x, q_group_size)

    assert x_q.dtype == torch.float32
    assert x_e.dtype == torch.float32

    # Convert the rounded floating point values in x_q to a uint4 index representing
    # which of the 16 mx4 values is contained within

    mx4_values_quantize = [
        # should be 0.0 but we handle +/-0 separately
        math.nan,  # 0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        # should be 0.0 but we handle +/-0 separately
        math.nan,  # -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    # The quantization function rounds to the nearest mx4 value, still represented
    # as a float32. We instead need to pack this into a uint32 tensor
    # FIXME: tinygemm tensor core packing takes int32, not int8 currently
    q = torch.full(x_q.size(), -128, dtype=torch.int32, device=x_q.device)
    for v, i in zip(mx4_values_quantize, range(16)):
        q = torch.where(x_q == v, i, q)

    # As 0.0 == -0.0, we need to handle zeros separately
    x_q_cast = x_q.view(dtype=torch.uint32)

    # handle +0.0
    q = torch.where(x_q_cast == 0, 0, q)

    # handle -0.0
    q = torch.where(x_q_cast == 2147483648, 8, q)

    # All values should have been quantized to one of the mx4_values
    assert (q == -128).sum() == 0

    # 0 exponent is index 127 (mx4 e8 range is [-127, 127] + NaN)
    assert (x_e > 128).sum() == 0
    e_int = (x_e + 127).to(torch.uint8)

    return q, e_int


def dequantize_mx4(q, e):
    num_groups = e.size(1)
    assert q.size(1) % num_groups == 0
    assert q.size(1) // num_groups > 0
    q_group_size = q.size(1) // num_groups

    # Dequantize int4 to mx4
    mx4_values_dequantize = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    mx4 = torch.full(q.size(), -128.0, dtype=torch.float, device=q.device)
    for v, i in zip(mx4_values_dequantize, range(16)):
        mx4 = torch.where(q == i, v, mx4)

    mx4_grouped = mx4.reshape(mx4.size(0), num_groups, q_group_size)

    e = e.float() - 127
    e_grouped = e.reshape(e.size(0), num_groups, 1).expand(
        e.size(0), num_groups, q_group_size
    )

    out_grouped = mx4_grouped * (2**e_grouped)
    return out_grouped.reshape(q.size())
