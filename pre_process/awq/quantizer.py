import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from .utils import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)

import torch
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import numpy as np

def quantize_row(row: np.ndarray, num_clusters: int) -> np.ndarray:
    row_data = row.reshape(-1, 1)
    n_clusters = min(num_clusters, len(row_data))
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(row_data)
    labels = kmeans.predict(row_data)
    return kmeans.cluster_centers_[labels].flatten()

def lut_quantize_rows_parallel(tensor: torch.Tensor, num_clusters: int, n_jobs: int = -1) -> torch.Tensor:
    """
    Applies LUT quantization in parallel to each row of a 2D tensor using K-Means.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (rows, cols).
        num_clusters (int): Number of clusters for each row.
        n_jobs (int): Number of parallel jobs (-1 = use all cores).

    Returns:
        torch.Tensor: Quantized tensor with same shape.
    """
    assert tensor.dim() == 2, "Only 2D tensors are supported"
    device = tensor.device
    dtype = tensor.dtype

    rows = tensor.cpu().numpy()
    quantized_rows = Parallel(n_jobs=n_jobs)(
        delayed(quantize_row)(row, num_clusters) for row in rows
    )

    quantized_tensor = np.stack(quantized_rows, axis=0)
    return torch.tensor(quantized_tensor, dtype=dtype, device=device)

def lut_quantize_rows(tensor: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """
    Applies LUT quantization independently to each row of a 2D tensor using K-Means.

    Args:
        tensor (torch.Tensor): A 2D tensor of shape (rows, cols).
        num_clusters (int): Number of clusters for each row.

    Returns:
        torch.Tensor: Quantized tensor of the same shape.
    """
    assert tensor.dim() == 2, "Only 2D tensors are supported"

    device = tensor.device
    dtype = tensor.dtype

    quantized_rows = []
    matrix = tensor.cpu().numpy()

    for i in tqdm(range(matrix.shape[0]), desc="Rows"):
        row = matrix[i]
        row_data = row.reshape(-1, 1)
        kmeans = KMeans(n_clusters=min(num_clusters, len(row_data)), n_init='auto')
        kmeans.fit(row_data)
        labels = kmeans.predict(row_data)
        quantized_row = kmeans.cluster_centers_[labels].flatten()
        quantized_rows.append(quantized_row)

    quantized_tensor = np.stack(quantized_rows, axis=0)
    return torch.tensor(quantized_tensor, dtype=dtype, device=device)

def any_lut_quantize_tensor(tensor, n_bit=8, n_jobs=-1):
  if n_jobs == 0:
    return lut_quantize_rows(tensor, num_clusters=2**n_bit)
  else:
    return lut_quantize_rows_parallel(tensor, num_clusters=2**n_bit, n_jobs=n_jobs)

def nf4_round(x):
  nf4_lut = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
  return grid_round(x, nf4_lut)

def fp4_round(x):
    fp4_lut = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0,  1.5, -1.5, 2.0, -2.0, 4.0, -4.0]
    return grid_round(x, fp4_lut)

def grid_round(x, value_list):
  # Convert list to tensor for broadcasting
  values = torch.tensor(value_list).to(x)

  # Compute absolute difference: shape (len(x), len(values))
  diff = torch.abs(x.flatten().unsqueeze(1) - values)

  # Get the index of the closest value
  nearest_indices = torch.argmin(diff, dim=1)

  # Use indices to index into original value list
  x_rounded = values[nearest_indices]

  return x_rounded.view(x.shape)

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False, numeric_type="int", **kwargs,
):
    if numeric_type == "int":
        return pseudo_int_quantize_tensor(w, n_bit=n_bit, zero_point=zero_point, q_group_size=q_group_size, inplace=inplace, get_scale_zp=get_scale_zp, **kwargs)
    elif numeric_type == "any":
        assert inplace is False
        return pseudo_any_quantize_tensor(w, n_bit=n_bit, zero_point=zero_point, q_group_size=q_group_size, get_scale_zp=get_scale_zp, **kwargs)
    elif numeric_type == "nf4":
        assert n_bit == 4
        assert zero_point is False
        assert inplace is False
        return pseudo_nf4_quantize_tensor(w, q_group_size=q_group_size, get_scale_zp=get_scale_zp, **kwargs)
    elif numeric_type == "fp4":
        assert n_bit == 4
        assert zero_point is False
        assert inplace is False
        return pseudo_fp4_quantize_tensor(w, q_group_size=q_group_size, get_scale_zp=get_scale_zp, **kwargs)
    else:
        raise ValueError(f"Unsupported numeric_type {numeric_type}.")

def pseudo_int_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False,
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

def pseudo_nf4_quantize_tensor(
    w, q_group_size=-1, get_scale_zp=False, manual=False,
):
    import bitsandbytes
    if manual:
        org_w_shape = w.shape
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        scales = max_val
        w_scaled = w / scales
        w_nf4 = nf4_round(w_scaled)
        w_deq = w_nf4 * scales
        w_deq = w_deq.reshape(org_w_shape)
        state_nf4 = scales
    else:
        w_nf4, state_nf4 = bitsandbytes.functional.quantize_nf4(w, blocksize=q_group_size)
        w_deq = bitsandbytes.functional.dequantize_nf4(w_nf4, quant_state=state_nf4, blocksize=q_group_size)

    if get_scale_zp:
        return w_deq, state_nf4
    else:
        return w_deq
    
def pseudo_fp4_quantize_tensor(
    w, q_group_size=-1, get_scale_zp=False, manual=False,
):
    if manual:
        org_w_shape = w.shape
        if q_group_size > 0:
            assert org_w_shape[-1] % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 4.0
        min_int = -4.0
        scales = max_val / max_int
        w_scaled = w / scales
        w_q = fp4_round(w_scaled)
        w_deq = w_q * scales
        w_deq = w_deq.reshape(org_w_shape)
        state_fp4 = scales
    else:
        import bitsandbytes
        w_fp4, state_fp4 = bitsandbytes.functional.quantize_fp4(w, blocksize=q_group_size)
        w_deq = bitsandbytes.functional.dequantize_fp4(w_fp4, quant_state=state_fp4, blocksize=q_group_size)

    if get_scale_zp:
        return w_deq, state_fp4
    else:
        return w_deq

def pseudo_any_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, get_scale_zp=False,
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        amax_val = w.abs().amax(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val) / (max_int - min_int)
        bias = min_int - min_val / scales
        zeros = bias
        # check
        # xq = x / scales + bias
        # min_val_q = min_val / scales + min_int - min_val / scales = min_int
        # max_val_q = max_val / scales + min_int - min_val / scales = min_int + (max_val - min_val)/scales = min_int + (max_int - min_int) = max_int
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    wscaled = w / scales + zeros
    # TODO: replace round() with LUT
    wq = any_lut_quantize_tensor(wscaled, n_bit=n_bit, n_jobs=-1)
    wdeq = (wq - zeros) * scales
    assert torch.isnan(wdeq).sum() == 0

    w = wdeq.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
