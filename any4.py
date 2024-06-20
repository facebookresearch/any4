from typing import Callable, List, Tuple, Type
import time
import torch
from joblib import Parallel, delayed
import numpy as np
import faiss
from sklearn.cluster import KMeans


def count_layer_type(model, layer_type=torch.nn.Linear, count=0):
    for _, module in model._modules.items():
        if isinstance(module, layer_type):
            count += 1
        
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(module, layer_type, 0)
    return count 


def convert(model: torch.nn.Module, layer_from: Type, layer_to: Callable, **kwargs):
    index = 0
    for name, module in model.named_modules():
        if isinstance(module, (layer_from)):
            print(f"\t{name}", end="", flush=True)
            # TODO: do this in a cleaner way
            # if "mlp" not in name:
            #     print("...Skip")
            #     continue
            layer_to(module, **kwargs)
            print("... Done", flush=True)
            index += 1
            if index == 6e6:
                break

    return model


# Performs row-wise (k/reduction dimension-wise) int4 group
# quantization on the m x k input tensor. Returns a tensor of the same
# size with values quantized to [0, 2^n_bit - 1], along with scale and zero point
# Reconstruction is bf16(int4_value) * scale + min_val
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


def apply_q_groups(w_orig, n_bit, q_group_size=128):
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

    w_new = to_quant.sub(min_val).div(scales).reshape(w_orig.size())
    w_new_zeros = torch.zeros(to_quant.size(), dtype=to_quant.dtype, device=to_quant.device).sub(min_val).div(scales).reshape(w_orig.size())

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

    return w_new, w_new_zeros, scales_and_zeros



# takes the scale or offset per each quantization group and reshapes/duplicates it to map
# to the original matrix size
def expand_q_groups(x, orig_size, q_group_size):
    out = x.reshape(orig_size[0], orig_size[1] // q_group_size, 1)
    out = out.expand(orig_size[0], orig_size[1] // q_group_size, q_group_size)
    return out.contiguous().view(orig_size)


# performs quantization and dequantization under N-bit grouped integer quantization
# (i.e., returns the effective result of the quantization algorithm)
def reconstruct_intN_grouped(x, n_bit = 4, q_group_size=128, parallelize=True):
    n_bit = 4
    int4, _, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size)
    int4.round_().clamp_(0, (2 ** n_bit) - 1).sub_(8)

    assert int4.size(1) == q_group_size * scales_and_zeros.size(0)

    if parallelize:
        scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
        zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

        scales = expand_q_groups(scales, x.size(), q_group_size)
        zeros = expand_q_groups(zeros, x.size(), q_group_size)

        reconstructed = int4 * scales + zeros
    else:
        reconstructed = torch.zeros_like(x)

        for r in range(x.size(0)):
            for c in range(x.size(1)):
                q_group = c // q_group_size
                reconstructed[r][c] = int4[r][c] * scales_and_zeros[q_group][r][0] + scales_and_zeros[q_group][r][1]

    return reconstructed


def intq(module: torch.nn.Module, n_bit: int = 4, group_size: int = 128, transpose=True):
    w = module.weight.clone()

    if transpose:
        w = w.t()

    w_deq = reconstruct_intN_grouped(w, n_bit=n_bit, q_group_size=group_size)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module


def cluster_matrix(x, n_bit=4, bias_pow=1.0, parallelize=True):
    if bias_pow != 1.0:
        # k-means should be roughly zero centered, since we should bias larger magnitude (negative or positive) values
        # for greater representation.
        # Values are in the range [0, 15] so subtract (15 - 0) / 2 = 7.5 to approximately zero center the data
        #
        # Note that there is no guarantee that each q-group is itself zero centered (there can be a "DC bias")
        # but note that across all q-groups, values closer to 0 and closer to 15 are extremal values
        x = x - ((2 ** n_bit) - 1) / 2. 
        # give more weight to extremal values by considering the signed square
        x = (x.abs() ** bias_pow) * torch.sign(x)

    if parallelize:
        assign, any4, assign_val = cluster_rows_parallel(x)
    else:
        assign, any4, assign_val = cluster_rows(x)

    if bias_pow != 1.0:
        # undo the pow

        any4 = (any4.abs() ** (1. / bias_pow)) * torch.sign(any4)

        # map values back to [0, 15]
        any4 = any4 + ((2 ** n_bit) - 1) / 2.

    return assign, any4, assign_val


def cluster_row(r, n_bit=4):
    clusters = KMeans(n_clusters=2**n_bit, random_state=0, n_init="auto").fit(r)
    any4 = torch.from_numpy(clusters.cluster_centers_).reshape(2**n_bit)
    assign = torch.from_numpy(clusters.labels_)
    assign_val = torch.from_numpy(clusters.cluster_centers_[clusters.predict(r)]).flatten()

    return assign, any4, assign_val

def cluster_rows(x, n_bit=4):
    assign = torch.zeros(x.size(), dtype=torch.int32, device=x.device)
    any4 = torch.zeros((x.size(0), 2**n_bit), dtype=x.dtype, device=x.device)
    assign_val = torch.zeros(x.size(), dtype=torch.int32, device=x.device)

    for row in range(x.size(0)):
        r = x[row].reshape(x.size(1), 1).cpu().numpy()
        any4[row], assign[row], assign_val[row] = cluster_row(r, n_bit)

    return assign, any4, assign_val


def cluster_rows_parallel(x, n_bit=4):
    x_np = x.cpu().detach().numpy()
    start = time.time()
    results: List = Parallel(n_jobs=-1, pre_dispatch="n_jobs//2")(delayed(cluster_row)(r.reshape(-1, 1), n_bit) for r in x_np)
    print(f"... {time.time() - start:.2f} s ", end="", flush=True)
    # Transpose the list of tuples to a tuple of lists
    results_transposed = tuple(zip(*results))
    # Convert each item in the tuple (which are tuples) to lists
    results_transposed = tuple(list(item) for item in results_transposed)

    # Unpack into different matrices
    assign = torch.stack(results_transposed[0], dim=0).contiguous().to(x.device)
    any4 = torch.stack(results_transposed[1], dim=0).contiguous().to(x.device)
    assign_val = torch.stack(results_transposed[2], dim=0).contiguous().to(x.device)

    return assign, any4, assign_val

def quantize_to_any4(x, q_group_size=128, n_bit = 4, bias_pow=1.0):
    to_cluster, to_cluster_group_zero_point, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size)
    assign, any4, assign_val = cluster_matrix(to_cluster, n_bit=n_bit, bias_pow=bias_pow)

    # any4 above is roughly in the range [0+eps, 15+eps], but dequant expects [-8+eps, 7+eps]
    # so adjust for usage
    any4 = any4 - 2.0 ** (n_bit - 1)
    any4 = any4.to(dtype=x.dtype)
    print(any4)

    return assign, any4.to(dtype=x.dtype), scales_and_zeros.to(dtype=x.dtype)


# performs quantization and dequantization under any4 scalar k-means grouped integer quantization
# (i.e., returns the effective result of the quantization algorithm)
def reconstruct_any4_grouped(x, n_bit=4, q_group_size=128, bias_pow=1.0, parallelize=True):
    to_cluster, _, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size)

    assign, any4, assign_val = cluster_matrix(to_cluster, n_bit=n_bit, bias_pow=bias_pow)
    any4.sub_(2**(n_bit - 1))
    assign_val.sub_(2**(n_bit - 1))

    if parallelize:
        scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
        zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

        scales = expand_q_groups(scales, x.size(), q_group_size)
        zeros = expand_q_groups(zeros, x.size(), q_group_size)

        reconstructed = assign_val * scales + zeros
    else:
        reconstructed = torch.zeros_like(x)
        for r in range(x.size(0)):
            for c in range(x.size(1)):
                q_group = c // q_group_size
                reconstructed[r][c] = (any4[r][assign[r][c]]) * scales_and_zeros[q_group][r][0] + scales_and_zeros[q_group][r][1]

    return reconstructed


def anyq(module: torch.nn.Module, n_bit: int = 4, group_size: int = 128, bias_pow=1.0, transpose=False):
    w = module.weight.clone()
    if transpose:
        w = w.t()

    w_deq = reconstruct_any4_grouped(w, n_bit=n_bit, q_group_size=group_size, bias_pow=bias_pow)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

