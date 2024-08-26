from typing import Callable, Dict, List, Tuple, Type
import time
import torch
from joblib import Parallel, delayed
import numpy as np
import sklearn.cluster

import bitsandbytes as bnb

import kmeans
import gc

def count_layer_type(model, layer_type=torch.nn.Linear, count=0):
    for _, module in model._modules.items():
        if isinstance(module, layer_type):
            count += 1
        
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(module, layer_type, 0)
    return count 


def convert(model: torch.nn.Module, layer_from: Type, layer_to: Callable, skip_modules=[], tokenizer=None, calibrate_args={}, **kwargs):
    index = 0

    calibrate_fn = None
    if "sample_weight" in kwargs:
        if isinstance(kwargs["sample_weight"], Callable):
            calibrate_fn = kwargs["sample_weight"]

    for name, module in model.named_modules():
        if isinstance(module, (layer_from)):
            print(f"\t{name}", end="", flush=True)
            if name in skip_modules:
                print("Skip")
                continue

            # Calibrate if necessary
            if calibrate_fn is not None:
                calibrate_args["seed"] = index
                kwargs["sample_weight"] = calibrate_fn(model=model, tokenizer=tokenizer, **calibrate_args)

            layer_to(module, name=name, **kwargs)
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


def apply_q_groups(w_orig, n_bit, q_group_size=128, scale_only=False):
    w = w_orig.float()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    if scale_only:
        # TODO: this wastes one bit. Review with Jeff.
        absmax_val = to_quant.abs().amax(dim=1, keepdim=True)
        absmax_int = 2**(n_bit - 1) - 1
        scales = absmax_val.clamp(min=1e-6) / absmax_int
        zeros = torch.zeros_like(scales)

        w_new = to_quant.div(scales).reshape(w_orig.size())
        w_new_zeros = torch.zeros_like(w_new)
    else:
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-6) / (max_int - min_int)
        zeros = min_val + scales * (2 ** (n_bit - 1))

        w_new = to_quant.sub(min_val).div(scales).reshape(w_orig.size())
        w_new_zeros = torch.zeros(to_quant.size(), dtype=to_quant.dtype, device=to_quant.device).sub(min_val).div(scales).reshape(w_orig.size())


    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(zeros).sum() == 0

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
def reconstruct_intN_grouped(x, n_bit = 4, q_group_size=128, parallelize=True, scale_only=False, **kwargs):
    int4, _, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size, scale_only=scale_only)
    int4.round_().clamp_(0, (2 ** n_bit) - 1).sub_(2**(n_bit - 1))

    assert int4.size(1) == q_group_size * scales_and_zeros.size(0)

    if parallelize:
        scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
        zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

        scales = expand_q_groups(scales, x.size(), q_group_size)
        zeros = expand_q_groups(zeros, x.size(), q_group_size)

        reconstructed = int4 * scales + zeros
        reconstructed = reconstructed.to(dtype=x.dtype)
    else:
        reconstructed = torch.zeros_like(x)

        for r in range(x.size(0)):
            for c in range(x.size(1)):
                q_group = c // q_group_size
                reconstructed[r][c] = int4[r][c] * scales_and_zeros[q_group][r][0] + scales_and_zeros[q_group][r][1]

    return reconstructed


def intq(module: torch.nn.Module, n_bit: int = 4, group_size: int = 128, transpose=False, **kwargs):
    w = module.weight

    if transpose:
        w = w.t()

    w_deq = reconstruct_intN_grouped(w, n_bit=n_bit, q_group_size=group_size, **kwargs)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

def cluster_row_custom(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, **kwargs):
    init = kmeans.build_init(x=r, n_clusters=2 ** n_bit, init_type=init)
    if init is None:
        init = "k-means++"
    sample_weight = kmeans.build_sample_weight(x=r, sample_weight_type=sample_weight)
    assign_val, any4, assign = kmeans.kmeans(r, n_clusters=2**n_bit, init=init, sample_weight=sample_weight, X_surrogate=r_surrogate, **kwargs)

    any4 = torch.from_numpy(any4).reshape(2**n_bit)
    assign = torch.from_numpy(assign).flatten()
    assign_val = torch.from_numpy(assign_val).flatten()

    return assign, any4, assign_val

def cluster_row_scikit(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, **kwargs):
    assert r_surrogate==None, "scikit clustering does not support surrogate_to_cluster"
    init = kmeans.build_init(x=r, n_clusters=2 ** n_bit, init_type=init)
    if init is None:
        init = "k-means++"
    sample_weight = kmeans.build_sample_weight(x=r, sample_weight_type=sample_weight)

    clusters = sklearn.cluster.KMeans(n_clusters=2**n_bit, init=init, random_state=0, n_init="auto", **kwargs).fit(r, sample_weight=sample_weight)
    any4 = torch.from_numpy(clusters.cluster_centers_).reshape(2**n_bit)
    assign = torch.from_numpy(clusters.labels_)
    assign_val = torch.from_numpy(clusters.cluster_centers_[clusters.predict(r)]).flatten()

    return assign, any4, assign_val

def cluster_row_agglomerative(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, **kwargs):
    assert r_surrogate==None, "scikit clustering does not support surrogate_to_cluster"
    assert init==None, "agglomerative clustering does not support init"

    clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=2**n_bit, **kwargs).fit(r)
    assign = np.array(clusters.labels_)
    any4 = np.array([np.average(r[assign == label].flatten(), weights=sample_weight[assign == label] if sample_weight is not None else None) for label in np.unique(assign)])
    assign_val = any4[assign]

    return torch.from_numpy(assign), torch.from_numpy(any4), torch.from_numpy(assign_val)

# TODO: change arg name x_cluster to x_surrogate
def cluster_matrix(x, n_bit=4, bias_pow=1.0, keep_outliers=False, cluster_row: Callable = cluster_row_scikit, init=None, sample_weight=None, parallelize=True, x_cluster=None, **kwargs):
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

    start = time.time()
    to_cluster = x.cpu().detach().numpy()
    surrogate_to_cluster = x_cluster.cpu().float().detach().numpy() if x_cluster is not None else None
    sample_weight = sample_weight.float().cpu().detach().numpy() if sample_weight is not None else None
    if parallelize:
        assign, any4, assign_val = cluster_rows_parallel(to_cluster, cluster_row=cluster_row, n_bit=n_bit, init=init, sample_weight=sample_weight, x_surrogate=surrogate_to_cluster, **kwargs)
    else:
        assign, any4, assign_val = cluster_rows(to_cluster, cluster_row=cluster_row, init=init, n_bit=n_bit, sample_weight=sample_weight, x_surrogate=surrogate_to_cluster, **kwargs)
    assign = assign.to(x.device)
    any4 = any4.to(x.device)
    assign_val = assign_val.to(x.device)
    print(f"... {time.time() - start:.2f} s ", end="", flush=True)


    if keep_outliers:
        max_outliers = torch.amax(x, dim=1, keepdim=True)
        min_outliers = torch.amin(x, dim=1, keepdim=True)

        any4.scatter_(dim=1, index=torch.argmax(any4, dim=1, keepdim=True), src=max_outliers)
        any4.scatter_(dim=1, index=torch.argmin(any4, dim=1, keepdim=True), src=min_outliers)

        assign_val.scatter_(dim=1, index=torch.argmax(assign_val, dim=1, keepdim=True), src=max_outliers)
        assign_val.scatter_(dim=1, index=torch.argmin(assign_val, dim=1, keepdim=True), src=min_outliers)

    if bias_pow != 1.0:
        # undo the pow

        any4 = (any4.abs() ** (1. / bias_pow)) * torch.sign(any4)
        assign_val = (assign_val.abs() ** (1. / bias_pow)) * torch.sign(assign_val)

        # map values back to [0, 15]
        any4 = any4 + ((2 ** n_bit) - 1) / 2.
        assign_val = assign_val + ((2 ** n_bit) - 1) / 2.

    # Clean up memory
    del to_cluster
    if surrogate_to_cluster is not None:
        del surrogate_to_cluster
    if sample_weight is not None:
        del sample_weight

    return assign, any4, assign_val

def get_sample_weight(sample_weight, index):
    if sample_weight is None:
        return None
    elif np.squeeze(sample_weight).ndim == 1:
        return sample_weight
    elif np.squeeze(sample_weight).ndim == 2:
        return sample_weight[index]

def cluster_rows(x, cluster_row: Callable = cluster_row_scikit, n_bit=4, x_surrogate=None, sample_weight=None, **kwargs):
    assign = torch.zeros(x.shape, dtype=torch.int32)
    any4 = torch.zeros((x.shape[0], 2**n_bit))
    assign_val = torch.zeros(x.shape)

    for row in range(x.shape[0]):
        r = x[row].reshape(x.shape[1], 1)
        if x_surrogate is not None:
            r_surrogate = x_surrogate[row].reshape(x_surrogate.shape[1], 1)
            assign[row], any4[row], assign_val[row] = cluster_row(r, n_bit, sample_weight=get_sample_weight(sample_weight, row), r_surrogate=r_surrogate,**kwargs)
        else:
            assign[row], any4[row], assign_val[row] = cluster_row(r, n_bit, sample_weight=get_sample_weight(sample_weight, row), **kwargs)

    return assign, any4, assign_val


def cluster_rows_parallel(x, cluster_row: Callable = cluster_row_scikit, x_surrogate=None, sample_weight=None, **kwargs):
    if x_surrogate is None:
        results: List = Parallel(n_jobs=-1, pre_dispatch="n_jobs//2")(delayed(cluster_row)(x[row].reshape(-1, 1), sample_weight=get_sample_weight(sample_weight, row), **kwargs) for row in range(x.shape[0]))
    else:
        results: List = Parallel(n_jobs=-1, pre_dispatch="n_jobs//2")(delayed(cluster_row)(x[row].reshape(-1, 1), sample_weight=get_sample_weight(sample_weight, row), r_surrogate=x_surrogate[row].reshape(-1, 1), **kwargs) for row in range(x.shape[0]))
    # Transpose the list of tuples to a tuple of lists
    results_transposed = tuple(zip(*results))
    # Convert each item in the tuple (which are tuples) to lists
    results_transposed = tuple(list(item) for item in results_transposed)

    # Unpack into different matrices
    assign = torch.stack(results_transposed[0], dim=0).contiguous()
    any4 = torch.stack(results_transposed[1], dim=0).contiguous()
    assign_val = torch.stack(results_transposed[2], dim=0).contiguous()

    return assign, any4, assign_val

# TODO: this needs to be revisited to verify that it is in sync with reconstruct_any4_grouped
def quantize_to_any4(x, q_group_size=128, n_bit = 4, scale_only=False, bias_pow=1.0, keep_outliers=False, cluster_row: Callable = cluster_row_scikit, init=None, sample_weight=None, surrogate_cluster=False, **kwargs):
    if q_group_size:
        to_cluster, to_cluster_group_zero_point, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size, scale_only=scale_only)
    else:
        to_cluster = x.float()

    if surrogate_cluster:
        surrogate_to_cluster = x
    else:
        surrogate_to_cluster = None

    assign, any4, assign_val = cluster_matrix(to_cluster, n_bit=n_bit, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row, init=init, sample_weight=sample_weight, x_cluster=surrogate_to_cluster, **kwargs)

    if q_group_size:
        # any4 above is roughly in the range [0+eps, 15+eps], but dequant expects [-8+eps, 7+eps]
        # so adjust for usage
        any4 = any4 - 2.0 ** (n_bit - 1)
        any4 = any4.to(dtype=x.dtype)
    print(any4)

    return assign, any4.to(dtype=x.dtype), scales_and_zeros.to(dtype=x.dtype)


# performs quantization and dequantization under any4 scalar k-means grouped integer quantization
# (i.e., returns the effective result of the quantization algorithm)
def reconstruct_any4_grouped(x, n_bit=4, q_group_size=128, scale_only=False, bias_pow=1.0, keep_outliers=False, cluster_row: Callable = cluster_row_scikit, init=None, sample_weight=None, parallelize=True, surrogate_cluster=False, **kwargs):
    if q_group_size:
        to_cluster, _, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size=q_group_size, scale_only=scale_only)

        scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
        zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

        scales = expand_q_groups(scales, x.size(), q_group_size)
        zeros = expand_q_groups(zeros, x.size(), q_group_size)

        if sample_weight is not None:
            # TODO: add options here to apply absolute() as well as scaling to sample weights
            sample_weight = sample_weight.to(scales.device) * scales

        del scales_and_zeros
    else:
        to_cluster = x.float()

    to_cluster = to_cluster.contiguous()
    if sample_weight is not None:
        sample_weight = sample_weight.contiguous()

    if surrogate_cluster:
        surrogate_to_cluster = x
    else:
        surrogate_to_cluster = None

    assign, any4, assign_val = cluster_matrix(to_cluster, n_bit=n_bit, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row, init=init, sample_weight=sample_weight, parallelize=parallelize, x_cluster=surrogate_to_cluster, **kwargs)

    if q_group_size:
        if not scale_only:
            any4.sub_(2**(n_bit - 1))
            assign_val.sub_(2**(n_bit - 1))
        reconstructed = assign_val * scales + zeros
        del scales, zeros
    else:
        reconstructed = assign_val

    del assign, any4
    gc.collect()
    return reconstructed

cluster_row_fn_dict = {
    "scikit": cluster_row_scikit,
    "custom": cluster_row_custom,
    "agglomerative": cluster_row_agglomerative,
}

def anyq(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, any_group_size: int = None, scale_only=False, parallelize=True, bias_pow=1.0, keep_outliers=False, transpose=False, cluster_row: str = "scikit", init=None, sample_weight=None, surrogate_cluster=False, **kwargs):
    w = module.weight
    if transpose:
        w = w.t()

    if isinstance(sample_weight, Dict):
        sample_weight = sample_weight[name]

    if any_group_size:
        w = w.view(-1, any_group_size)

    w_deq = reconstruct_any4_grouped(w, n_bit=n_bit, q_group_size=group_size, scale_only=scale_only, parallelize=parallelize, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row_fn_dict[cluster_row], init=init, sample_weight=sample_weight, surrogate_cluster=surrogate_cluster, **kwargs)

    if any_group_size:
        w_deq = w_deq.view(module.weight.shape)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

def fp4(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, transpose=False):
    assert n_bit==4, "fp4 only supports 4-bit"

    w = module.weight.clone()
    if transpose:
        w = w.t()

    wq, wq_state = bnb.functional.quantize_fp4(w, blocksize=group_size)
    w_deq = bnb.functional.dequantize_fp4(wq, wq_state, blocksize=group_size)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

def nf4(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, transpose=False):
    assert n_bit==4, "nf4 only supports 4-bit"

    w = module.weight.clone()
    if transpose:
        w = w.t()

    wq, wq_state = bnb.functional.quantize_nf4(w, blocksize=group_size)
    w_deq = bnb.functional.dequantize_nf4(wq, wq_state, blocksize=group_size)

    if transpose:
        w_deq = w_deq.t()

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

quant_methods = {
    "intq": intq,
    "anyq": anyq,
    "nf4": nf4,
    "fp4": fp4,
}
