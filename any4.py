# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Tuple, Type
import random
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

    # TODO: use tqdm instead of printing each layer name
    for name, module in list(model.named_modules()):
        if isinstance(module, (layer_from)):
            print(f"{name}")
            if name in skip_modules:
                print("\tSkip")
                continue

            # Calibrate if necessary
            if calibrate_fn is not None:
                calibrate_args["seed"] = index
                if calibrate_args.get("return_activations", False):
                    # TODO: rename "sample_weight" to "sample_mean_activations" ?
                    kwargs["sample_weight"], kwargs["sample_activations"] = calibrate_fn(model=model, tokenizer=tokenizer, layers=[name], **calibrate_args)
                    kwargs["sample_activations"] = kwargs["sample_activations"][name]
                else:
                    kwargs["sample_weight"] = calibrate_fn(model=model, tokenizer=tokenizer, layers=[name], **calibrate_args)

            new_module = layer_to(module, name=name, **kwargs)

            # Get the parent module
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
            # Replace the old module with the new one
            setattr(parent_module, name.split('.')[-1], new_module)

            index += 1

            # Save memory
            if calibrate_fn is not None:
                if "sample_weight" in kwargs:
                    del kwargs["sample_weight"]
                if "sample_activations" in kwargs:
                    del kwargs["sample_activations"]
            torch.cuda.empty_cache()
            gc.collect()

    return model

def pack_scales_and_zeros(scales, zeros, w_shape):
    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w_shape[0], -1)
    zeros = zeros.view(w_shape[0], -1)
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
    return scales_and_zeros

# TODO: add option to group_q to decide max and min of scaling: 0 to 15? -1 to 1? -7 to 8? -7.5 to 8.5?
def group_q(w_orig, n_bit, q_group_size=128, assymetric=True, unsigned=True, zero_point=True):
    w = w_orig.float()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    if assymetric:
        max_val = to_quant.amax(dim=1, keepdim=True)
        min_val = to_quant.amin(dim=1, keepdim=True)
        if unsigned:
            min_int = 0
            max_int = 2**n_bit - 1
        else:
            min_int = -(2**(n_bit - 1))
            max_int = 2**(n_bit - 1) - 1
        scales = (max_val - min_val).clamp(min=1e-6) / (max_int - min_int)
        if zero_point:
            zeros = min_val + scales * (2 ** (n_bit - 1))
        else:
            zeros = min_val

        w_new = to_quant.sub(min_val).div(scales).reshape(w_orig.size())
        w_new_zeros = torch.zeros(to_quant.size(), dtype=to_quant.dtype, device=to_quant.device).sub(min_val).div(scales).reshape(w_orig.size())
    else:
        # TODO: this wastes one bit. Review with Jeff.
        # TODO: check how to handle "unsigned" argument
        absmax_val = to_quant.abs().amax(dim=1, keepdim=True)
        absmax_int = 2**(n_bit - 1) - 1
        scales = absmax_val.clamp(min=1e-6) / absmax_int
        zeros = torch.zeros_like(scales)

        w_new = to_quant.div(scales).reshape(w_orig.size())
        w_new_zeros = torch.zeros_like(w_new)


    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(zeros).sum() == 0

    scales_and_zeros = pack_scales_and_zeros(scales, zeros, w.shape)

    return w_new, w_new_zeros, scales_and_zeros

def extract_scales_and_zeros(scales_and_zeros, w_shape, q_group_size):
    scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
    zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

    scales = expand_q_groups(scales, w_shape, q_group_size)
    zeros = expand_q_groups(zeros, w_shape, q_group_size)

    return scales, zeros

def degroup_q(w_c, scales_and_zeros=None, scales=None, zeros=None, n_bit=4, q_group_size=128, centering=True):
    if scales_and_zeros is not None:
        scales1, zeros1 = extract_scales_and_zeros(scales_and_zeros, w_c.shape, q_group_size)
        if scales is None:
            scales = scales1
        if zeros is None:
            zeros = zeros1

    if q_group_size:
        if centering:
            w_c = w_c - (2**(n_bit - 1))
        reconstructed = w_c * scales + zeros
    else:
        reconstructed = w_c
    return reconstructed

# takes the scale or offset per each quantization group and reshapes/duplicates it to map
# to the original matrix size
def expand_q_groups(x, orig_size, q_group_size):
    out = x.reshape(orig_size[0], orig_size[1] // q_group_size, 1)
    out = out.expand(orig_size[0], orig_size[1] // q_group_size, q_group_size)
    return out.contiguous().view(orig_size)

def intq_quantize(x, n_bit = 4, q_group_size=128, parallelize=True, scale_only=False, new_grouping=False, unsigned=False, zero_point=False, **kwargs):
    if new_grouping:
        intq, scales, zeros = group_q1(x, n_bit=n_bit, assymetric=not scale_only, q_group_size=q_group_size, inplace=False, get_scale_zp=True)
        scales_and_zeros = pack_scales_and_zeros(scales, zeros, x.shape)
        intq = intq.round()
        intq_zeros = zeros
    else:
        intq, intq_zeros, scales_and_zeros = group_q(x, n_bit, q_group_size=q_group_size, assymetric=not scale_only, unsigned=unsigned, zero_point=zero_point)
        intq = intq.round()
        assert intq.size(1) == q_group_size * scales_and_zeros.size(0)

    intq = intq.to(torch.int32)
    scales_and_zeros = scales_and_zeros.to(x.dtype)

    return intq, intq_zeros, scales_and_zeros

def intq_dequantize(intq, scales_and_zeros=None, scales=None, zeros=None, n_bit=4, q_group_size=128, new_grouping=False, dtype=torch.float16, unsigned=False):
    if new_grouping:
        reconstructed = degroup_q1(intq, scales_and_zeros=scales_and_zeros, scales=scales, zeros=zeros, q_group_size=q_group_size, inplace=False)
    else:
        reconstructed = degroup_q(intq, scales_and_zeros=scales_and_zeros, scales=scales, zeros=zeros, n_bit=n_bit, q_group_size=q_group_size, centering=False)
    reconstructed = reconstructed.to(dtype=dtype)
    return reconstructed

# performs quantization and dequantization under N-bit grouped integer quantization
# (i.e., returns the effective result of the quantization algorithm)
def intq_reconstruct(x, n_bit = 4, q_group_size=128, parallelize=True, unsigned=False, new_grouping=False, zero_point=False,  scale_only=False, dtype=torch.float16, *args, **kwargs):
    intq, intq_zeros, scales_and_zeros = intq_quantize(x, n_bit=n_bit, q_group_size=q_group_size, new_grouping=new_grouping, scale_only=scale_only, unsigned=unsigned, zero_point=zero_point)
    reconstructed = intq_dequantize(intq, scales_and_zeros=scales_and_zeros, n_bit=n_bit, q_group_size=q_group_size, dtype=dtype, new_grouping=new_grouping, unsigned=unsigned)

    return reconstructed

def pseudo_quantize_tensor(
    w, n_bit=8, assymetric=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if assymetric:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
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

# TODO: add min_int and max_int as optional args
def group_q1(
    w, n_bit=4, assymetric=True, q_group_size=-1, inplace=False, get_scale_zp=False, clamp=True,
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if assymetric:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        w.div_(scales).add_(zeros)
        if clamp:
            w.clamp_(min_int, max_int)
    else:
        w = (w / scales) + zeros
        if clamp:
            w = torch.clamp(w, min_int, max_int)
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales, zeros
    else:
        return w


def degroup_q1(
    w, scales_and_zeros=None, scales=None, zeros=None, q_group_size=-1, inplace=False
):
    if scales is None:
        scales, zeros = extract_scales_and_zeros(scales_and_zeros, w.shape, q_group_size)

    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        w.sub_(zeros).mul_(scales)
    else:
        w = (w - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    return w

def intq(module: torch.nn.Linear, n_bit: int = 4, group_size: int = 128, transpose=False, pseudo=True, **kwargs):
    w = module.weight

    if transpose:
        w = w.t()

    if pseudo:
        w_deq = intq_reconstruct(w, n_bit=n_bit, q_group_size=group_size, **kwargs)
        # w_deq = pseudo_quantize_tensor(w, n_bit=n_bit, assymetric=not kwargs.get("scale_only", False), q_group_size=group_size)

        if transpose:
            w_deq = w_deq.t()

        module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    else:
        intq, _, scales_and_zeros = intq_quantize(w, n_bit=n_bit, q_group_size=group_size, **kwargs)

        if transpose:
            intq = intq.t()
            scales, zeros = extract_scales_and_zeros(scales_and_zeros, w.shape, group_size)
            scales = scales.t()
            zeros = zeros.t()
            scales_and_zeros = pack_scales_and_zeros(scales, zeros, w.shape)
        if n_bit == 4:
            from modules import Int4Linear
            qmodule = Int4Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                qtype=f"int{n_bit}",
                dtype=module.weight.dtype,
                group_size=group_size,
            )
        else:
            raise ValueError(f"No int quantized modules supported for n_bit={n_bit}. You may consider setting pseudo=True.")
        qmodule.weight.data = intq.to(device=module.weight.device)
        qmodule.scales_and_zeros.data = scales_and_zeros.to(device=module.weight.device)
        qmodule.bias = module.bias
        qmodule.reshape_weight()
        module = qmodule

    return module

def cluster_row_custom(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, abs_sample_weight=True, **kwargs):
    init = kmeans.build_init(x=r, n_clusters=2 ** n_bit, init_type=init)
    if init is None:
        init = "k-means++"
    sample_weight = kmeans.build_sample_weight(x=r, sample_weight_type=sample_weight, abs=abs_sample_weight)
    assign_val, any4, assign = kmeans.kmeans(r, n_clusters=2**n_bit, init=init, sample_weight=sample_weight, X_surrogate=r_surrogate, **kwargs)

    any4 = torch.from_numpy(any4).reshape(2**n_bit)
    assign = torch.from_numpy(assign).flatten()
    assign_val = torch.from_numpy(assign_val).flatten()

    return assign, any4, assign_val

def cluster_row_scikit(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, abs_sample_weight=True, **kwargs):
    assert r_surrogate==None, "scikit clustering does not support surrogate_to_cluster"
    init = kmeans.build_init(x=r, n_clusters=2 ** n_bit, init_type=init)
    if init is None:
        init = "k-means++"
    sample_weight = kmeans.build_sample_weight(x=r, sample_weight_type=sample_weight, abs=abs_sample_weight)

    clusters = sklearn.cluster.KMeans(n_clusters=2**n_bit, init=init, random_state=0, n_init="auto", **kwargs).fit(r, sample_weight=sample_weight)
    any4 = torch.from_numpy(clusters.cluster_centers_).reshape(2**n_bit)
    assign = torch.from_numpy(clusters.labels_)
    assign_val = torch.from_numpy(clusters.cluster_centers_[clusters.predict(r)]).flatten()

    return assign, any4, assign_val

def cluster_row_agglomerative(r, n_bit=4, init=None, sample_weight=None, r_surrogate=None, abs_sample_weight=True, **kwargs):
    assert r_surrogate==None, "scikit clustering does not support surrogate_to_cluster"
    assert init==None, "agglomerative clustering does not support init"
    sample_weight = kmeans.build_sample_weight(x=r, sample_weight_type=sample_weight, abs=abs_sample_weight)

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
    to_cluster = x.cpu().detach().float().numpy()
    surrogate_to_cluster = x_cluster.cpu().float().detach().numpy() if x_cluster is not None else None
    sample_weight = sample_weight.cpu().float().detach().numpy() if isinstance(sample_weight, torch.Tensor) else sample_weight
    print(f"\tClustering...", end=" ", flush=True)
    if parallelize:
        assign, any4, assign_val = cluster_rows_parallel(to_cluster, cluster_row=cluster_row, n_bit=n_bit, init=init, sample_weight=sample_weight, x_surrogate=surrogate_to_cluster, **kwargs)
    else:
        assign, any4, assign_val = cluster_rows(to_cluster, cluster_row=cluster_row, init=init, n_bit=n_bit, sample_weight=sample_weight, x_surrogate=surrogate_to_cluster, **kwargs)
    assign_val = assign_val.to(x.device)
    print(f"{time.time() - start:.2f} s", flush=True)


    if keep_outliers:
        max_outliers = torch.amax(x, dim=1, keepdim=True)
        min_outliers = torch.amin(x, dim=1, keepdim=True)

        any4.scatter_(dim=1, index=torch.argmax(any4, dim=1, keepdim=True), src=max_outliers.to(any4.device))
        any4.scatter_(dim=1, index=torch.argmin(any4, dim=1, keepdim=True), src=min_outliers.to(any4.device))

        assign_val.scatter_(dim=1, index=torch.argmax(assign_val, dim=1, keepdim=True), src=max_outliers.to(assign_val.device))
        assign_val.scatter_(dim=1, index=torch.argmin(assign_val, dim=1, keepdim=True), src=min_outliers.to(assign_val.device))

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
            assign[row], any4[row], assign_val[row] = cluster_row(r, n_bit, sample_weight=get_sample_weight(sample_weight, row), r_surrogate=r_surrogate, **kwargs)
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

def anyq_quantize(W, n_bit=4, q_group_size=128, new_grouping=False, per_row=True, zero_point=True, scale_only=False, bias_pow=1.0, keep_outliers=False, cluster_row: Callable = cluster_row_scikit, init=None, sample_weight=None, sample_weight_preprocess=None, sample_activations=None, scale_sample_weight=False, abs_weight_sample_weight=False, parallelize=True, surrogate_cluster=False, nnq=False, nnq_args={}, device=None, **kwargs):
    orig_device = W.device
    if device is not None:
        W = W.to(device)

    if not per_row:
        orig_shape = W.shape
        W = W.reshape(1, -1)

    if sample_weight_preprocess:
        assert sample_weight is not None and isinstance(sample_weight, torch.Tensor)
        # We won't apply absolute here and it can be applied in another call to build_sample_weight before clustering
        sample_weight = kmeans.build_sample_weight(sample_weight.unsqueeze(dim=1).detach().cpu().numpy(), sample_weight_preprocess, abs=False)
        sample_weight = torch.Tensor(sample_weight)

    if q_group_size:
        # TODO: create separate function that fuses scales and zeros into scales_and_zeros, and only use that when actually quantizing rather than reconstructing
        if new_grouping:
            W = W.detach()
            Wg, scales, zeros = group_q1(W, n_bit, q_group_size=q_group_size, assymetric=not scale_only, get_scale_zp=True, inplace=True)
            scales_and_zeros = pack_scales_and_zeros(scales, zeros, W.shape)
        else:
            Wg, _, scales_and_zeros = group_q(W, n_bit, q_group_size=q_group_size, assymetric=not scale_only, zero_point=zero_point)
            scales, zeros = extract_scales_and_zeros(scales_and_zeros, Wg.shape, q_group_size)

        if scale_sample_weight:
            if sample_weight is None:
               sample_weight = torch.ones_like(W[0])
            sample_weight = sample_weight.to(scales.device) * scales
    else:
        Wg = W.float()
        scales, zeros = torch.ones(W.shape[0]), torch.zeros(W.shape[0])
        scales_and_zeros = pack_scales_and_zeros(scales, zeros, W.shape)

    if abs_weight_sample_weight:
        if sample_weight is None:
            sample_weight = torch.ones_like(W[0])
        sample_weight = sample_weight.to(W.device) * W.abs()

    Wg = Wg.contiguous()
    if sample_weight is not None and isinstance(sample_weight, torch.Tensor):
        sample_weight = sample_weight.contiguous()

    if surrogate_cluster:
        surrogate_to_cluster = W
    else:
        surrogate_to_cluster = None

    if cluster_row is not None:
        assign, any4, Wc = cluster_matrix(Wg, n_bit=n_bit, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row, init=init, sample_weight=sample_weight, parallelize=parallelize, x_cluster=surrogate_to_cluster, **kwargs)
    else:
        assert nnq, "We should either enabling clustering (cluster_row should be not None) or enable neural network learning (nnq should be True) but we have cluster_row: {cluster_row}, nnq: {nnq}"
        assign, any4, Wc = None, None, Wg

    if nnq:
        try:
            assign, any4, Wc = learn_anyq(
                Wc=Wc,
                scales=scales,
                zeros=zeros,
                W=W,
                n_bit=n_bit,
                q_group_size=q_group_size,
                scale_only=scale_only,
                init_values=any4,
                X_train=sample_activations,
                X_val=sample_weight,
                device=device,
                **nnq_args,
            )
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f"Hit OOM so will not update this layer")
            else:
                raise
        except Exception as e:
            raise
        # Ensure tensors are back on same device as weight
        Wc = Wc.to(W.device)

    if not per_row:
        W = W.reshape(orig_shape)
        assign = assign.reshape(orig_shape)
        scales_and_zeros = scales_and_zeros.reshape(-1, orig_shape[0], 2)
        any4 = any4.squeeze()

    return assign.to(orig_device), any4.to(orig_device, W.dtype), scales_and_zeros.to(orig_device, W.dtype)

def anyq_dequantize(assign, any4, scales_and_zeros, n_bit=4, q_group_size=128, per_row=True, new_grouping=False, scale_only=False):
    if not per_row:
        orig_shape = assign.shape
        assign = assign.reshape(1, -1)
        scales_and_zeros = scales_and_zeros.reshape(-1, 1, 2)
        Wc = any4[assign]
    else:
        Wc = torch.gather(input=any4, dim=1, index=assign.long())
    scales, zeros = extract_scales_and_zeros(scales_and_zeros, assign.shape, q_group_size)
    # TODO: create separate de_group function
    if q_group_size:
        if new_grouping:
            Wdeq = degroup_q1(Wc, scales=scales, zeros=zeros, q_group_size=q_group_size)
        else:
            # Commenting this inplace operation because this applies a side effect on an input argument that won't be noticed when calling the function
            # if not scale_only:
            #     any4.sub_(2**(n_bit - 1))
            Wdeq = degroup_q(Wc, scales=scales, zeros=zeros, n_bit=n_bit, q_group_size=q_group_size, centering=not scale_only)
        del scales, zeros
    else:
        Wdeq = Wc

    if not per_row:
        Wdeq = Wdeq.reshape(orig_shape)

    return Wdeq

class STEMin(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input.min(dim=-1)

  @staticmethod
  def backward(ctx, grad_output):
    return torch.nn.functional.hardtanh(grad_output)

class AnyQNN(torch.nn.Module):
    def __init__(self, n_values=16, n_rows=1):
        super(AnyQNN, self).__init__()
        # Initialize trainable values
        self.values = torch.nn.Parameter(torch.randn(n_rows, n_values))
        # Initialize trainable mappings
        self.n_values = n_values
        self.n_rows = n_rows

    def forward(self, x):
        # Expand input tensor and trainable values for vectorized distance computation
        x_expanded = x.unsqueeze(dim=-1)
        values_expanded = self.values.unsqueeze(dim=1).expand(-1, x.shape[-1], -1)

        # Calculate distances between input values and trainable values
        distances = (x_expanded - values_expanded)**2

        # Find the index of the minimum distance for each element
        # _, min_indices = distances.min(dim=-1)
        _, min_indices = STEMin.apply(distances)
        min_indices.reshape(x.shape)

        # Select the values with minimum distance
        # Create a tensor for the row indices
        row_indices = torch.arange(min_indices.size(0)).unsqueeze(1).expand_as(min_indices)
        # Use advanced indexing to select the elements
        selected_values = self.values[row_indices, min_indices]

        return selected_values

def nlc_loss(output, label):
    # Calculate the cosine similarity between output and label
    cosine_sim = torch.nn.functional.cosine_similarity(output, label).mean().abs()
    # Calculate the negative log-likelihood loss
    nlc = -torch.log(cosine_sim)
    return nlc

# TODO: try lr schedule.
# TODO: in each iteration feed different activations
def learn_anyq(Wc, scales, zeros, W, n_bit=4, q_group_size=128, scale_only=False, init_values=None, objective="Y_mse", X_val=None, X_train=None, lr=0.001, transpose=False, overfit=True, dtype=None, device=None, epochs=500):
    n_rows, dim = Wc.shape
    n_values = 2**n_bit
    if device == "cpu":
        if dtype is None:
            # CPU does not support torch.half
            dtype = torch.float32
    if dtype is None:
        dtype = W.dtype
    W = W.to(dtype=dtype, device=device)
    Wc = Wc.to(dtype=dtype, device=device)
    if scales is not None:
        scales = scales.to(device=device)
    if zeros is not None:
        zeros = zeros.to(device=device)

    # Create network
    net = AnyQNN(n_values=n_values, n_rows=n_rows)
    if init_values is not None:
        net.values.data = init_values.to(dtype=dtype, device=W.device)
    net.to(dtype=dtype, device=W.device)
    net.train()

    # Create learning objective
    if objective.endswith("mse"):
        criterion = torch.nn.MSELoss()
    elif objective.endswith("cossim"):
        criterion = nlc_loss
    else:
        raise ValueError(f"Unsupoorted objective {objective}")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    Wcqn = net(Wc)
    Wqn = degroup_q(Wcqn, scales=scales, zeros=zeros, n_bit=n_bit, q_group_size=q_group_size, centering=not scale_only).to(dtype)
    # TODO: we will probably need to refactor the codeo to handle transpose and decide when we should transpose and de-transpose
    if transpose:
        Wqn = Wqn.T

    if X_train is not None:
        X_train = [X_i.to(device=W.device, dtype=dtype) for X_i in X_train]

    if X_val is None:
        # TODO: add bs and slen as arguments to this function?
        bs, slen = 32, 1024
        X_val =  torch.randn(bs, slen, dim, device=W.device, requires_grad=False, dtype=dtype)
    else:
        X_val = X_val.to(device=W.device, dtype=dtype)

    # Check final outputs
    Y_val = torch.matmul(X_val, W.T)
    Yqn_val = torch.matmul(X_val, Wqn.T)

    W_mse = torch.nn.functional.mse_loss(W.squeeze(), Wqn.squeeze())
    Y_val_mse = torch.nn.functional.mse_loss(Y_val, Yqn_val)
    W_cossim = torch.nn.functional.cosine_similarity(W.flatten(), Wqn.flatten(), dim=0)

    print("W_mse:", W_mse.item(), "W_cossim:", W_cossim.item())
    print("Y_val_mse:", Y_val_mse.item())

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        Wcqn = net(Wc)
        Wqn = degroup_q(Wcqn, scales=scales, zeros=zeros, n_bit=n_bit, q_group_size=q_group_size, centering=not scale_only).to(dtype)

        if transpose:
            Wqn = Wqn.T

        if objective == "W_mse":
            output = Wqn
            label = W
        elif objective == "Y_mse":
            if X_train is not None:
                Xi = X_train[random.randint(0, len(X_train)-1)]
                Yi = torch.matmul(Xi, W.T)
            elif overfit:
                Xi = X_val
                Yi = Y_val.squeeze()
            else:
                Xi = torch.randn_like(X_val)
                Yi = torch.matmul(Xi, W.T)
            output = torch.matmul(Xi, Wqn.T).squeeze()
            label = Yi.squeeze()

        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    net.hard_max = True
    Wcqn = net(Wc)
    Wqn = degroup_q(Wcqn, scales=scales, zeros=zeros, n_bit=n_bit, q_group_size=q_group_size, centering=not scale_only).to(dtype)
    if transpose:
        Wqn = Wqn.T

    # Check final outputs
    Y_val = torch.matmul(X_val, W.T)
    Yqn_val = torch.matmul(X_val, Wqn.T)

    W_mse = torch.nn.functional.mse_loss(W.squeeze(), Wqn.squeeze())
    Y_val_mse = torch.nn.functional.mse_loss(Y_val.squeeze(), Yqn_val.squeeze())
    W_cossim = torch.nn.functional.cosine_similarity(W.view(-1), Wqn.view(-1), dim=0)

    # print("Y_val:", Y_val, "Yqn_val:", Yqn_val)

    print("W_mse:", W_mse.item(), "W_cossim:", W_cossim.item())
    print("Y_val_mse:", Y_val_mse.item())

    assign_vals = Wcqn
    any4 = net.values.data
    # FIXME: fill assign rather than setting it to None
    assign = None

    del net, X_train, X_val, Wc, Yqn_val
    torch.cuda.empty_cache()
    gc.collect()

    return assign, any4, assign_vals

# performs quantization and dequantization under any4 scalar k-means grouped integer quantization
# (i.e., returns the effective result of the quantization algorithm)
def anyq_reconstruct(W, n_bit=4, q_group_size=128, new_grouping=False, per_row=True, scale_only=False, bias_pow=1.0, keep_outliers=False, cluster_row: Callable = cluster_row_scikit, init=None, sample_weight=None, sample_weight_preprocess=None, sample_activations=None, scale_sample_weight=False, abs_weight_sample_weight=False, parallelize=True, surrogate_cluster=False, nnq=False, nnq_args={}, device=None, **kwargs):
    assign, any4, scales_and_zeros = anyq_quantize(W, n_bit=n_bit, q_group_size=q_group_size, per_row=per_row, new_grouping=new_grouping, scale_only=scale_only, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row, init=init, sample_weight=sample_weight, sample_weight_preprocess=sample_weight_preprocess, sample_activations=sample_activations, scale_sample_weight=scale_sample_weight, abs_weight_sample_weight=abs_weight_sample_weight, parallelize=parallelize, surrogate_cluster=surrogate_cluster, nnq=nnq, nnq_args=nnq_args, device=device, **kwargs)
    Wdeq = anyq_dequantize(assign, any4, scales_and_zeros, n_bit=n_bit, q_group_size=q_group_size, per_row=per_row, new_grouping=new_grouping, scale_only=scale_only)

    del assign, any4
    torch.cuda.empty_cache()
    gc.collect()

    return Wdeq

cluster_row_fn_dict = {
    "scikit": cluster_row_scikit,
    "custom": cluster_row_custom,
    "agglomerative": cluster_row_agglomerative,
}

# TODO: create anyq, nf4, fp4, intq functions that take weight tensor as input and return weight tensor as output?
def anyq(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, any_group_size: int = None, per_row = True, scale_only=False, parallelize=True, bias_pow=1.0, keep_outliers=False, transpose=False, cluster_row: str = "scikit", init=None, sample_weight=None, surrogate_cluster=False, pseudo=True, kernel: str = "linear_y_f16RM_x_f16RM_W_any4TC",  w_inner_k: int = 4, **kwargs):
    w = module.weight
    if isinstance(sample_weight, Dict):
        sample_weight = sample_weight[name]

    if pseudo:
        if transpose:
            w = w.t()
        if any_group_size:
            w = w.view(-1, any_group_size)

        # TODO: implement this try-except with a decorator to the function?
        try:
            w_deq = anyq_reconstruct(w, n_bit=n_bit, q_group_size=group_size, per_row=per_row, scale_only=scale_only, parallelize=parallelize, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row_fn_dict[cluster_row], init=init, sample_weight=sample_weight, surrogate_cluster=surrogate_cluster, **kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Hit OOM so will move weights to CPU and re-run")
                orig_device = w.device
                w.to("cpu")
                w_deq = anyq_reconstruct(w, n_bit=n_bit, q_group_size=group_size, per_row=per_row, scale_only=scale_only, parallelize=parallelize, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row_fn_dict[cluster_row], init=init, sample_weight=sample_weight, surrogate_cluster=surrogate_cluster, **kwargs)
                w_deq.to(orig_device)
            else:
                raise
        except Exception as e:
            raise

        if any_group_size:
            w_deq = w_deq.view(module.weight.shape)
        if transpose:
            w_deq = w_deq.t()

        module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    else:
        assert transpose is False, "anyq quantized modules don't yet support transpose."
        assert any_group_size is None, "anyq quantized modules don't yet support transpose."
        intq, lut, scales_and_zeros = anyq_quantize(w, n_bit=n_bit, q_group_size=group_size, per_row=per_row, scale_only=scale_only, parallelize=parallelize, bias_pow=bias_pow, keep_outliers=keep_outliers, cluster_row=cluster_row_fn_dict[cluster_row], init=init, sample_weight=sample_weight, surrogate_cluster=surrogate_cluster, **kwargs)

        if n_bit == 4:
            from modules import Any4Linear
            qmodule = Any4Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
                group_size=group_size if group_size is not 0 else w.shape[1],
                per_row=per_row,
                kernel=kernel,
                w_inner_k=w_inner_k,
            )
        else:
            raise ValueError(f"No int quantized modules supported for n_bit={n_bit}. You may consider setting pseudo=True.")
        qmodule.weight.data = intq.to(device=module.weight.device)
        qmodule.scales_and_zeros.data = scales_and_zeros.to(device=module.weight.device)
        qmodule.lut.data = lut.to(device=module.weight.device) - (2**(n_bit - 1))
        qmodule.bias = module.bias
        qmodule.reshape_weight()
        module = qmodule

    # Save memory
    if isinstance(sample_weight, Dict):
        del sample_weight[name]
        sample_weight[name]= None
        torch.cuda.empty_cache()
        gc.collect()

    return module

def fp4(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, transpose=False, **kwargs):
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

def nf4(module: torch.nn.Module, name="", n_bit: int = 4, group_size: int = 128, transpose=False, **kwargs):
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
