from typing import Callable, List, Tuple, Type
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
            print(f"\t{name}", end="")
            # TODO: do this in a cleaner way
            # if "mlp" not in name:
            #     print("...Skip")
            #     continue
            layer_to(module, **kwargs)
            print("...Done")
            index += 1
            if index == 6e6:
                break

    return model


# Performs row-wise (k/reduction dimension-wise) int4 group
# quantization on the m x k input tensor. Returns a tensor of the same
# size with values quantized to [0, 2^n_bit - 1], along with scale and zero point
# Reconstruction is bf16(int4_value) * scale + min_val
def group_quantize_tensor(w_orig, n_bit, q_group_size=128):
    out, scales_and_zeros = apply_q_groups(w_orig, n_bit, q_group_size)

    max_int = 2**n_bit - 1
    min_int = 0

    out = out.round().clamp_(min_int, max_int)
    out = out.to(dtype=torch.int32).reshape(w_orig.shape)

    scales_and_zeros = scales_and_zeros.to(w_orig.dtype)

    return out, scales_and_zeros

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

    out = to_quant.sub(min_val).div(scales)
    assert torch.isnan(out).sum() == 0

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

    return out, scales_and_zeros



# takes the scale or offset per each quantization group and reshapes/duplicates it to map
# to the original matrix size
def expand_q_groups(x, orig_size, q_group_size):
    out = x.reshape(orig_size[0], orig_size[1] // q_group_size, 1)
    out = out.expand(orig_size[0], orig_size[1] // q_group_size, q_group_size)
    return out.contiguous().view(orig_size)


def group_dequantize_tensor(x, scales_and_zeros, n_bit, q_group_size=128):
    scales = scales_and_zeros.transpose(0, 1)[:, :, 0]
    zeros = scales_and_zeros.transpose(0, 1)[:, :, 1]

    scales = expand_q_groups(scales, x.size(), q_group_size)
    zeros = expand_q_groups(zeros, x.size(), q_group_size)

    min_val = zeros - scales * (2 ** (n_bit - 1))

    out = x * scales + min_val

    return out


def intq(module: torch.nn.Module, n_bit: int = 4, group_size: int = 128):
    w_q, scales_and_zeros = group_quantize_tensor(module.weight.t(), n_bit, group_size)
    w_deq = group_dequantize_tensor(w_q, scales_and_zeros, n_bit, group_size)

    module.weight.data = w_deq.t().to(device=module.weight.device, dtype=module.weight.dtype)
    return module


def cluster_matrix(x, n_bit=4):
    assign = torch.zeros(x.size(), dtype=torch.int32, device=x.device)
    any4 = torch.zeros((x.size(0), 2**n_bit), dtype=x.dtype, device=x.device)
    assign_val = torch.zeros(x.size(), dtype=torch.int32, device=x.device)
    for row in range(x.size(0)):
        r = x[row].reshape(x.size(1), 1).cpu().numpy()
        clusters = KMeans(n_clusters=2**n_bit, random_state=0, n_init="auto").fit(r)
        any4[row] = torch.from_numpy(clusters.cluster_centers_).reshape(2**n_bit)
        assign[row] = torch.from_numpy(clusters.labels_)
        assign_val[row] = torch.from_numpy(clusters.cluster_centers_[clusters.predict(r)]).flatten()

    return assign, any4, assign_val


def cluster_row(r, n_bit=4):
    clusters = KMeans(n_clusters=2**n_bit, random_state=0, n_init="auto").fit(r)
    any4 = torch.from_numpy(clusters.cluster_centers_).reshape(2**n_bit)
    assign = torch.from_numpy(clusters.labels_)
    assign_val = torch.from_numpy(clusters.cluster_centers_[clusters.predict(r)]).flatten()

    return assign, any4, assign_val


def cluster_matrix_parallel(x, n_bit=4):
    x_np = x.detach().cpu().numpy()
    results: List = Parallel(n_jobs=-1)(delayed(cluster_row)(r.reshape(-1, 1), n_bit) for r in x_np)
    # Transpose the list of tuples to a tuple of lists
    results_transposed = tuple(zip(*results))
    # Convert each item in the tuple (which are tuples) to lists
    results_transposed = tuple(list(item) for item in results_transposed)

    # Unpack into different matrices
    assign = torch.stack(results_transposed[0], dim=0).contiguous().to(x.device)
    any4 = torch.stack(results_transposed[1], dim=0).contiguous().to(x.device)
    assign_val = torch.stack(results_transposed[2], dim=0).contiguous().to(x.device)

    return assign, any4, assign_val


def quantize_to_any4(x, n_bit = 4, q_group_size=128, bias_extreme_values=True, parallelize=True):
    to_cluster, scales_and_zeros = apply_q_groups(x, n_bit, q_group_size)
    # print(to_cluster.unique())

    assign = None
    any4 = None
    assign_val = None

    if bias_extreme_values:
        # k-means should be roughly zero centered, since we should bias larger magnitude (negative or positive) values
        # for greater representation.
        # Values are in the range [0, 15] so subtract (15 - 0) / 2 = 7.5 to approximately zero center the data
        #
        # Note that there is no guarantee that each q-group is itself zero centered (there can be a "DC bias")
        # but note that across all q-groups, values closer to 0 and closer to 15 are extremal values
        to_cluster = to_cluster - ((2 ** n_bit) - 1) / 2. 
        # give more weight to extremal values by considering the signed square
        to_cluster = (to_cluster ** 2) * torch.sign(to_cluster)

    if parallelize:
        assign, any4, assign_val = cluster_matrix_parallel(to_cluster, n_bit)
    else:
        assign, any4, assign_val = cluster_matrix(to_cluster, n_bit)
    if bias_extreme_values:
        # undo the square
        any4 = torch.sqrt(any4.abs()) * torch.sign(any4)

        # map values back to [0, 15]
        any4 = any4 + ((2 ** n_bit) - 1) / 2.

    # dequant is in the range [-8, 7] so adjust again
    any4 = any4 - 2 ** (n_bit - 1)

    return assign, any4.to(dtype=x.dtype), assign_val.to(dtype=x.dtype), scales_and_zeros.to(dtype=x.dtype)


def anyq(module: torch.nn.Module, n_bit: int = 4, group_size: int = 128, bias_extreme_values: bool = False):
    w = module.weight.t()

    if w.shape[-1] % group_size != 0:
        # group_size = w.shape[-1]
        # TODO: perhaps check if emedding and unembedding are tied, and exit if that's the case
        return module

    _, _, assign_val, scales_and_zeros = quantize_to_any4(w, n_bit, group_size, bias_extreme_values)
    w_deq = group_dequantize_tensor(assign_val, scales_and_zeros, n_bit, group_size)
    w_deq = w_deq.reshape(w.shape).t()
    assert w_deq.shape == module.weight.shape

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

def kmeans_clustering_vector(v):
    kmeans = KMeans(n_clusters=16, random_state=0, n_init="auto").fit(v.reshape(-1, 1))
    return kmeans.cluster_centers_[kmeans.predict(v.reshape(-1, 1))].flatten()

def any4(module: torch.nn.Module, granularity: str = "col", quantization: str = "clustering"):
    weight = module.weight.clone()

    # reshape based on granularity
    match granularity:
        case "row":
            pass
        case "col":
            weight = weight.transpose(0, 1)
        case "8x8":
            # TODO: reshape or review into <fo//8, fi//8, 8, 8>
            raise ValueError(f"Unsupported {granularity} type")
        case _:
            raise ValueError(f"Unsupported {granularity} type")

    match quantization:
        case "scalar":
            groups, dim = weight.shape
            # QT_4bit allocates 4 bits per dimension
            sq = faiss.ScalarQuantizer(dim, faiss.ScalarQuantizer.QT_4bit)

            w = weight
            w_proc = w # torch.cat((w, w.quantile(q=0.0, dim=-1).repeat(1, 20), w.quantile(q=1.0, dim=-1).repeat(1, 20)), dim=-1)
            try:
                # this should work if faiss-gpu is working
                sq.train(w_proc.detach())
            except:
                # this should work if faiss-cpu is working
                if module.weight.dtype == torch.bfloat16:
                    w_proc = w_proc.half()
                sq.train(w_proc.detach().cpu())

            # decode 
            try:
                # this should work if faiss-gpu is working
                codes_proc = sq.compute_codes(w_proc.detach())
            except:
                # this should work if faiss-cpu is working
                codes_proc = sq.compute_codes(w_proc.detach().cpu())
            wq_proc = sq.decode(codes_proc)
            wq = wq_proc # wq_proc[:out_features, :in_features]

        case "clustering":
            try:
                w = weight.detach().numpy()
            except:
                w = weight.cpu().detach().numpy()
            wq = w
            wq = Parallel(n_jobs=-1)(delayed(kmeans_clustering_vector)(v) for v in w)

        case _:
            raise ValueError(f"Unsupported {quantization} type")

    # reshape based on granularity
    match granularity:
        case "row":
            pass
        case "col":
            wq = np.transpose(wq)
        case "8x8":
            # TODO: reshape or review into <fo, fi>
            raise ValueError(f"Unsupported {granularity} type")
        case _:
            raise ValueError(f"Unsupported {granularity} type")

    module.weight.data = torch.from_numpy(wq).to(device=module.weight.device, dtype=module.weight.dtype)

    return module
