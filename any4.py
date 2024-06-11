from typing import Callable, Type
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
    w_q, scales_and_zeros = group_quantize_tensor(module.weight, n_bit, group_size)
    w_deq = group_dequantize_tensor(w_q, scales_and_zeros, n_bit, group_size)

    module.weight.data = w_deq.to(device=module.weight.device, dtype=module.weight.dtype)
    return module

def anyq(module: torch.nn.Module, n_bits: int=4, n_rows: int= 2048, n_iter: int=20):
    out_features, in_features = module.weight.shape
    n_rows = n_rows if n_rows else out_features
    for fo in range(0, out_features, n_rows):
        rem_rows = min(out_features - fo, n_rows)
        kmeans = faiss.Kmeans(d=1, k=2**n_bits, niter=n_iter, gpu=torch.cuda.is_available())
        points: torch.Tensor = module.weight[fo:fo+rem_rows, :].detach().flatten().unsqueeze(dim=1)
        kmeans.train(x=points)

        # D is the distances from each vector to its assigned cluster center
        # I is the index of the assigned cluster for each vector
        # centroid is the centre of each cluster
        D, I = kmeans.assign(x=points)
        centroids = kmeans.centroids
        clustered_points = centroids[I]

        module.weight.data[fo:fo+rem_rows, :] = torch.from_numpy(clustered_points).reshape(rem_rows, in_features).to(module.weight.device)

    return module


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

            def kmeans_clustering_vector(v):
                kmeans = KMeans(n_clusters=16, random_state=0, n_init="auto").fit(v.reshape(-1, 1))
                return kmeans.cluster_centers_[kmeans.predict(v.reshape(-1, 1))].flatten()

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
