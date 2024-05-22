from typing import Callable, Type
import torch
import numpy as np
import faiss

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
            print(f"name: {name}", end="")
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


def any4(module: torch.nn.Module):
    out_features, in_features = module.weight.shape
    d = in_features
    # QT_4bit allocates 4 bits per dimension
    sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_4bit)

    w = module.weight
    w_proc = w # torch.cat((w, w.quantile(q=0.0, dim=-1).repeat(1, 20), w.quantile(q=1.0, dim=-1).repeat(1, 20)), dim=-1)
    sq.train(w_proc.detach())

    # encode 
    codes_proc = sq.compute_codes(w_proc.detach())
    wq_proc = sq.decode(codes_proc)
    wq = wq_proc # wq_proc[:out_features, :in_features]

    # decode
    module.weight.data = torch.from_numpy(wq).to(module.weight.device)

    return module
