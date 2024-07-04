import torch
import time
from utils import log, get_max_n_numbers, get_min_n_numbers
import numpy as np
import re

nf4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

def build_init(x, n_clusters, init_type):
    K = n_clusters
    N, D = x.shape  # Number of samples, dimension of the ambient space

    match init_type:
        case None:
            return None

        case "random":
            return "random"

        case "int":
            # NOTE: What I did hear doesn't really make sense for D > 1. 
            init = torch.zeros(K, D)
            for i in range(D):
                init[:, i] = torch.linspace(start=x[:, i].min(), end=x[:, i].max(), steps=K)
            return init

        case "pow":
            # NOTE: What I did hear doesn't really make sense for D > 1
            init = torch.zeros(K, D)
            for i in range(D):
                init[:, i] = torch.logspace(start=1, end=log(x[:, i].max()) / log(x[:, i].min()), steps=K, base=x[:, i].min())
            return init

        case "nf4":
            # NOTE: What I did hear doesn't really make sense for D > 1. 
            assert K == 16, "nf4 only works with 16 clusters"
            init = torch.zeros(K, D)
            for i in range(D):                
                init_vals = torch.Tensor(nf4)       # -1 to +1
                init_vals += 1                  # 0 to +2
                init_vals /= 2                  # 0 to +1
                init_vals *= x[:, i].max() - x[:, i].min() # 0 to max-min
                init_vals += x[:, i].min()      # min to max 
                init[:, i] = init_vals
            return init


def build_sample_weight(x, sample_weight_type: str):
    N, _ = x.shape  # Number of samples, dimension of the ambient space

    if sample_weight_type is None:
        return None
    elif sample_weight_type.startswith("outlier"):
        # This pattern accepts "outlier_{factor}_{num}" or "outlier_{factor}".
        pattern = r'^outlier_([0-9]*\.?[0-9]+)(?:_([0-9]+))?$'

        # Match the input string against the pattern
        match = re.match(pattern, sample_weight_type)

        if match:
            # Extract 'factor' and 'num' (if present)
            factor = float(match.group(1))  # Convert factor to float
            num = int(match.group(2)) if match.group(2) is not None else 1  # This will be None if 'num' is not present

            sample_weight = np.ones(N)
            max_values = np.partition(np.unique(x), -num)[-num:]
            min_values = np.partition(np.unique(x), num)[:num]
            sample_weight[np.argwhere(np.isin(x, max_values))] = factor
            sample_weight[np.argwhere(np.isin(x, min_values))] = factor

            return sample_weight
    elif sample_weight_type.startswith("gradual"):
        # This pattern accepts "gradual_{factor_max}_{factor_min}" or "gradual_{factor_max}".
        pattern = r'^gradual_([0-9]*\.?[0-9]+)(?:_([0-9]*\.?[0-9]+))?(?:_pow([0-9]*\.?[0-9]+))?$'

        # Match the input string against the pattern
        match = re.match(pattern, sample_weight_type)

        if match:
            # Extract 'factor_max' and 'factor_min' (if present)
            factor_max = float(match.group(1))  # Convert factor_max to float
            factor_min = float(match.group(2)) if match.group(2) is not None else 1.0
            pow = float(match.group(3)) if match.group(3) is not None else 1.0

            x_max = np.max(x)
            x_min = np.min(x)
            x_mid = (x_max + x_min) / 2
            sample_weight = (factor_max - factor_min) * (np.abs(x - x_mid) / (x_max - x_mid))**pow + factor_min

            return sample_weight.squeeze()
        else:
            # Raise Error
            raise ValueError(f"Failed to parse {sample_weight_type}")
    else:
        raise ValueError(f"Unsupported sample weight type {sample_weight_type}.")


def KMeans(x, n_clusters=10, max_iter=10, init=None, sample_weight=None, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    start = time.time()
    K = n_clusters
    N, D = x.shape  # Number of samples, dimension of the ambient space

    if init is None or init=="random":
        index = torch.randint_like(x[:K, :], low=0, high=N, dtype=int)
        c = x[index].squeeze(-1)
    else:
        c = init

    x_i = torch.Tensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = torch.Tensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(max_iter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
    xc = c[cl]

    if verbose:  # Fancy display -----------------------------------------------
        if "cuda" in x.device:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                max_iter, end - start, max_iter, (end - start) / max_iter
            )
        )

    return cl, c, xc