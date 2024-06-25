import torch
import time

def KMeans(x, n_clusters=10, max_iter=10, verbose=False):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    K = n_clusters

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # c = x[:K, :].clone()  # Simplistic initialization for the centroids
    index = torch.randint_like(x[:K, :], low=0, high=N, dtype=int)
    c = x[index].squeeze(-1)

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