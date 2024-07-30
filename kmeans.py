import torch
import time
from utils import log, get_max_n_numbers, get_min_n_numbers
import numpy as np
from scipy.sparse import csr_matrix
import re

from typing import Union, Optional, Callable, Tuple

nf4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

def build_init(x, n_clusters, init_type):
    K = n_clusters
    N, D = x.shape  # Number of samples, dimension of the ambient space

    match init_type:
        case None:
            return None

        case "random":
            return "random"

        case "manual_random":
            index = np.random.choice(N, K, replace=False)
            init = x[index, :]
            return init

        # TODO: change to numpy
        case "int":
            # NOTE: What I did here doesn't really make sense for D > 1.
            init = torch.zeros(K, D)
            for i in range(D):
                init[:, i] = torch.linspace(start=x[:, i].min(), end=x[:, i].max(), steps=K)
            return init

        case "pow":
            # NOTE: What I did here doesn't really make sense for D > 1.
            init = torch.zeros(K, D)
            for i in range(D):
                init[:, i] = torch.logspace(start=1, end=log(x[:, i].max()) / log(x[:, i].min()), steps=K, base=x[:, i].min())
            return init

        case "nf4":
            # NOTE: What I did here doesn't really make sense for D > 1.
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

        case _:
            raise ValueError(f"Unsupported init type {init_type}")


def build_sample_weight(x, sample_weight_type: str, abs: bool = False):
    N, _ = x.shape  # Number of samples, dimension of the ambient space

    if sample_weight_type is None:
        return None
    elif isinstance(sample_weight_type, torch.Tensor):
        sample_weight = sample_weight_type.squeeze().cpu().numpy()
        assert sample_weight.shape == (N,), f"sample_weight.shape {sample_weight.shape} should be ({N},)"
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
        else:
            raise ValueError(f"Failed to parse {sample_weight_type}.")
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

            # reduce x along features dimension
            x = np.mean(x, axis=1)

            x_max = np.max(x)
            x_min = np.min(x)
            x_mid = (x_max + x_min) / 2
            sample_weight = (factor_max - factor_min) * (np.abs(x - x_mid) / (x_max - x_mid))**pow + factor_min

            sample_weight = sample_weight.squeeze()
        else:
            raise ValueError(f"Failed to parse {sample_weight_type}.")
    else:
        raise ValueError(f"Unsupported sample weight type {sample_weight_type}.")

    if abs:
        sample_weight = sample_weight.absolute()

    return sample_weight

# Example usage:
# X = np.random.rand(100, 5)  # 100 samples, 5 features
# centroids, inertia, labels = kmeans(X, n_clusters=3, verbose=1)

def kmeans(X: Union[np.ndarray, csr_matrix], n_clusters: int = 8, init: Union[str, Callable, np.ndarray] = 'k-means++',
           n_init: Union[str, int] = 'auto', max_iter: int = 300, tol: float = 1e-4, verbose: int = 0,
           random_state: Optional[int] = None, sample_weight: Optional[np.ndarray] = None, X_surrogate: Optional[Union[np.ndarray, csr_matrix]] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Perform KMeans clustering on data.

    Args:
        X (Union[np.ndarray, csr_matrix]): Training instances to cluster.
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
        init (Union[str, Callable, np.ndarray]): Method for initialization.
        n_init (Union[str, int]): Number of times the k-means algorithm is run with different centroid seeds.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run.
        tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers.
        verbose (int): Verbosity mode.
        random_state (Optional[int]): Determines random number generation for centroid initialization.
        sample_weight (Optional[np.ndarray]): The weights for each observation in X.
        X_surrogate (Optional[Union[np.ndarray, csr_matrix]]): Optional values to use when finding centre of instances. Should have the same shape of X.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: Centroids, inertia, and labels for each sample.
    """
    if isinstance(X, csr_matrix):
        X = X.tocsr()
    else:
        X = np.ascontiguousarray(X)

    if random_state is not None:
        np.random.seed(random_state)

    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])

    if n_init == 'auto':
        n_init = 10 if init in ['random', callable] else 1

    best_inertia = np.inf
    best_centroids = None
    best_labels = None

    for _ in range(n_init):  # Multiple initializations to find best clustering
        centroids = initialize_centroids(X, n_clusters, init)
        inertia, centroids, labels = run_kmeans(X, centroids, max_iter, tol, verbose, sample_weight, X_surrogate)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids[best_labels], best_centroids, best_labels

def initialize_centroids(X: np.ndarray, n_clusters: int, init: Union[str, Callable, np.ndarray]) -> np.ndarray:
    """
    Initialize centroids for KMeans clustering.

    Args:
        X (np.ndarray): Data points.
        n_clusters (int): Number of clusters.
        init (Union[str, Callable, np.ndarray]): Initialization method.

    Returns:
        np.ndarray: Initialized centroids.
    """
    if init == 'k-means++':
        centroids = [X[np.random.randint(0, X.shape[0])]]
        for _ in range(1, n_clusters):  # k-means++ initialization
            distances = np.min(np.square(X[:, np.newaxis] - centroids).sum(axis=2), axis=1)
            probabilities = distances / distances.sum()
            centroids.append(X[np.random.choice(X.shape[0], p=probabilities)])
        return np.array(centroids)
    elif init == 'random':
        indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        return X[indices]
    elif callable(init):
        return init(X, n_clusters, np.random.RandomState())
    elif isinstance(np.ndarray):
        return np.array(init)
    else:
        raise ValueError(f"Unsupported data type for init: {type(init)}.")

def run_kmeans(X: np.ndarray, centroids: np.ndarray, max_iter: int, tol: float, verbose: int, sample_weight: np.ndarray, X_surrogate: Optional[np.ndarray] = None, sample_weight_eps: Optional[float] = 1e-6) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run the KMeans clustering algorithm.

    Args:
        X (np.ndarray): Data points.
        centroids (np.ndarray): Initial centroids.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        verbose (int): Verbosity level.
        sample_weight (np.ndarray): Weights for each sample.
        X (np.ndarray): Surrogate data points to use when finding centre of points.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Inertia, final centroids, and labels.
    """
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=int)

    for i in range(max_iter):
        # Calculate distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign labels based on closest centroid
        new_labels = np.argmin(distances, axis=1)

        # Check if labels have changed, if not, possibly break the loop
        if np.array_equal(labels, new_labels):
            if verbose:
                print(f"Convergence reached at iteration {i} (labels did not change)")
            break
        labels = new_labels

        # Compute new centroids
        for j in range(n_clusters):
            # Select data points assigned to cluster j
            cluster_points = X[labels == j] if X_surrogate is None else X_surrogate[labels == j]
            cluster_weights = sample_weight[labels == j] if sample_weight is not None else None

            # Calculate weighted average to find new centroid
            if cluster_points.size:
                centroids[j] = np.average(cluster_points, axis=0, weights=cluster_weights)

        # Check for convergence: if the centroids do not change significantly, break the loop
        if i > 0 and np.linalg.norm(centroids - old_centroids) < tol:
            if verbose:
                print(f"Convergence reached at iteration {i} (centroid positions changed less than {tol})")
            break
        old_centroids = centroids.copy()

    # Calculate inertia: sum of squared distances of samples to their closest cluster center
    inertia = np.sum((X - centroids[labels])**2)

    return inertia, centroids, labels
