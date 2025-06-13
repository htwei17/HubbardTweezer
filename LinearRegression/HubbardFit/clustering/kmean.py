from typing import Iterable, Union
from typing import Optional
import torch
import numpy as np


def seeded_kmeans(
    X: torch.Tensor,
    k: int = 2,
    num_iters: int = 200,
    tol: float = 1e-4,
    init: Optional[Union[torch.Tensor, Iterable]] = None,
    prune_percentile: int = 95,
    silhouette: float = -1,
):
    """
    Minimal k-means in PyTorch.
    Args
    ----
    X          : (N, D) data tensor (float32/float64) on any device
                 N = number of samples, D = number of features
    k          : number of clusters
    num_iters  : max EM iterations
    tol        : stop if centroid shift < tol         (optional)
    init       : initial centroids, can be:
                 - None: random sample k points from X
                 - (k, D) tensor: initial centroids
                 - Iterable of length k: initial centroids
    prune_percentile: distance-percentile to drop from *good* cluster (id 0)
    silhouette:  float, use per-point silhouette < s to prune instead, -1 to skip
    Returns
    -------
    centroids  : (k, D) tensor of final centroids
    labels     : (N,)  tensor of cluster IDs for each sample
    """
    N, D = X.shape
    # ---- 1. initial centroids --------------------------------------------
    if init is None:
        centroids = X[torch.randperm(N)[:k]]
    else:
        if isinstance(init, (np.ndarray, torch.Tensor)):
            m = init.shape[0]
        elif isinstance(init, Iterable):
            m = len(init)
            init = torch.tensor(init, dtype=X.dtype, device=X.device).reshape(m, D)
        pad = X[torch.randperm(N)[: k - m]]
        centroids = torch.vstack([init, pad]).clone()

    # ---- 2. Lloyd iterations ---------------------------------------------
    for _ in range(num_iters):
        dists = torch.cdist(X, centroids)
        labels = dists.argmin(1)
        # recompute each centroid; keep old if empty
        new_centroids = torch.stack(
            [
                (
                    X[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
                )  # avoid empty cluster crash
                for i in range(k)
            ]
        )
        # convergence test
        if torch.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    # ---- 3. outlier pruning inside *good* cluster (cluster 0) ------------
    good_mask = labels == 0
    d_good = torch.cdist(X[good_mask], centroids[0][None]).squeeze(1)

    if silhouette >= 0:  # option B: silhouette rule
        # quick 2-cluster silhouette approx
        a = d_good  # intra-cluster distances
        # inter-cluster distances
        b = torch.cdist(X[good_mask], centroids[1][None]).squeeze(1)
        s = (b - a) / torch.maximum(a, b)
        prune = s < silhouette  # border or mis-assigned
    else:  # option A: percentile rule
        thr = torch.quantile(d_good, prune_percentile / 100.0)
        prune = d_good > thr

    pruned_idx = good_mask.nonzero().squeeze(1)[prune]
    labels[pruned_idx] = -1  # mark as outlier
    return centroids, labels, pruned_idx
