"""Clustering utilities for panel time-series dataframes."""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform, pdist, cosine, cdist

def energy_distance_multivariate(X1, X2):
    """Energy distance between two multivariate samples."""
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    d_xx = cdist(X1, X1)
    d_yy = cdist(X2, X2)
    d_xy = cdist(X1, X2)
    return np.sqrt(max(2 * d_xy.mean() - d_xx.mean() - d_yy.mean(), 0))

def calculate_distances(df, metric="cosine"):
    """Pairwise distances between columns."""
    return pdist(df.T.values, metric=metric)

def find_pairs(distances_matrix):
    """Returns closest and furthest users."""
    size = distances_matrix.shape
    na, ma = np.unravel_index(np.argmin(distances_matrix + np.identity(size[0]), axis=None), size)
    nb, mb = np.unravel_index(np.argmax(distances_matrix, axis=None), size)
    return na, nb, ma, mb

def init_clusters(df):
    """Initializes hierarchical clustering linkage and distances matrix."""
    distances = calculate_distances(df)
    Z = shc.linkage(distances, method="average")
    return Z, squareform(distances)

def get_clusters(Z, n_clusters):
    """Computes flat clusters from linkage."""
    labels = shc.fcluster(Z, n_clusters, criterion="maxclust")
    cluster_indices = [np.where(labels == i)[0] for i in range(1, n_clusters + 1)]
    return labels, cluster_indices

def get_centroids(df, cluster_indices):
    """Computes per-cluster centroids."""
    return [df.iloc[:, idx].mean(axis=1) for idx in cluster_indices]

def get_cluster_dicts(df, cluster_indices):
    """Builds dict of cluster->sub-dataframe."""
    return {f"cluster_{i}": df.iloc[:, idx] for i, idx in enumerate(cluster_indices)}

def get_cluster_distances(df, cluster_indices):
    """Computes intra- and inter-cluster cosine distances."""
    intra_distances, inter_distances = {}, {}
    centroids = get_centroids(df, cluster_indices)

    for i, idx in enumerate(cluster_indices):
        if len(idx) > 1:
            d = []
            for j in range(len(idx)):
                for k in range(j + 1, len(idx)):
                    d.append(cosine(df.iloc[:, idx[j]].values, df.iloc[:, idx[k]].values))
            intra_distances[i] = np.mean(d)
        else:
            intra_distances[i] = np.nan

        if len(cluster_indices) > 1:
            for j in range(i + 1, len(cluster_indices)):
                inter_distances[(i, j)] = cosine(centroids[i].values, centroids[j].values)
        else:
            inter_distances = {0: 0}

    return intra_distances, inter_distances

def get_cluster_heterogeneity(df, cluster_indices):
    """Returns heterogeneity proxy from intra/inter distances."""
    intra_distances, inter_distances = get_cluster_distances(df, cluster_indices)
    intra = list(intra_distances.values())
    inter = list(inter_distances.values())
    return np.nanmean(intra) / (np.mean(inter) + 1) if len(inter) > 0 else np.nanmean(intra)
