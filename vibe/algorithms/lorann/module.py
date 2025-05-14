import lorannlib
import numpy as np

from ..base.module import BaseANN


class Lorann(BaseANN):

    def __init__(self, metric, quantization_bits, n_clusters, global_dim, rank, train_size):
        self.metric = metric
        self.quantization_bits = quantization_bits
        self.n_clusters = n_clusters
        self.global_dim = global_dim
        self.rank = rank
        self.train_size = train_size

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        n_samples, dim = X.shape

        if self.global_dim > dim:
            raise ValueError(f"LoRANN: global_dim ({self.global_dim}) larger than data dim ({dim})")

        self.index = lorannlib.LorannIndex(
            X,
            n_samples,
            dim,
            self.quantization_bits,
            self.n_clusters,
            self.global_dim,
            self.rank,
            self.train_size,
            self.metric == "euclidean",
            False,
        )
        self.index.build(True, 1)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)

        if self.metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]

        n_samples, dim = X_train.shape
        self.index = lorannlib.LorannIndex(
            X_train,
            n_samples,
            dim,
            self.quantization_bits,
            self.n_clusters,
            self.global_dim,
            self.rank,
            self.train_size,
            self.metric == "euclidean",
            False,
        )
        self.index.build(True, 1, X_learn)

    def set_query_arguments(self, query_args):
        self.clusters_to_search, self.points_to_rerank = query_args

    def query(self, q, n):
        return self.index.search(q, n, self.clusters_to_search, self.points_to_rerank, False, 1)

    def __str__(self):
        str_template = "Lorann(q=%d, nc=%d, gd=%d, r=%d, ts=%d, cs=%d, pr=%d)"
        return str_template % (
            self.quantization_bits,
            self.n_clusters,
            self.global_dim,
            self.rank,
            self.train_size,
            self.clusters_to_search,
            self.points_to_rerank,
        )
