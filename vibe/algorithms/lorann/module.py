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
        n_samples, dim = X.shape

        if self.metric == "hamming":
            index_type = lorannlib.BinaryLorannIndex
            dim *= 8
        elif X.dtype == np.uint8:
            index_type = lorannlib.U8LorannIndex
        else:
            index_type = lorannlib.FP32LorannIndex

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
            distance = lorannlib.IP
        elif self.metric == "euclidean":
            distance = lorannlib.L2
        elif self.metric == "hamming":
            distance = lorannlib.HAMMING
        else:
            distance = lorannlib.IP

        if self.global_dim > dim:
            raise ValueError(f"LoRANN: global_dim ({self.global_dim}) larger than data dim ({dim})")

        self.index = index_type(
            X,
            n_samples,
            dim,
            self.quantization_bits,
            self.n_clusters,
            self.global_dim,
            self.rank,
            self.train_size,
            distance,
            False,
            False,
        )
        self.index.build(True, False, 1)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        n_samples, dim = X_train.shape

        if self.metric == "hamming":
            index_type = lorannlib.BinaryLorannIndex
            dim *= 8
        elif X_train.dtype == np.uint8:
            index_type = lorannlib.U8LorannIndex
        else:
            index_type = lorannlib.FP32LorannIndex

        if self.metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
            distance = lorannlib.IP
        elif self.metric == "euclidean":
            distance = lorannlib.L2
        elif self.metric == "hamming":
            distance = lorannlib.HAMMING
        else:
            distance = lorannlib.IP

        if self.global_dim > dim:
            raise ValueError(f"LoRANN: global_dim ({self.global_dim}) larger than data dim ({dim})")

        self.index = index_type(
            X_train,
            n_samples,
            dim,
            self.quantization_bits,
            self.n_clusters,
            self.global_dim,
            self.rank,
            self.train_size,
            distance,
            False,
            False,
        )
        self.index.build(True, False, 1, X_learn)

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
