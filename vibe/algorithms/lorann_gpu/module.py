import numpy as np
import torch
from lorann_gpu import LorannIndex

from ..base.module import BaseANN


class LorannGPU(BaseANN):
    def __init__(self, metric, global_dim, rank, train_size, n_clusters, precision):
        self.metric = metric
        self.global_dim = global_dim
        self.rank = rank
        self.train_size = train_size
        self.n_clusters = n_clusters
        self.dtype = torch.float16 if precision == 16 else torch.float32

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self.index = LorannIndex(
            data=X,
            n_clusters=self.n_clusters,
            global_dim=self.global_dim,
            euclidean=self.metric == "euclidean",
            dtype=self.dtype,
        )
        self.index.build()

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)

        if self.metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]

        self.index = LorannIndex(
            data=X_train,
            n_clusters=self.n_clusters,
            global_dim=self.global_dim,
            euclidean=self.metric == "euclidean",
            dtype=self.dtype,
        )
        self.index.build(training_queries=X_learn)

    def set_query_arguments(self, query_args):
        self.clusters_to_search, self.points_to_rerank = query_args

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        L = self.index.search(X, n, self.clusters_to_search, self.points_to_rerank)
        self.res = L.detach().cpu().numpy()

    def get_batch_results(self):
        return [list(x[x != -1]) for x in self.res]

    def __str__(self):
        return "LorannGPU(global_dim=%d, rank=%d, n_clusters=%d, n_probes=%d, rerank=%d)" % (
            self.global_dim,
            self.rank,
            self.n_clusters,
            self.clusters_to_search,
            self.points_to_rerank,
        )
