import mlann
import numpy as np

from ..base.module import BaseANN


class MLANN(BaseANN):
    def __init__(self, metric, count, index_type, n_trees_max):
        self.metric = metric
        self.k = count
        self.index_type = index_type
        self.n_trees_max = n_trees_max

        if metric in ["cosine", "ip", "normalized"]:
            self.dist = mlann.IP
        else:
            self.dist = mlann.L2

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_learn = np.ascontiguousarray(X_learn, dtype=np.float32)

        if self.metric == "cosine":
            X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
            X_learn = X_learn / np.linalg.norm(X_learn, axis=1, keepdims=True)

        index = mlann.MLANNIndex(X_train, self.index_type)
        self.profile = index.autotune(
            X_learn, X_learn_neighbors, k=self.k, target_recall=None, n_trees_max=self.n_trees_max, dist=self.dist
        )

    def set_query_arguments(self, target_recall):
        self.index = self.profile.subset(target_recall)

    def query(self, v, n):
        v = np.ascontiguousarray(v, dtype=np.float32)
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        ids = self.index.ann(v, n)
        return ids[ids >= 0]

    def __str__(self):
        return f"MLANN(target_recall={self.index.autotune_result.target_recall:g})"
