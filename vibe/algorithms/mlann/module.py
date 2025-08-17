import mlann
import numpy as np

from ..base.module import BaseANN


class MLANN(BaseANN):
    def __init__(self, metric, index_type, n_trees, depth):
        self.metric = metric
        self.index_type = index_type
        self.n_trees = n_trees
        self.depth = depth

        if metric in ["cosine", "ip", "normalized"]:
            self.dist = mlann.IP
        else:
            self.dist = mlann.L2

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)
        if X_learn.dtype != np.float32:
            X_learn = X_learn.astype(np.float32)

        if self.metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
            X_learn /= np.linalg.norm(X_learn, axis=1)[:, np.newaxis]

        self.index = mlann.MLANNIndex(X_train, self.index_type)
        self.index.build(X_learn, X_learn_neighbors, self.n_trees, self.depth)

    def set_query_arguments(self, voting_threshold):
        self.voting_threshold = voting_threshold

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.index.ann(v, n, self.voting_threshold, self.dist, return_distances=False)

    def __str__(self):
        return "MLANN(type=%s, n_trees=%d, depth=%d, voting_thresh=%g)" % (
            self.index_type,
            self.n_trees,
            self.depth,
            self.voting_threshold,
        )
