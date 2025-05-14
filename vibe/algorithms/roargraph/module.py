import numpy as np
from ..base.module import BaseANN
from RoarGraph import IndexRoarGraph, Metric


class RoarGraph(BaseANN):
    def __init__(self, metric, M_sq, M_pjbp, L_pjpq):
        if metric not in ["cosine", "ip", "normalized"]:
            raise NotImplementedError(f"RoarGraph does not support metric {metric}")

        self.metric = metric
        self.M_sq = M_sq
        self.M_pjbp = M_pjbp
        self.L_pjpq = L_pjpq

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        n, dimension = X_train.shape
        sq_num, k_dim = X_learn_neighbors.shape
        num_threads = 1

        if self.metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]

        self.index = IndexRoarGraph(dimension, n + sq_num, Metric.IP)
        self.index.setThreads(num_threads)

        self.index.build(sq_num, k_dim, n, self.M_sq, self.M_pjbp, self.L_pjpq, num_threads, X_learn_neighbors, X_train)

    def set_query_arguments(self, L_pq):
        self.L_pq = L_pq

    def query(self, q, k):
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)

        result_ids = np.zeros(k, dtype=np.uint32)
        result_distances = np.zeros(k, dtype=np.float32)
        self.index.search(q, k, self.L_pq, result_ids, result_distances, 1, 1, False)
        return result_ids

    def __str__(self):
        str_template = "RoarGraph(%d, %d, %d, %d)"
        return str_template % (
            self.M_sq,
            self.M_pjbp,
            self.L_pjpq,
            self.L_pq,
        )
