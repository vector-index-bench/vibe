import symphonyqg
import numpy as np

from ..base.module import BaseANN


class SymphonyQG(BaseANN):
    def __init__(self, metric, degree_bound, EF, num_iter):
        if metric not in ["euclidean", "cosine", "normalized"]:
            raise NotImplementedError(f"Metric {metric} not implemented for SymphonyQG")

        self.metric = metric
        self.degree_bound = degree_bound
        self.EF = EF
        self.num_iter = num_iter

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        N, D = X.shape
        if D > 1024:
            raise NotImplementedError("Dimensionality > 1024 not supported by SymphonyQG")

        self.index = symphonyqg.Index("QG", "L2", num_elements=N, dimension=D, degree_bound=self.degree_bound)
        self.index.build_index(X, self.EF, num_iter=self.num_iter, num_thread=1)

    def set_query_arguments(self, beam_size):
        self.beam_size = beam_size
        self.index.set_ef(beam_size)

    def query(self, v, n):
        if self.metric == "cosine":
            v /= np.linalg.norm(v)
        return self.index.search(v, n)

    def __str__(self):
        return "SymphonyQG(degree_bound=%d, EF=%d, num_iter=%d, beam_size=%d)" % (
            self.degree_bound,
            self.EF,
            self.num_iter,
            self.beam_size,
        )
