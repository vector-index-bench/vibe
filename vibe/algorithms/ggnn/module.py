import ggnn
import torch
import numpy as np

from ..base.module import BaseANN


class GGNN(BaseANN):
    def __init__(self, metric, k_build, tau_build):
        if metric == "cosine":
            self.metric = ggnn.DistanceMeasure.Cosine
        elif metric == "euclidean" or metric == "normalized":
            self.metric = ggnn.DistanceMeasure.Euclidean
        else:
            raise NotImplementedError(f"Metric {metric} not supported for GGNN")

        self.k_build = k_build
        self.tau_build = tau_build

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        X_tensor = torch.tensor(X, device="cpu")

        self.index = ggnn.GGNN()
        self.index.set_base(X_tensor)
        self.index.build(k_build=self.k_build, tau_build=self.tau_build, refinement_iterations=2, measure=self.metric)

    def batch_query(self, X, n):
        L, D = self.index.query(X, n, self.tau_query, self.max_iterations, self.metric)
        self.res = L.detach().cpu().numpy()

    def get_batch_results(self):
        return [list(x[x != -1]) for x in self.res]

    def set_query_arguments(self, tau_query, max_iterations):
        self.tau_query = tau_query
        self.max_iterations = max_iterations

    def __str__(self):
        return "GGNN(k_build=%d, tau_build=%g, tau_query=%g, max_iterations=%d)" % (
            self.k_build,
            self.tau_build,
            self.tau_query,
            self.max_iterations,
        )
