import numpy as np
import torch

import jasper

from ..base.module import BaseANN


class Jasper(BaseANN):
    def __init__(self, metric, n_neighbors, alpha):
        self.metric = metric
        self.distance = {"euclidean": "l2", "cosine": "ip", "ip": "ip", "normalized": "ip"}[metric]
        self.n_neighbors = n_neighbors
        self.alpha = alpha

    def _to_device(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        X = torch.tensor(X, device="cuda")
        if self.metric == "cosine":
            X = X / X.norm(dim=1, keepdim=True)
        return X.to(torch.float16)

    def fit(self, X):
        self.num_points = len(X)
        X_tensor = self._to_device(X)
        self.graph = jasper.Graph.build(
            X_tensor, n_neighbors=self.n_neighbors, distance=self.distance, alpha=self.alpha
        )

    def set_query_arguments(self, beam_width):
        self.beam_width = beam_width

    def batch_query(self, X, n):
        X_tensor = self._to_device(X)
        indices, _ = self.graph.search(X_tensor, k=n, beam_width=self.beam_width)
        self.res = indices.detach().cpu().numpy()

    def get_batch_results(self):
        return [list(x[(x >= 0) & (x < self.num_points)]) for x in self.res]

    def __str__(self):
        return "Jasper(n_neighbors=%d, alpha=%g, beam_width=%d)" % (
            self.n_neighbors,
            self.alpha,
            self.beam_width,
        )


class JasperDirectional(BaseANN):
    def __init__(self, metric, search_type, n_neighbors, alpha, k_ranks):
        self.metric = metric
        self.distance = {"euclidean": "l2", "cosine": "ip", "ip": "ip", "normalized": "ip"}[metric]
        self.search_type = search_type
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.k_ranks = k_ranks

    def _to_device(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        X = torch.tensor(X, device="cuda")
        if self.metric == "cosine":
            X = X / X.norm(dim=1, keepdim=True)
        return X.to(torch.float16)

    def fit(self, X):
        self.num_points = len(X)
        X_tensor = self._to_device(X)
        build_kwargs = {"build_lsh": True} if self.search_type == "cph" else {"build_pq": True}
        self.graph = jasper.Graph.build(
            X_tensor,
            n_neighbors=self.n_neighbors,
            distance=self.distance,
            alpha=self.alpha,
            k_ranks=self.k_ranks,
            **build_kwargs,
        )

    def set_query_arguments(self, beam_width):
        self.beam_width = beam_width

    def batch_query(self, X, n):
        X_tensor = self._to_device(X)
        search_fn = self.graph.directional_search if self.search_type == "cph" else self.graph.pq_search
        indices, _ = search_fn(X_tensor, k=n, beam_width=self.beam_width)
        self.res = indices.detach().cpu().numpy()

    def get_batch_results(self):
        return [list(x[(x >= 0) & (x < self.num_points)]) for x in self.res]

    def __str__(self):
        return "JasperDirectional(search_type=%s, n_neighbors=%d, alpha=%g, k_ranks=%d, beam_width=%d)" % (
            self.search_type,
            self.n_neighbors,
            self.alpha,
            self.k_ranks,
            self.beam_width,
        )
