import numpy as np
import scann

from ..base.module import BaseANN


class Scann(BaseANN):
    def __init__(self, metric, n_leaves, avq_threshold, dims_per_block):
        self.metric = metric
        self.n_leaves = n_leaves
        self.avq_threshold = avq_threshold
        self.dims_per_block = dims_per_block

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
            spherical = True
            dist = "dot_product"
        elif self.metric == "ip" or self.metric == "normalized":
            spherical = True
            dist = "dot_product"
        else:
            spherical = False
            dist = "squared_l2"

        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, dist)
            .tree(self.n_leaves, 1, training_sample_size=len(X), spherical=spherical, quantize_centroids=True)
            .score_ah(self.dims_per_block, anisotropic_quantization_threshold=self.avq_threshold)
            .reorder(1)
            .build()
        )

    def set_query_arguments(self, leaves_to_search, reorder):
        self.leaves_to_search = leaves_to_search
        self.reorder = reorder

    def query(self, v, n):
        return self.searcher.search(v, n, self.reorder, self.leaves_to_search)[0]

    def __str__(self):
        return "ScaNN(n_leaves=%d, avg_threshold=%g, dims_per_block=%d, leaves_to_search=%d, reorder=%d)" % (
            self.n_leaves,
            self.avq_threshold,
            self.dims_per_block,
            self.leaves_to_search,
            self.reorder,
        )
