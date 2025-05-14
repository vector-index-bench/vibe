import annoy
import numpy as np

from ..base.module import BaseANN


class Annoy(BaseANN):
    def __init__(self, metric, n_trees):
        self.n_trees = n_trees
        self.search_k = None
        self.metric = {
            "cosine": "angular",
            "euclidean": "euclidean",
            "ip": "dot",
            "normalized": "dot",
            "hamming": "hamming",
        }[metric]

    def fit(self, X):
        n, d = X.shape

        if self.metric == "hamming":
            X = np.unpackbits(X).reshape(n, -1).astype(np.float32)
            d *= 8

        self.annoy = annoy.AnnoyIndex(d, metric=self.metric)
        for i, x in enumerate(X):
            self.annoy.add_item(i, x.tolist())

        self.annoy.build(self.n_trees, 1)

    def set_query_arguments(self, search_k):
        self.search_k = search_k

    def query(self, v, n):
        if self.metric == "hamming":
            v = np.unpackbits(v).astype(np.float32)
        return self.annoy.get_nns_by_vector(v.tolist(), n, self.search_k)

    def __str__(self):
        return "Annoy(n_trees=%d, search_k=%d)" % (self.n_trees, self.search_k)
