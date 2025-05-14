import pynndescent

from ..base.module import BaseANN


class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors, pruning_degree_multiplier, diversify_prob):
        self._n_neighbors = n_neighbors
        self._pruning_degree_multiplier = pruning_degree_multiplier
        self._diversify_prob = diversify_prob

        self._metric = {
            "cosine": "dot",
            "euclidean": "euclidean",
            "normalized": "dot",
            "hamming": "bit_hamming",
        }[metric]

    def fit(self, X):
        self._index = pynndescent.NNDescent(
            X,
            n_neighbors=self._n_neighbors,
            metric=self._metric,
            low_memory=True,
            pruning_degree_multiplier=self._pruning_degree_multiplier,
            diversify_prob=self._diversify_prob,
            n_jobs=1,
            verbose=False,
        )

        if hasattr(self._index, "prepare"):
            self._index.prepare()
        else:
            self._index._init_search_graph()
            if hasattr(self._index, "_init_search_function"):
                self._index._init_search_function()

    def set_query_arguments(self, epsilon):
        self._epsilon = float(epsilon)

    def query(self, v, n):
        ind, dist = self._index.query(v.reshape(1, -1), k=n, epsilon=self._epsilon)
        return ind[0]

    def __str__(self):
        str_template = "PyNNDescent(n_neighbors=%d, pruning_mult=%.2f, diversify_prob=%.3f, epsilon=%.3f)"
        return str_template % (
            self._n_neighbors,
            self._pruning_degree_multiplier,
            self._diversify_prob,
            self._epsilon,
        )
