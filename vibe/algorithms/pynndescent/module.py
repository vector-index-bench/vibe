import pynndescent

from ..base.module import BaseANN


class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors, pruning_degree_multiplier, diversify_prob):
        self.n_neighbors = n_neighbors
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.diversify_prob = diversify_prob

        self.metric = {
            "cosine": "dot",
            "euclidean": "euclidean",
            "normalized": "dot",
            "hamming": "bit_hamming",
        }[metric]

    def fit(self, X):
        self.index = pynndescent.NNDescent(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            low_memory=True,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
            diversify_prob=self.diversify_prob,
            n_jobs=1,
            verbose=False,
        )

        if hasattr(self.index, "prepare"):
            self.index.prepare()
        else:
            self.index._init_search_graph()
            if hasattr(self.index, "_init_search_function"):
                self.index._init_search_function()

    def set_query_arguments(self, epsilon):
        self.epsilon = float(epsilon)

    def query(self, v, n):
        ind, dist = self.index.query(v.reshape(1, -1), k=n, epsilon=self.epsilon)
        return ind[0]

    def __str__(self):
        return "PyNNDescent(n_neighbors=%d, pruning_mult=%.2f, diversify_prob=%.3f, epsilon=%.3f)" % (
            self.n_neighbors,
            self.pruning_degree_multiplier,
            self.diversify_prob,
            self.epsilon,
        )
