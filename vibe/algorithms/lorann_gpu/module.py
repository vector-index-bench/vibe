import numpy as np
import functools

from ..base.module import BaseANN


class LorannGPU(BaseANN):

    def __init__(self, metric, global_dim, rank, train_size, n_clusters, precision):
        import lorann_gpu.jax
        import jax.numpy as jnp

        dtype = jnp.float16 if precision == 16 else jnp.float32

        self.metric = metric
        self.global_dim = global_dim
        self.rank = rank
        self.train_size = train_size
        self.n_clusters = n_clusters
        self.index_type = functools.partial(lorann_gpu.jax.Lorann.build, data_dtype=jnp.float16, dtype=dtype)

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self.index = self.index_type(
            X, self.n_clusters, self.global_dim, self.rank, self.train_size, self.metric == "euclidean"
        )

    def set_query_arguments(self, clusters_to_search, points_to_rerank):
        import jax

        jax.clear_caches()

        self.clusters_to_search, self.points_to_rerank = clusters_to_search, points_to_rerank

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.res = self.index.search(X, n, self.clusters_to_search, self.points_to_rerank)

    def get_batch_results(self):
        return [list(x[x != -1]) for x in self.res]

    def __str__(self):
        str_template = "LorannGPU(gd=%d, r=%d, nc=%d, cs=%d, pr=%d)"
        return str_template % (
            self.global_dim,
            self.rank,
            self.n_clusters,
            self.clusters_to_search,
            self.points_to_rerank,
        )
