from cuvs.neighbors import brute_force, cagra, ivf_flat, ivf_pq, refine
import numpy as np
import cupy as cp

from ..base.module import BaseANN


class cuVSBruteForce(BaseANN):
    def __init__(self, metric):
        self.metric = {
            "euclidean": "sqeuclidean",
            "cosine": "cosine",
            "ip": "inner_product",
            "normalized": "inner_product",
        }[metric]

    def fit(self, X):
        self.dataset = cp.array(X)
        self.index = brute_force.build(self.dataset, metric=self.metric)

    def batch_query(self, X, n):
        _, ids = brute_force.search(self.index, cp.array(X), n)
        self.res = cp.asarray(ids, cp.uint32).get()

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "cuVSBruteForce()"


class cuVSIVF(BaseANN):
    def __init__(self, metric, n_list):
        self.metric = {
            "euclidean": "sqeuclidean",
            "cosine": "cosine",
            "ip": "inner_product",
            "normalized": "inner_product",
        }[metric]
        self.n_list = n_list

    def fit(self, X):
        self.num_points = len(X)
        index_params = ivf_flat.IndexParams(n_lists=self.n_list, metric=self.metric)
        self.index = ivf_flat.build(index_params, cp.array(X))

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe
        self.search_params = ivf_flat.SearchParams(n_probes=n_probe)

    def batch_query(self, X, n):
        D, L = ivf_flat.search(self.search_params, self.index, cp.array(X), n)
        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[(x >= 0) & (x < self.num_points)]) for x in L]

    def __str__(self):
        return "cuVSIVF(n_list={}, n_probes={})".format(self.n_list, self.n_probe)


class cuVSIVFPQ(BaseANN):
    def __init__(self, metric, n_list, pq_dim, pq_bits):
        self.metric = {
            "euclidean": "sqeuclidean",
            "cosine": "cosine",
            "ip": "inner_product",
            "normalized": "inner_product",
        }[metric]
        self.n_list = n_list
        self.pq_dim = pq_dim
        self.pq_bits = pq_bits

    def fit(self, X):
        self.num_points = len(X)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
            metric = "sqeuclidean"
        else:
            metric = self.metric

        n, d = X.shape
        index_params = ivf_pq.IndexParams(
            n_lists=self.n_list, metric=metric, pq_dim=self.pq_dim, pq_bits=self.pq_bits
        )
        self.dataset = cp.array(X)
        self.index = ivf_pq.build(index_params, self.dataset)

    def set_query_arguments(self, n_probe, lut_dtype, refine_ratio):
        self.n_probe = n_probe
        self.lut_dtype = lut_dtype
        self.refine_ratio = refine_ratio
        ldtype = np.float16 if lut_dtype == 16 else np.float32

        self.search_params = ivf_pq.SearchParams(n_probes=n_probe, lut_dtype=ldtype)

    def batch_query(self, X, n):
        queries = cp.array(X)
        if self.metric == "cosine":
            queries /= cp.linalg.norm(queries, axis=1)[:, cp.newaxis]

        if self.refine_ratio > 1:
            _, C = ivf_pq.search(self.search_params, self.index, queries, self.refine_ratio * n)
            D, L = refine(self.dataset, queries, C, n)
        else:
            D, L = ivf_pq.search(self.search_params, self.index, queries, n)

        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[(x >= 0) & (x < self.num_points)]) for x in L]

    def __str__(self):
        return "cuVSIVFPQ(n_list={}, pq_dim={}, pq_bits={}, n_probes={}, ldtype={}, ratio={})".format(
            self.n_list, self.pq_dim, self.pq_bits, self.n_probe, self.lut_dtype, self.refine_ratio
        )


class cuVSCAGRA(BaseANN):
    def __init__(self, metric, graph_degree, intermediate_graph_degree):
        self.metric = {
            "euclidean": "sqeuclidean",
            "cosine": "cosine",
            "ip": "inner_product",
            "normalized": "inner_product",
        }[metric]
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree

        if graph_degree > intermediate_graph_degree:
            raise Exception()

    def fit(self, X):
        self.num_points = len(X)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
            metric = "inner_product"
        else:
            metric = self.metric

        n, d = X.shape
        index_params = cagra.IndexParams(
            metric=metric,
            graph_degree=self.graph_degree,
            intermediate_graph_degree=self.intermediate_graph_degree,
            build_algo="nn_descent",
        )
        self.index = cagra.build(index_params, cp.array(X))

    def set_query_arguments(self, itopk, search_width):
        self.itopk = itopk
        self.search_width = search_width
        self.search_params = cagra.SearchParams(itopk_size=itopk, search_width=search_width)

    def batch_query(self, X, n):
        if n > self.itopk:
            self.res = (np.full((len(X), n), -1), np.full((len(X), n), -1))
            return

        queries = cp.array(X)
        if self.metric == "cosine":
            queries /= cp.linalg.norm(queries, axis=1)[:, cp.newaxis]

        D, L = cagra.search(self.search_params, self.index, queries, n)
        self.res = (cp.asarray(D).get(), cp.asarray(L).get())

    def get_batch_results(self):
        D, L = self.res
        return [list(x[(x >= 0) & (x < self.num_points)]) for x in L]

    def __str__(self):
        return "cuVSCAGRA(graph_deg={}, i_graph_deg={}, itopk={}, search_w={})".format(
            self.graph_degree, self.intermediate_graph_degree, self.itopk, self.search_width
        )
