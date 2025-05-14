import faiss
import numpy as np

from ..base.module import BaseANN

# Implementation based on
# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa


class FaissGPU(BaseANN):

    def __init__(self, metric):
        self.metric = metric
        self.res = faiss.StandardGpuResources()
        self.index = None

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.res = self.index.search(X, n)

    def get_batch_results(self):
        D, L = self.res
        return [list(x[x != -1]) for x in L]


class FaissGPUIVF(FaissGPU):
    def __init__(self, metric, n_list, float16):
        super().__init__(metric)
        self.n_list = n_list
        self._float16 = float16

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        n, d = X.shape
        index = faiss.index_factory(d, f"IVF{self.n_list},Flat", faiss_metric)

        co = faiss.GpuClonerOptions()
        co.useFloat16 = bool(self._float16)

        self.index = faiss.index_cpu_to_gpu(self.res, 0, index, co)
        self.index.train(X)
        self.index.add(X)

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe
        self.index.nprobe = n_probe

    def __str__(self):
        return "FaissGPUIVF(n_list={}, n_probes={}, float16={})".format(self.n_list, self.n_probe, self._float16)


class FaissGPUIVFPQ(FaissGPU):
    def __init__(self, metric, n_list, code_size):
        super().__init__(metric)
        self.n_list = n_list
        self.code_size = code_size

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        n, d = X.shape
        index = faiss.index_factory(d, f"IVF{self.n_list},PQ{self.code_size}", faiss_metric)

        co = faiss.GpuClonerOptions()
        # GpuIndexIVFPQ with 56 bytes per code or more requires use of the
        # float16 IVFPQ mode due to shared memory limitations
        co.useFloat16 = True

        self.index = faiss.index_cpu_to_gpu(self.res, 0, index, co)
        self.index.train(X)
        self.index.add(X)

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe
        self.index.nprobe = n_probe

    def __str__(self):
        return "FaissGPUIVFPQ(n_list={}, n_probes={}, code_size={})".format(self.n_list, self.n_probe, self.code_size)
