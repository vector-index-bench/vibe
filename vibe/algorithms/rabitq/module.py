import numpy as np

from rabitqlib import HnswIndex, IvfIndex, SymqgIndex

from ..base.module import BaseANN


def _as_float32(X):
    return np.ascontiguousarray(X, dtype=np.float32)


def _normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _metric_name(metric):
    if metric == "euclidean":
        return "l2"
    if metric in ["cosine", "ip", "normalized"]:
        return "ip"
    raise NotImplementedError(f"RaBitQ does not support metric {metric}")


def _cluster_data(X, num_clusters, metric, num_threads):
    import faiss

    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)

    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index = faiss.index_factory(X.shape[1], f"IVF{num_clusters},Flat", faiss_metric)
    index.train(X)

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    _, cluster_ids = index.quantizer.search(X, 1)
    return _as_float32(centroids), np.ravel(cluster_ids).astype(np.uint32)


class _RabitqBase(BaseANN):
    def __init__(self, metric):
        self.metric = metric
        self.rabitq_metric = _metric_name(metric)

    def _prepare_data(self, X):
        X = _as_float32(X)
        if self.metric == "cosine":
            X = _normalize(X)
        return X

    def _prepare_queries(self, X):
        X = _as_float32(X)
        if self.metric == "cosine":
            X = _normalize(X)
        return X

    def _prepare_query(self, v):
        return self._prepare_queries(np.expand_dims(v, axis=0))

    def get_batch_results(self):
        return self.res


class RabitqIVF(_RabitqBase):
    def __init__(self, metric, num_clusters, total_bits, fast_quantization=True):
        super().__init__(metric)
        self.num_clusters = num_clusters
        self.total_bits = total_bits
        self.fast_quantization = fast_quantization
        self.nprobe = 1
        self.high_accuracy = True

    def fit(self, X):
        X = self._prepare_data(X)
        centroids, cluster_ids = _cluster_data(X, self.num_clusters, self.rabitq_metric, 1)
        self.index = IvfIndex(
            dim=X.shape[1],
            max_elements=X.shape[0],
            num_clusters=self.num_clusters,
            nbits=self.total_bits,
            metric=self.rabitq_metric,
        )
        self.index.build(X, centroids, cluster_ids, num_threads=1, fast_quantization=self.fast_quantization)

    def set_query_arguments(self, nprobe, high_accuracy=True):
        self.nprobe = nprobe
        self.high_accuracy = high_accuracy

    def query(self, v, n):
        ids, _ = self.index.search(
            self._prepare_query(v),
            k=n,
            nprobe=self.nprobe,
            high_accuracy=self.high_accuracy,
            num_threads=1,
        )
        return ids[0]

    def batch_query(self, X, n):
        self.res, _ = self.index.search(
            self._prepare_queries(X),
            k=n,
            nprobe=self.nprobe,
            high_accuracy=self.high_accuracy,
            num_threads=1,
        )

    def __str__(self):
        return "RabitqIVF(num_clusters=%d, total_bits=%d, nprobe=%d, high_accuracy=%s)" % (
            self.num_clusters,
            self.total_bits,
            self.nprobe,
            self.high_accuracy,
        )


class RabitqHNSW(_RabitqBase):
    def __init__(self, metric, num_clusters, M, efConstruction, total_bits, fast_quantization=False):
        super().__init__(metric)
        self.num_clusters = num_clusters
        self.M = M
        self.efConstruction = efConstruction
        self.total_bits = total_bits
        self.fast_quantization = fast_quantization
        self.ef = 10

    def fit(self, X):
        X = self._prepare_data(X)
        centroids, cluster_ids = _cluster_data(X, self.num_clusters, self.rabitq_metric, 1)
        self.index = HnswIndex(
            dim=X.shape[1],
            max_elements=X.shape[0],
            M=self.M,
            ef_construction=self.efConstruction,
            nbits=self.total_bits,
            metric=self.rabitq_metric,
        )
        self.index.build(X, centroids, cluster_ids, num_threads=1, fast_quantization=self.fast_quantization)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        ids, _ = self.index.search(self._prepare_query(v), k=n, ef=self.ef, num_threads=1)
        return ids[0]

    def batch_query(self, X, n):
        self.res, _ = self.index.search(self._prepare_queries(X), k=n, ef=self.ef, num_threads=1)

    def __str__(self):
        return "RabitqHNSW(num_clusters=%d, M=%d, efConstruction=%d, total_bits=%d, ef=%d)" % (
            self.num_clusters,
            self.M,
            self.efConstruction,
            self.total_bits,
            self.ef,
        )


class SymphonyQG(_RabitqBase):
    def __init__(self, metric, max_degree, efConstruction):
        super().__init__(metric)
        self.max_degree = max_degree
        self.efConstruction = efConstruction
        self.ef = 10

    def fit(self, X):
        X = self._prepare_data(X)
        self.index = SymqgIndex(dim=X.shape[1], max_degree=self.max_degree, metric=self.rabitq_metric)
        self.index.build(X, ef_construction=self.efConstruction, num_threads=1)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        ids, _ = self.index.search(self._prepare_query(v), k=n, ef=self.ef, num_threads=1)
        return ids[0]

    def batch_query(self, X, n):
        self.res, _ = self.index.search(self._prepare_queries(X), k=n, ef=self.ef, num_threads=1)

    def __str__(self):
        return "SymphonyQG(max_degree=%d, efConstruction=%d, ef=%d)" % (
            self.max_degree,
            self.efConstruction,
            self.ef,
        )
