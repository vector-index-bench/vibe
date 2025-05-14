import faiss
import numpy as np

from ..base.module import BaseANN
from ...distance import metrics


class Faiss(BaseANN):
    def query(self, v, n):
        q = np.expand_dims(v, axis=0)
        if q.dtype != np.float32:
            q = q.astype(np.float32)
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)
        _, I = self.index.search(q, n)
        return I[0]

    def batch_query(self, X, n):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        self.res = self.index.search(X, n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(L)):
            res.append([l for l in L[i] if l != -1])
        return np.array(res)


class FaissFlat(Faiss):
    def __init__(self, metric):
        self.metric = metric

    def fit(self, X):
        self.data = X
        self.ensure_float = True

        d = X.shape[1]
        if X.dtype == np.float32:
            if self.metric == "ip" or self.metric == "normalized":
                self.index = faiss.IndexFlatIP(d)
            elif self.metric == "cosine":
                faiss.normalize_L2(X)
                self.index = faiss.IndexFlatIP(d)
            elif self.metric == "euclidean":
                self.index = faiss.IndexFlatL2(d)
            else:
                raise ValueError("unsupported metric:", self.metric)
            self.index.add(X)
        elif X.dtype == np.uint8:
            if self.metric == "euclidean":
                self.index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit_direct)
                X_float = X.astype(np.float32)
                self.index.train(X_float)
                self.index.add(X_float)
            elif self.metric == "hamming":
                self.index = faiss.IndexBinaryFlat(8 * d)
                self.index.add(X)
                self.ensure_float = False
            else:
                raise ValueError("Unsupported metric for FaissFlat:", self.metric)
        else:
            raise ValueError("Unsupported dtype for FaissFlat:", X.dtype)

    def query(self, v, n):
        q = np.expand_dims(v, axis=0)
        if self.ensure_float and q.dtype != np.float32:
            q = q.astype(np.float32)
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)
        _, I = self.index.search(q, n)
        return I[0]

    def query_with_distances(self, v, n, return_avg_dist=False):
        q = np.expand_dims(v, axis=0)

        if self.ensure_float and q.dtype != np.float32:
            q = q.astype(np.float32)
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)

        D, I = self.index.search(q, self.index.ntotal)
        D = D[0]
        I = I[0][:n]

        def fix(index):
            ep = self.data[index]
            ev = v
            return (index, metrics[self.metric].distance(ep, ev))

        if return_avg_dist:
            if self.metric == "euclidean":
                D = np.sqrt(D)
            elif self.metric == "cosine" or self.metric == "normalized":
                D = 1 - D
            elif self.metric == "ip":
                D = -D

            avg_dist = D.mean()
            return map(fix, I), avg_dist

        return map(fix, I)

    def batch_query(self, X, n):
        if self.ensure_float and X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        self.res = self.index.search(X, n)


class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self.metric = metric
        self.n_list = n_list

    def fit(self, X):
        d = X.shape[1]
        if self.metric in ["cosine", "ip", "normalized"]:
            quantizer = faiss.IndexFlatIP(d)
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(d)
            faiss_metric = faiss.METRIC_L2

        self.index = faiss.IndexIVFFlat(quantizer, d, self.n_list, faiss_metric)

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.index.train(X)
        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self.n_probe = n_probe
        self.index.nprobe = self.n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self.n_list}  # noqa

    def __str__(self):
        return "FaissIVF(n_list=%d, n_probe=%d)" % (self.n_list, self.n_probe)


class FaissIVFSQ(Faiss):
    def __init__(self, metric, n_list):
        self.metric = metric
        self.n_list = n_list

    def fit(self, X):
        d = X.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFScalarQuantizer(quantizer, d, self.n_list, faiss.ScalarQuantizer.QT_8bit_direct)
        self.index.by_residual = False

        self.index.train(X)
        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe
        self.index.nprobe = self.n_probe

    def __str__(self):
        return "FaissIVFSQ(n_list=%d, n_probe=%d)" % (self.n_list, self.n_probe)


class FaissIVFPQfs(Faiss):
    def __init__(self, metric, n_list, dim_reduction):
        self.metric = metric
        self.n_list = n_list
        self._dim_reduction = dim_reduction

    def fit(self, X):
        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        d = X.shape[1]
        if self._dim_reduction > 0:
            factory_string = f"OPQ{self._dim_reduction // 2}_{self._dim_reduction},IVF{self.n_list},PQ{self._dim_reduction // 2}x4fsr"
        else:
            factory_string = f"IVF{self.n_list},PQ{d // 2}x4fsr"

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        index = faiss.index_factory(d, factory_string, faiss_metric)
        index.train(X)
        index.add(X)
        self.base_index = index

        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
        self.refine_index = index_refine

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe, k_factor):
        faiss.cvar.indexIVF_stats.reset()
        self.n_probe = n_probe
        self.k_factor = k_factor
        self.base_index.nprobe = self.n_probe
        self.refine_index.k_factor = self.k_factor
        if self.k_factor == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis + faiss.cvar.indexIVF_stats.nq * self.n_list}  # noqa

    def __str__(self):
        return "FaissIVFPQfs(n_list=%d, dim_reduction=%d, n_probe=%d, k_factor=%d)" % (
            self.n_list,
            self._dim_reduction,
            self.n_probe,
            self.k_factor,
        )


class FaissBinaryIVF(Faiss):
    def __init__(self, metric, n_list):
        self.metric = metric
        self.n_list = n_list

    def fit(self, X):
        d = 8 * X.shape[1]
        quantizer = faiss.IndexBinaryFlat(d)
        self.index = faiss.IndexBinaryIVF(quantizer, d, self.n_list)
        self.index.train(X)
        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe
        self.index.nprobe = self.n_probe

    def query(self, v, n):
        q = np.expand_dims(v, axis=0)
        D, I = self.index.search(q, n)
        return I[0]

    def batch_query(self, X, n):
        self.res = self.index.search(X, n)

    def __str__(self):
        return "FaissBinaryIVF(n_list=%d, n_probe=%d)" % (self.n_list, self.n_probe)


class FaissIVFRaBitQ(Faiss):
    def __init__(self, metric, n_list):
        self.n_list = n_list
        self.metric = metric

    def fit(self, X):
        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        d = X.shape[1]
        factory_string = f"IVF{self.n_list},RaBitQ,Refine(Flat)"
        self.refine_index = faiss.index_factory(d, factory_string, faiss_metric)
        self.base_index = faiss.downcast_index(self.refine_index.base_index)

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.refine_index.train(X)
        self.refine_index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, n_probe, k_factor):
        self.n_probe = n_probe
        self.k_factor = k_factor
        self.base_index.nprobe = self.n_probe
        self.refine_index.k_factor = self.k_factor
        if self.k_factor == 0:
            self.index = self.base_index
        else:
            self.index = self.refine_index

    def __str__(self):
        return "FaissIVFRaBitQ(n_list=%d, n_probe=%d, k_factor=%d)" % (self.n_list, self.n_probe, self.k_factor)


class FaissHNSW(Faiss):
    def __init__(self, metric, M, efConstruction):
        self.metric = metric
        self.M = M
        self.efConstruction = efConstruction

    def fit(self, X):
        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        d = X.shape[1]
        self.index = faiss.IndexHNSWFlat(d, self.M, faiss_metric)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.verbose = False

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.hnsw.efSearch = ef

    def get_additional(self):
        return {"dist_comps": faiss.cvar.hnsw_stats.ndis}

    def __str__(self):
        return "FaissHNSW(%s, %d)" % (self.M, self.index.hnsw.efSearch)


class FaissHNSWQ(Faiss):
    def __init__(self, metric, M, efConstruction, direct_sq):
        self.metric = metric
        self.M = M
        self.efConstruction = efConstruction
        self.direct_sq = direct_sq

    def fit(self, X):
        d = X.shape[1]

        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        if self.direct_sq:
            self.index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit_direct, self.M, faiss_metric)
        else:
            self.index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, self.M, faiss_metric)

        self.index.hnsw.efConstruction = self.efConstruction
        self.index.verbose = False

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.index.train(X)
        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        self.index.hnsw.efSearch = ef

    def __str__(self):
        return "FaissHNSWQ(%s, %d)" % (self.M, self.index.hnsw.efSearch)


class FaissBinaryHNSW(Faiss):
    def __init__(self, metric, M, efConstruction):
        self.metric = metric
        self.M = M
        self.efConstruction = efConstruction

    def fit(self, X):
        d = 8 * X.shape[1]
        self.index = faiss.IndexBinaryHNSW(d, self.M)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.verbose = False

        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        self.index.hnsw.efSearch = ef

    def query(self, v, n):
        q = np.expand_dims(v, axis=0)
        D, I = self.index.search(q, n)
        return I[0]

    def batch_query(self, X, n):
        self.res = self.index.search(X, n)

    def __str__(self):
        return "FaissBinaryHNSW(%s, %d)" % (self.M, self.index.hnsw.efSearch)


class FaissNSG(Faiss):
    def __init__(self, metric, degree):
        self.metric = metric
        self.degree = degree

    def fit(self, X):
        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        d = X.shape[1]
        self.index = faiss.IndexNSGFlat(d, self.degree, faiss_metric)
        self.index.build_type = 1
        self.index.verbose = False

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, search_L):
        self.index.nsg.search_L = search_L

    def __str__(self):
        return "FaissNSG(%d, %d)" % (self.degree, self.index.nsg.search_L)


class FaissNSGQ(Faiss):
    def __init__(self, metric, degree, direct_sq):
        self.metric = metric
        self.degree = degree
        self.direct_sq = direct_sq

    def fit(self, X):
        d = X.shape[1]

        if self.metric in ["cosine", "ip", "normalized"]:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        else:
            faiss_metric = faiss.METRIC_L2

        if self.direct_sq:
            self.index = faiss.IndexNSGSQ(d, faiss.ScalarQuantizer.QT_8bit_direct, self.degree, faiss_metric)
        else:
            self.index = faiss.IndexNSGSQ(d, faiss.ScalarQuantizer.QT_8bit, self.degree, faiss_metric)

        self.index.build_type = 1
        self.index.verbose = False

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)

        self.index.train(X)
        self.index.add(X)

        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, search_L):
        self.index.nsg.search_L = search_L

    def __str__(self):
        return "FaissNSGQ(%d, %d)" % (self.degree, self.index.nsg.search_L)