import ctypes as ct

import faiss
import numpy as np

from ..hingezero_cpp20.module import HingeZeroCpp20


class HingeZeroLearnedSupport(HingeZeroCpp20):
    def __init__(self, metric, field_size=256):
        if metric != "ip":
            raise ValueError("Learned-support index supports inner product")
        super().__init__(metric, field_size)
        self.ranked_ids = None
        self.pool_size = 32768
        self.index = None
        self.neighbour_queries = 0
        self.returned = 0

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        bank = np.asarray(X_train, dtype=np.float32)
        labels = np.asarray(X_learn_neighbors, dtype=np.int32)
        if X_learn is None or labels.ndim != 2 or len(labels) != len(X_learn):
            raise ValueError("Learning queries and labels differ")
        if labels.size == 0 or np.min(labels) < 0 or np.max(labels) >= len(bank):
            raise ValueError("Invalid learning-neighbour labels")
        faiss.omp_set_num_threads(1)
        super().fit(bank)
        counts = np.bincount(labels[:, :100].ravel(), minlength=len(bank))
        active = np.flatnonzero(counts)
        order = np.lexsort((active, -counts[active]))
        self.ranked_ids = np.ascontiguousarray(active[order], dtype=np.int32)
        self._raw = np.memmap(self.root / "original-f32.bin", mode="r", dtype="<f4",
                              shape=(self.count, self.dimension))
        self.lib.hz20_rank_subset.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int,
            ct.c_int, ct.c_void_p
        ]
        self.lib.hz20_rank_subset.restype = ct.c_int
        self.set_query_arguments(min(32768, len(self.ranked_ids)))

    def set_query_arguments(self, pool_size):
        if self.ranked_ids is None:
            raise RuntimeError("Index must be trained before configuring queries")
        self.pool_size = int(pool_size)
        if self.pool_size < 100 or self.pool_size > len(self.ranked_ids):
            raise ValueError("Pool size exceeds indexed learning support")
        self.pool_ids = np.ascontiguousarray(self.ranked_ids[:self.pool_size], dtype=np.int32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.ascontiguousarray(self._raw[self.pool_ids], dtype=np.float32))

    def query(self, vector, n):
        if self.index is None or not self.handle:
            raise RuntimeError("Memory index not ready")
        q = np.ascontiguousarray(vector, dtype=np.float32)
        k = int(n)
        if q.shape != (self.dimension,) or not np.isfinite(q).all():
            raise ValueError("Invalid query")
        if not 1 <= k <= self.pool_size:
            raise ValueError("Invalid Top-k count")
        _, local = self.index.search(q[None, :], k)
        result_ids = np.ascontiguousarray(self.pool_ids[local[0]], dtype=np.int32)
        output = np.empty(k, dtype=np.int32)
        rc = self.lib.hz20_rank_subset(
            self.handle, ct.c_void_p(q.ctypes.data),
            ct.c_void_p(result_ids.ctypes.data), len(result_ids),
            k, ct.c_void_p(output.ctypes.data)
        )
        self._check(rc)
        if len(np.unique(output)) != k:
            raise RuntimeError("Duplicate output identifiers")
        self.neighbour_queries += 1
        self.returned += k
        return output

    def batch_query(self, X, n):
        self.res = [self.query(q, n) for q in X]

    def get_additional(self):
        return {
            "engine": "HingeZero single-core learned-support exact pool",
            "pool_size": self.pool_size,
            "learn_supported": int(len(self.ranked_ids)),
            "witness_verified": int(self.returned),
            "iterations": 0,
            "original_memory_preserved": True,
        }

    def __str__(self):
        return f"HingeZeroLearnedSupport(pool={self.pool_size})"
