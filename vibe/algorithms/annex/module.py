import numpy as np
import annexdb

from ..base.module import BaseANN


class Annex(BaseANN):
    def __init__(self, metric, M, efConstruction, sq8=False):
        self._metric = metric
        self._m = M
        self._ef_construct = efConstruction
        self._sq8 = sq8
        self._ef = 64
        self._index = None

    def fit(self, X):
        ann_metric = {
            "cosine": "cosine",
            "normalized": "cosine",
            "euclidean": "euclidean",
            "l2": "euclidean",
            "ip": "dot",
        }.get(self._metric, "cosine")

        self._index = annexdb.Index.build(
            np.asarray(X, dtype=np.float32),
            metric=ann_metric,
            m=self._m,
            ef_construct=self._ef_construct,
            quantize=self._sq8,
        )

    def set_query_arguments(self, ef):
        self._ef = ef

    def query(self, v, n):
        ids, _ = self._index.search(
            np.asarray(v, dtype=np.float32),
            k=n,
            ef=self._ef,
            sq8_screen=self._sq8,
        )
        return ids

    def batch_query(self, X, n):
        ids, _ = self._index.search_batch(
            np.asarray(X, dtype=np.float32),
            k=n,
            ef=self._ef,
            sq8_screen=self._sq8,
            threads=1,
        )
        self._res = ids

    def get_batch_results(self):
        return self._res

    def __str__(self):
        sq8 = "+sq8" if self._sq8 else ""
        return f"Annex(M={self._m}, efC={self._ef_construct}{sq8}, ef={self._ef})"
