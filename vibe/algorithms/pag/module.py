import os
import shutil
import tempfile

import numpy as np
import pag

from ..base.module import BaseANN


class PAG(BaseANN):
    METRICS = {
        "euclidean": pag.Metric.L2,
        "cosine": pag.Metric.Cosine,
        "ip": pag.Metric.MaximumInnerProduct,
        "normalized": pag.Metric.MaximumInnerProduct,
    }

    def __init__(self, metric, efConstruction, targetDegree, projectionLevels, maxSearchK=1000):
        if metric not in self.METRICS:
            raise NotImplementedError(f"PAG does not support metric {metric}")
        if projectionLevels <= 0 or projectionLevels % 8 != 0:
            raise ValueError("PAG projectionLevels must be a positive multiple of 8")

        self.metric = metric
        self.efConstruction = efConstruction
        self.targetDegree = targetDegree
        self.projectionLevels = projectionLevels
        self.maxSearchK = maxSearchK
        self.index_dir = None

    def fit(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index_dir = tempfile.mkdtemp(dir=os.getcwd())

        options = pag.BuildOptions()
        options.index_path = self.index_dir
        options.metric = self.METRICS[self.metric]
        options.max_search_k = self.maxSearchK
        options.ef_construction = self.efConstruction
        options.target_degree = self.targetDegree
        options.projection_levels = self.projectionLevels

        self.index = pag.Index()
        self.index.build(X, options)

    def set_query_arguments(self, ef):
        self.ef = ef

    def query(self, v, n):
        if n > self.maxSearchK:
            raise ValueError(f"PAG maxSearchK={self.maxSearchK} is smaller than requested k={n}")
        ids, _ = self.index.search(np.ascontiguousarray(v, dtype=np.float32), top_k=n, ef_search=self.ef)
        return ids[0]

    def batch_query(self, X, n):
        if n > self.maxSearchK:
            raise ValueError(f"PAG maxSearchK={self.maxSearchK} is smaller than requested k={n}")
        self.res, _ = self.index.search(np.ascontiguousarray(X, dtype=np.float32), top_k=n, ef_search=self.ef)

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "PAG(efConstruction=%d, targetDegree=%d, projectionLevels=%d, maxSearchK=%d, ef=%d)" % (
            self.efConstruction,
            self.targetDegree,
            self.projectionLevels,
            self.maxSearchK,
            self.ef,
        )

    def __del__(self):
        if self.index_dir and os.path.exists(self.index_dir):
            try:
                shutil.rmtree(self.index_dir)
            except OSError:
                pass
