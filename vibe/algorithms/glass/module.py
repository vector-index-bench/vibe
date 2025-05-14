import glassppy as glass
import numpy as np
import os
import tempfile

from ..base.module import BaseANN


class Glass(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.metric = metric
        self.R = method_param["R"]
        self.L = method_param["L"]
        self.level = method_param["level"]
        self.name = "glass_(%s)" % (method_param)
        self.dir = tempfile.mkdtemp(dir=os.getcwd())
        self.path = f"dim_{dim}_R_{self.R}_L_{self.L}.glass"

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        metric_type = {"cosine": "IP", "euclidean": "L2", "ip": "IP", "normalized": "IP"}[self.metric]
        if self.path not in os.listdir(self.dir):
            p = glass.Index("HNSW", dim=X.shape[1], metric=metric_type, R=self.R, L=self.L)
            g = p.build(X)
            g.save(os.path.join(self.dir, self.path))

        g = glass.Graph(os.path.join(self.dir, self.path))
        self.searcher = glass.Searcher(g, X, metric_type, self.level)
        self.searcher.optimize(1)

    def set_query_arguments(self, ef):
        self.searcher.set_ef(ef)

    def query(self, q, n):
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)
        return self.searcher.search(q, n)

    def __del__(self):
        import shutil

        if self.dir and os.path.exists(self.dir):
            try:
                shutil.rmtree(self.dir)
            except:
                pass
