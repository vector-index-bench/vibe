import glass
import numpy as np

from ..base.module import BaseANN

glass.set_num_threads(1)


class Glass(BaseANN):
    def __init__(self, metric, L, R, quant, search_quant, refine_quant):
        self.metric = metric
        self.L = L
        self.R = R
        self.quant = quant
        self.search_quant = search_quant
        self.refine_quant = refine_quant

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        metric_type = {"cosine": "IP", "euclidean": "L2", "ip": "IP", "normalized": "IP"}[self.metric]

        index = glass.Index(index_type="HNSW", metric=metric_type, R=self.R, L=self.L, quant=self.quant)
        graph = index.build(X)

        self.searcher = glass.Searcher(
            graph=graph, data=X, metric=metric_type, quantizer=self.search_quant, refine_quant=self.refine_quant
        )
        self.searcher.optimize(num_threads=1)

    def set_query_arguments(self, ef):
        self.ef = ef
        self.searcher.set_ef(ef)

    def query(self, q, n):
        if self.metric == "cosine":
            q = q / np.linalg.norm(q)
        return self.searcher.search(q, n)[0]

    def __str__(self):
        return "Glass(L=%d, R=%d, efQuery=%d, quant=%s, search_quant=%s, refine_quant=%s)" % (
            self.L,
            self.R,
            self.ef,
            self.quant,
            self.search_quant,
            self.refine_quant,
        )
