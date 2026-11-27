import os
from tempfile import NamedTemporaryFile

import glass
import numpy as np

from ..base.module import BaseANN

glass.set_num_threads(1)


class Glass(BaseANN):
    def __init__(self, metric, L, R, quant):
        self.metric = metric
        self.metric_type = {"cosine": "IP", "euclidean": "L2", "ip": "IP", "normalized": "IP"}[metric]
        self.L = L
        self.R = R
        self.quant = quant
        self.searcher = None

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        index = glass.Index(index_type="HNSW", metric=self.metric_type, R=self.R, L=self.L, quant=self.quant)
        graph = index.build(X)
        graph_file = NamedTemporaryFile(prefix="vibe-glass-", suffix=".graph")
        graph.save(graph_file.name)
        edge_bytes = len(X) * self.R * np.dtype(np.int32).itemsize
        if os.path.getsize(graph_file.name) <= edge_bytes:
            graph_file.close()
            raise OSError("PyGlass did not save the complete graph")

        self.data = X
        self.graph = graph
        self.graph_file = graph_file
        self.searcher = None

    def set_query_arguments(self, search_quant, refine_quant, ef):
        if self.searcher is None or (search_quant, refine_quant) != (self.search_quant, self.refine_quant):
            # Release the old searcher before loading another graph and allocating quantized data.
            self.searcher = None
            graph = self.graph
            self.graph = None  # PyGlass takes ownership of the graph when it creates a searcher.
            if graph is None:
                graph = glass.Graph(self.graph_file.name)

            searcher = glass.Searcher(
                graph=graph,
                data=self.data,
                metric=self.metric_type,
                quantizer=search_quant,
                refine_quant=refine_quant,
            )
            searcher.optimize(num_threads=1)
            self.searcher = searcher
            self.search_quant = search_quant
            self.refine_quant = refine_quant

        self.searcher.set_ef(ef)
        self.ef = ef

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
