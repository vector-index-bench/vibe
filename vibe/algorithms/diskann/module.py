import os
import tempfile
import numpy as np
from diskannpy import build_memory_index, StaticMemoryIndex

from ..base.module import BaseANN


class Vamana(BaseANN):

    def __init__(self, metric, graph_degree, complexity, alpha):
        self.metric = {"cosine": "cosine", "euclidean": "l2", "ip": "mips", "normalized": "mips"}[metric]
        self.graph_degree = graph_degree
        self.complexity = complexity
        self.alpha = alpha

    def fit(self, X):
        if X.dtype == np.uint8 and self.metric != "l2":
            raise NotImplementedError(f"DiskANN: metric {self.metric} not supported with uint8 data")

        self.temp_dir = tempfile.mkdtemp(dir=os.getcwd())

        build_memory_index(
            data=X,
            distance_metric=self.metric,
            index_directory=self.temp_dir,
            complexity=self.complexity,
            graph_degree=self.graph_degree,
            num_threads=1,
            alpha=self.alpha,
        )

        self.index = StaticMemoryIndex(
            index_directory=self.temp_dir,
            num_threads=1,
            initial_search_complexity=self.complexity,
        )

    def set_query_arguments(self, search_complexity):
        self.search_complexity = search_complexity

    def query(self, v, n):
        results = self.index.search(query=v, k_neighbors=n, complexity=self.search_complexity)
        return results.identifiers

    def __str__(self):
        str_template = "Vamana(%d, %d, %g, %d)"
        return str_template % (self.complexity, self.graph_degree, self.alpha, self.search_complexity)

    def __del__(self):
        import shutil

        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
