import hnswlib
import numpy as np

from ..base.module import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, M, efConstruction):
        self.metric = {"ip": "ip", "normalized": "ip", "cosine": "cosine", "euclidean": "l2"}[metric]
        self.M = M
        self.efConstruction = efConstruction

    def fit(self, X):
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(max_elements=len(X), ef_construction=self.efConstruction, M=self.M)
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.ef_query = ef
        self.p.set_ef(ef)

    def query(self, v, n):
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def __str__(self):
        return "HNSWLib(M=%d, efConstruction=%d, efQuery=%d)" % (
            self.M,
            self.efConstruction,
            self.ef_query,
        )
