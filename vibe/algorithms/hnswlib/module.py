import hnswlib
import numpy as np

from ..base.module import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"ip": "ip", "normalized": "ip", "cosine": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param

    def fit(self, X):
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = "hnswlib(%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]
