import flatnav
from flatnav.data_type import DataType
import numpy as np

from ..base.module import BaseANN


class FlatNav(BaseANN):
    def __init__(self, metric, max_edges_per_node, ef_construction):
        self.metric = metric
        self.max_edges_per_node = max_edges_per_node
        self.ef_construction = ef_construction

    def fit(self, X):
        data_types = {
            "float32": DataType.float32,
            "uint8": DataType.uint8,
            "int8": DataType.int8,
        }

        if self.metric == "ip" or self.metric == "normalized":
            distance_type = "angular"
        elif self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
            distance_type = "angular"
        else:
            distance_type = "l2"

        self.index = flatnav.index.create(
            distance_type=distance_type,
            index_data_type=data_types[str(X.dtype)],
            dim=X.shape[1],
            dataset_size=X.shape[0],
            max_edges_per_node=self.max_edges_per_node,
            verbose=True,
            collect_stats=False,
        )
        self.index.set_num_threads(1)

        self.index.add(data=X, ef_construction=self.ef_construction)

    def set_query_arguments(self, ef):
        self.ef_search = ef

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)

        distances, indices = self.index.search_single(
            query=v,
            ef_search=self.ef_search,
            K=n,
        )

        return indices

    def __str__(self):
        return "FlatNav(M=%d, efConstruction=%d, efSearch=%d)" % (
            self.max_edges_per_node,
            self.ef_construction,
            self.ef_search,
        )
