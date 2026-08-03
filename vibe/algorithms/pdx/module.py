import numpy as np
from pdxearch import IndexPDXIVF, IndexPDXIVFSQ8, IndexPDXIVFTree, IndexPDXIVFTreeSQ8

from ..base.module import BaseANN


class PDX(BaseANN):
    INDEX_TYPES = {
        "ivf": IndexPDXIVF,
        "ivf_sq8": IndexPDXIVFSQ8,
        "tree": IndexPDXIVFTree,
        "tree_sq8": IndexPDXIVFTreeSQ8,
    }

    METRICS = {
        "euclidean": "l2sq",
        "cosine": "cosine",
        "ip": "ip",
        "normalized": "cosine",
    }

    def __init__(
        self,
        metric,
        dimension,
        index_type,
        num_clusters,
        num_meso_clusters,
        sampling_fraction,
        kmeans_iters,
    ):
        if metric not in self.METRICS:
            raise ValueError(f"Unsupported PDX metric: {metric}")
        if index_type not in self.INDEX_TYPES:
            raise ValueError(f"Unsupported PDX index type: {index_type}")

        self.metric = metric
        self.dimension = dimension
        self.index_type = index_type
        self.num_clusters = num_clusters
        self.num_meso_clusters = num_meso_clusters
        self.sampling_fraction = sampling_fraction
        self.kmeans_iters = kmeans_iters
        self.n_probe = 32

    def fit(self, X):
        if X.shape[1] != self.dimension:
            raise ValueError(f"PDX dimension mismatch: expected {self.dimension}, got {X.shape[1]}")

        index_class = self.INDEX_TYPES[self.index_type]
        index_kwargs = {
            "num_dimensions": self.dimension,
            "distance_metric": self.METRICS[self.metric],
            "normalize": self.metric in ("cosine", "normalized"),
            "num_clusters": self.num_clusters,
            "sampling_fraction": self.sampling_fraction,
            "kmeans_iters": self.kmeans_iters,
            "n_threads": 1,
        }
        if self.index_type.startswith("tree"):
            index_kwargs["num_meso_clusters"] = self.num_meso_clusters

        self.index = index_class(**index_kwargs)
        self.index.build(np.ascontiguousarray(X, dtype=np.float32))

    def set_query_arguments(self, n_probe):
        self.n_probe = n_probe

    def query(self, v, n):
        ids, _ = self.index.search(np.ascontiguousarray(v, dtype=np.float32), n, nprobe=self.n_probe)
        return ids

    def get_additional(self):
        return {
            "pdx_index_size": self.index.in_memory_size_bytes,
            "pdx_num_clusters": self.index.num_clusters,
        }

    def __str__(self):
        return (
            "PDX(index_type=%s, num_clusters=%d, num_meso_clusters=%d, "
            "sampling_fraction=%.2f, kmeans_iters=%d, n_probe=%d)"
            % (
                self.index_type,
                self.num_clusters,
                self.num_meso_clusters,
                self.sampling_fraction,
                self.kmeans_iters,
                self.n_probe,
            )
        )
