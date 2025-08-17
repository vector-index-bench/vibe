import numpy as np
import svs
import os
import tempfile

from ..base.module import BaseANN


class SVSVamana(BaseANN):
    def __init__(self, metric, graph_max_degree, alpha, window_size):
        # using Cosine or MIP metric currently yields bad results
        # for now, support cosine distance by normalizing vectors
        if metric not in ["euclidean", "cosine", "normalized"]:
            raise NotImplementedError(f"SVSVamana does not support metric {metric}")
        self.metric = metric

        self.graph_max_degree = graph_max_degree
        self.alpha = alpha
        self.window_size = window_size

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        parameters = svs.VamanaBuildParameters(
            graph_max_degree=self.graph_max_degree,
            alpha=self.alpha,
            window_size=self.window_size,
        )

        self.index = svs.Vamana.build(
            parameters,
            X,
            distance_type=svs.DistanceType.L2,
            num_threads=1,
        )

    def set_query_arguments(self, search_window_size):
        self.search_window_size = search_window_size
        self.index.search_window_size = search_window_size

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        I, D = self.index.search(v, n)
        return I[0]

    def __str__(self):
        return "SVSVamana(graph_max_degree=%d, alpha=%g, window_size=%d, search_window_size=%d)" % (
            self.graph_max_degree,
            self.alpha,
            self.window_size,
            self.search_window_size,
        )


class SVSVamanaLVQ(BaseANN):
    def __init__(self, metric, graph_max_degree, alpha, window_size):
        if metric not in ["euclidean", "cosine", "normalized"]:
            raise NotImplementedError(f"SVSVamanaLVQ does not support metric {metric}")
        self.metric = metric

        self.graph_max_degree = graph_max_degree
        self.alpha = alpha
        self.window_size = window_size

    def fit(self, X):
        if X.dtype == np.float32:
            suffix = ".fvecs"
            dtype = svs.DataType.float32
        else:
            raise NotImplementedError(f"SVSVamanaLVQ: dtype {X.dtype} is invalid")

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        temp_file = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=True, suffix=suffix)
        temp_filename = os.path.basename(temp_file.name)
        svs.write_vecs(X, temp_filename)

        data_loader = svs.VectorDataLoader(temp_filename, dtype, dims=X.shape[1])

        B1 = 4
        B2 = 8
        padding = 32
        strategy = svs.LVQStrategy.Turbo
        compressed_loader = svs.LVQLoader(data_loader, primary=B1, residual=B2, strategy=strategy, padding=padding)

        parameters = svs.VamanaBuildParameters(
            graph_max_degree=self.graph_max_degree,
            alpha=self.alpha,
            window_size=self.window_size,
            max_candidate_pool_size=80,
        )

        self.index = svs.Vamana.build(
            parameters,
            compressed_loader,
            distance_type=svs.DistanceType.L2,
            num_threads=1,
        )

        temp_file.close()

    def set_query_arguments(self, search_window_size):
        self.search_window_size = search_window_size
        self.index.search_window_size = search_window_size

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        I, D = self.index.search(v, n)
        return I[0]

    def __str__(self):
        return "SVSVamanaLVQ(graph_max_degree=%d, alpha=%g, window_size=%d, search_window_size=%d)" % (
            self.graph_max_degree,
            self.alpha,
            self.window_size,
            self.search_window_size,
        )


class SVSVamanaLeanVec(BaseANN):
    def __init__(self, metric, graph_max_degree, alpha, window_size, reduced_dimensions):
        if metric not in ["euclidean", "cosine", "normalized"]:
            raise NotImplementedError(f"SVSVamanaLeanVec does not support metric {metric}")
        self.metric = metric

        self.graph_max_degree = graph_max_degree
        self.alpha = alpha
        self.window_size = window_size
        self.reduced_dimensions = reduced_dimensions

    def fit(self, X):
        if X.dtype == np.float32:
            suffix = ".fvecs"
            dtype = svs.DataType.float32
        else:
            raise ValueError(f"invalid dtype: {X.dtype}")

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        temp_file = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=True, suffix=suffix)
        temp_filename = os.path.basename(temp_file.name)
        svs.write_vecs(X, temp_filename)

        data_loader = svs.VectorDataLoader(temp_filename, dtype, dims=X.shape[1])

        compressed_loader = svs.LeanVecLoader(
            data_loader,
            self.reduced_dimensions,
            primary_kind=svs.LeanVecKind.lvq8,
            secondary_kind=svs.LeanVecKind.float16,
        )

        parameters = svs.VamanaBuildParameters(
            graph_max_degree=self.graph_max_degree,
            alpha=self.alpha,
            window_size=self.window_size,
            max_candidate_pool_size=80,
        )

        self.index = svs.Vamana.build(
            parameters,
            compressed_loader,
            distance_type=svs.DistanceType.L2,
            num_threads=1,
        )

        temp_file.close()

    def set_query_arguments(self, search_window_size):
        self.search_window_size = search_window_size
        self.index.search_window_size = search_window_size

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        I, D = self.index.search(v, n)
        return I[0]

    def __str__(self):
        return (
            "SVSVamanaLeanVec(graph_max_degree=%d, alpha=%g, window_size=%d, search_window_size=%d, reduced_dim=%d)"
            % (
                self.graph_max_degree,
                self.alpha,
                self.window_size,
                self.search_window_size,
                self.reduced_dimensions,
            )
        )
