import time
import numpy as np
import deglib

from ..base.module import BaseANN


_METRIC_MAP = {
    "euclidean": (deglib.Metric.FP32_L2, deglib.Metric.Int8_L2, deglib.Metric.FP16_L2),
    "cosine": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "ip": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
    "normalized": (deglib.Metric.FP32_InnerProduct, deglib.Metric.Int8_InnerProduct, deglib.Metric.FP16_InnerProduct),
}


class DEG(BaseANN):
    """
    Dynamic Exploration Graph (DEG) using Float32 throughout.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        prune_non_rng: bool = False,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = 1
        self.eps_or_ef = 0.1

        self.metric_enum = _METRIC_MAP[self.metric][0]
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]
        self.graph = None
        self.searcher = None

    def fit(self, X: np.ndarray):
        """Builds the DEG graph in FP32."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)

        # 1. FLAS 1D Pre-sorting
        print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
        sorted_indices = deglib.optimization.presort(
            X,
            metric=self.metric_enum,
            threads=self.threads,
        )

        # 2. Build graph in FP32
        print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
        graph = deglib.builder.build_from_data(
            data=X[sorted_indices],
            labels=sorted_indices,
            edges_per_vertex=self.k,
            metric=self.metric_enum,
            seed=7,
            optimization_target=self.opt_enum,
            thread_count=self.threads,
        )

        # 3. Optional MRNG edge pruning
        if self.prune_non_rng:
            deglib.optimization.prune_non_rng_edges(graph, num_threads=self.threads)

        self.graph = graph.to_readonly()
        self.searcher = deglib.search.create_searcher(graph=self.graph)

        # 4. Optimize entry vertices via k-means medoids
        t_opt = time.time()
        self.searcher.optimize()
        print(f"Optimized Searcher for graph in {time.time() - t_opt:.2f}s", flush=True)

    def set_query_arguments(self, eps_or_ef: float | int):
        """Sets query-time parameter: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with Float32 via C++ searcher."""
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.searcher.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            eps_or_ef=self.eps_or_ef,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        if self.eps_or_ef >= 1.0:
            return f"DEG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, ef={int(round(self.eps_or_ef))})"
        return f"DEG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, eps={self.eps_or_ef})"


class QG(BaseANN):
    """
    Quantized DEG (DEG-QG): Graph search with INT8 quantized vectors and FP16 reranking.
    """

    def __init__(
        self,
        metric: str,
        k: int = 30,
        opt_target: str = "LowLID",
        prune_non_rng: bool = False,
    ):
        self.metric = metric.lower().strip()
        if self.metric not in _METRIC_MAP:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_METRIC_MAP.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.prune_non_rng = bool(prune_non_rng)
        self.threads = 1
        self.rerank_size_factor = 1.0
        self.eps_or_ef = 0.1

        self.base_metric, self.int8_metric, self.fp16_metric = _METRIC_MAP[self.metric]
        self.opt_enum = deglib.builder.OptimizationTarget[self.opt_target]

        self.graph = None
        self.searcher = None
        self.quantizer = None
        self.original_features_fp16 = None
        self.rerank_space_fp16 = None

    def fit(self, X: np.ndarray):
        """Builds DEG graph, quantizes vectors to INT8 using ScalarQuantizer, and prepares C++ searcher."""
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        X = np.ascontiguousarray(X, dtype=np.float32)
        dims = X.shape[1]

        self.original_features_fp16 = deglib.distances.floats_to_fp16(X)
        self.rerank_space_fp16 = deglib.FloatSpace.create(dim=dims, metric=self.fp16_metric)

        # 1. FLAS 1D Pre-sorting
        print(f"Running FLAS 1D Pre-sorting (threads={self.threads})...", flush=True)
        sorted_indices = deglib.optimization.presort(
            X,
            metric=self.base_metric,
            threads=self.threads,
        )

        # 2. Build graph in FP32
        print(f"Building DEG graph (K={self.k}, Opt={self.opt_target}, threads={self.threads})...", flush=True)
        graph = deglib.builder.build_from_data(
            data=X[sorted_indices],
            labels=sorted_indices,
            edges_per_vertex=self.k,
            metric=self.base_metric,
            seed=7,
            optimization_target=self.opt_enum,
            thread_count=self.threads,
        )

        # 3. Optional MRNG edge pruning
        if self.prune_non_rng:
            deglib.optimization.prune_non_rng_edges(graph, num_threads=self.threads)

        # 4. Finalize ReadOnlyGraph with INT8 features using calibrated ScalarQuantizer
        self.quantizer = deglib.optimization.make_scalar_quantizer_int8(X)
        int8_features = self.quantizer.quantize(X, num_threads=self.threads)
        target_space = deglib.FloatSpace.create(dim=dims, metric=self.int8_metric)
        self.graph = graph.to_readonly(target_space, int8_features)

        # 5. Initialize C++ Zero-overhead Searcher
        self.searcher = deglib.search.create_searcher(
            graph=self.graph,
            quantizer=self.quantizer,
            refine_space=self.rerank_space_fp16,
            refine_data=self.original_features_fp16,
        )

        # 6. Optimize entry vertices via k-means medoids
        t_opt = time.time()
        self.searcher.optimize()
        print(f"Optimized Searcher for the provided graph and hardware in {time.time() - t_opt:.2f}s", flush=True)

    def set_query_arguments(self, rerank_size_factor: float = 1.0, eps_or_ef: float | int = 0.1):
        """Sets query-time parameters: values >= 1.0 are treated as ef, < 1.0 as eps."""
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        """Single query search on 1 thread with INT8 search and FP16 reranking directly in C++."""
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.searcher.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        if self.eps_or_ef >= 1.0:
            return (
                f"DEG-QG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, "
                f"rerank_factor={self.rerank_size_factor}, ef={int(round(self.eps_or_ef))})"
            )
        return (
            f"DEG-QG(k={self.k}, opt={self.opt_target}, prune_rng={self.prune_non_rng}, "
            f"rerank_factor={self.rerank_size_factor}, eps={self.eps_or_ef})"
        )
