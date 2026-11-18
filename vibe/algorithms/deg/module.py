import deglib
import numpy as np

from ..base.module import BaseANN


_VIBE_METRIC_KIND = {
    "euclidean": "L2",
    "cosine": "InnerProduct",
    "ip": "InnerProduct",
    "normalized": "InnerProduct",
}


class _BaseDEG(BaseANN):
    """Shared setup for DEG and QG adapters."""

    def __init__(self, metric: str, k: int = 30, opt_target: str = "LowLID"):
        self.metric = metric.lower().strip()
        if self.metric not in _VIBE_METRIC_KIND:
            raise ValueError(f"Unsupported metric '{self.metric}'. Choose from: {list(_VIBE_METRIC_KIND.keys())}")

        self.k = int(k)
        self.opt_target = opt_target
        self.eps_or_ef = 0.1
        self._metric_kind = _VIBE_METRIC_KIND[self.metric]
        self.build_metric = deglib.Metric[f"FP32_{self._metric_kind}"]
        self.opt_enum = deglib.OptimizationTarget.resolve(self.opt_target)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return np.ascontiguousarray(X, dtype=np.float32)

    def _build_graph(self, X: np.ndarray) -> deglib.DynamicExplorationGraph:
        return deglib.build(
            X,
            presort=True,
            edges_per_vertex=self.k,
            metric=self.build_metric,
            optimization_target=self.opt_enum,
            optimize=False,
            threads=1,
        )


class DEG(_BaseDEG):
    """Dynamic Exploration Graph (DEG) using Float32 throughout."""

    def __init__(self, metric: str, k: int = 30, opt_target: str = "LowLID"):
        super().__init__(metric=metric, k=k, opt_target=opt_target)
        self.searcher: deglib.Searcher | None = None

    def fit(self, X: np.ndarray):
        """Builds the DEG graph in FP32 and prepares the Searcher."""
        X = self._prepare(X)
        graph = self._build_graph(X)
        self.searcher = graph.to_searcher(
            normalize_query=(self.metric == "cosine"),
            threads=1,
        )

    def set_query_arguments(self, eps_or_ef: float):
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        return self.searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        ef = f"ef={int(round(self.eps_or_ef))}" if self.eps_or_ef >= 1.0 else f"eps={self.eps_or_ef}"
        return f"DEG(k={self.k}, opt={self.opt_target}, {ef})"


class QG(_BaseDEG):
    """Quantized DEG (DEG-QG): quantized graph search (Uint4/Uint7/Int8) with FP16 reranking."""

    def __init__(self, metric: str, k: int = 30, opt_target: str = "LowLID"):
        super().__init__(metric=metric, k=k, opt_target=opt_target)
        quant_list = ["Uint4_L2", "Uint7_L2", "Int8_InnerProduct"]
        self.quant_metrics = {q: deglib.Metric.resolve(q) for q in quant_list}
        self.rerank_metric = deglib.Metric[f"FP16_{self._metric_kind}"]
        self.rerank_size_factor = 1.0
        self.searchers: dict[str, deglib.Searcher] = {}
        self.active_searcher: deglib.Searcher | None = None

    def fit(self, X: np.ndarray):
        """Builds the FP32 base graph once, creates a shared FP16 reranker, and builds one Searcher per quant."""
        X = self._prepare(X)
        base_graph = self._build_graph(X)
        reranker = base_graph.to_reranker(self.rerank_metric, threads=1)

        self.searchers.clear()
        for q_name, q_metric in self.quant_metrics.items():
            self.searchers[q_name] = base_graph.to_searcher(
                metric=q_metric,
                reranker=reranker,
                normalize_query=(self.metric == "cosine"),
                threads=1,
            )

    def set_query_arguments(self, quants: str, rerank_size_factor: float = 1.0, eps_or_ef: float = 0.1):
        if quants not in self.searchers:
            raise ValueError(f"Unknown quant '{quants}'. Available: {list(self.searchers.keys())}")
        self.active_searcher = self.searchers[quants]
        self.rerank_size_factor = float(rerank_size_factor)
        self.eps_or_ef = float(eps_or_ef)

    def query(self, v: np.ndarray, n: int) -> np.ndarray:
        return self.active_searcher.search(
            v,
            k=n,
            eps_or_ef=self.eps_or_ef,
            rerank_factor=self.rerank_size_factor,
            threads=1,
            return_distances=False,
            unsorted=True,
        )

    def __str__(self) -> str:
        q = self.active_searcher.graph.get_metric().name if self.active_searcher else "unfitted"
        ef = f"ef={int(round(self.eps_or_ef))}" if self.eps_or_ef >= 1.0 else f"eps={self.eps_or_ef}"
        return f"DEG-QG[{q}](k={self.k}, opt={self.opt_target}, rerank={self.rerank_size_factor}, {ef})"
