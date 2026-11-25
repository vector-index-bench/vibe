import deglib
import numpy as np

from ..base.module import BaseANN


class DEG(BaseANN):
    """Dynamic Exploration Graph (DEG) using the high-level deglib.build + deglib.searcher API."""

    def __init__(self, metric: str, k: int = 30):
        self.metric = metric.lower().strip()
        self.k = int(k)
        self.eps_or_ef = 0.1
        self.searcher: deglib.Searcher | None = None

    def fit(self, X: np.ndarray):
        """Builds the DEG graph and prepares an optimized Searcher."""
        graph = deglib.build(
            X,
            edges_per_vertex=self.k,
            metric=self.metric,
            presort=True,
            optimize=False,
            threads=1,
        )
        self.searcher = deglib.searcher(graph, optimize=True, threads=1)

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
        ).indices

    def __str__(self) -> str:
        param = f"ef={int(round(self.eps_or_ef))}" if self.eps_or_ef >= 1.0 else f"eps={self.eps_or_ef}"
        return f"DEG(k={self.k}, {param})"


class QG(BaseANN):
    """Quantized DEG (DEG-QG): 4bit / 8bit quantized graph search with FP16 reranking."""

    def __init__(self, metric: str, k: int = 30):
        self.metric = metric.lower().strip()
        self.k = int(k)
        self.eps_or_ef = 0.1
        self.quants = ("4bit", "8bit")
        self.rerank_size_factor = 1.0
        self.searchers: dict[str, deglib.Searcher] = {}
        self.active_searcher: deglib.Searcher | None = None

    def fit(self, X: np.ndarray):
        """Builds the base graph once, creates a shared FP16 reranker, and builds one Searcher per quant."""
        graph = deglib.build(
            X,
            edges_per_vertex=self.k,
            metric=self.metric,
            presort=True,
            optimize=False,
            threads=1,
        )
        reranker = deglib.reranker(graph, compress="16bit", threads=1)

        self.searchers.clear()
        for q_name in self.quants:
            self.searchers[q_name] = deglib.searcher(
                graph,
                compress=q_name,
                reranker=reranker,
                optimize=True,
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
        ).indices

    def __str__(self) -> str:
        q = self.active_searcher.graph.get_metric().name if self.active_searcher else "unfitted"
        param = f"ef={int(round(self.eps_or_ef))}" if self.eps_or_ef >= 1.0 else f"eps={self.eps_or_ef}"
        return f"DEG-QG[{q}](k={self.k}, rerank={self.rerank_size_factor}, {param})"
