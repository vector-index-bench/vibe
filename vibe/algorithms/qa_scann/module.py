import numpy as np
from qa_scann import QAScaNN as _QAScaNN

from ..base.module import BaseANN


class QAScaNN(BaseANN):
    """ScaNN with query-aware placement; out-of-distribution datasets only.

    There is no fit(): the placement needs the learning queries, which VIBE
    passes to fit_ood() on out-of-distribution datasets only.
    """

    def __init__(self, metric, centroids, num_leaves, p, dims_per_block, alpha, avq_threshold, max_mean_copies=None):
        if metric not in ("cosine", "ip", "normalized", "euclidean"):
            raise NotImplementedError(f"QA-ScaNN does not support metric {metric}")
        self.metric = metric
        # what ScaNN's k-means tree is trained on: "data" or "queries" (X_learn)
        self.centroids = centroids
        # None: floor(sqrt(n)) leaves, QA-ScaNN's default
        self.num_leaves = num_leaves
        # leaves per learning query the placement is optimised for
        self.p = p
        self.dims_per_block = dims_per_block
        self.alpha = alpha
        # ScaNN's anisotropic_quantization_threshold (NaN: off), as in VIBE's ScaNN avq_threshold
        self.avq_threshold = avq_threshold
        # cap on the mean copies per point (None: no cap; needs qa_scann >= 0.1.1)
        self.max_mean_copies = max_mean_copies
        self.leaves_to_search = None
        self.reorder = None
        # ScaNN's dot-product index on inner-product and unit-norm data: on unit
        # vectors it ranks like squared L2, and with a tree ScaNN gives it residual
        # AH codes, which are much more accurate. Squared L2 on Euclidean data.
        self.distance = "squared_l2" if metric == "euclidean" else "dot_product"

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if self.metric == "cosine":
            X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
            X_learn = X_learn / np.linalg.norm(X_learn, axis=1, keepdims=True)
        cap = {} if self.max_mean_copies is None else {"max_mean_copies": self.max_mean_copies}
        self.index = _QAScaNN(distance=self.distance, p=self.p, alpha=self.alpha,
                              centroids=self.centroids, num_leaves=self.num_leaves,
                              dims_per_block=self.dims_per_block,
                              anisotropic_quantization_threshold=self.avq_threshold, training_threads=1, **cap)
        self.index.fit(X_train, X_learn, X_learn_neighbors)

    def set_query_arguments(self, leaves_to_search, reorder):
        self.leaves_to_search = leaves_to_search
        self.reorder = reorder

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.index.search(v, n, self.leaves_to_search, self.reorder)[0]

    def batch_query(self, X, n):
        if self.metric == "cosine":
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        I, _ = self.index.search_batched(X, n, self.leaves_to_search, self.reorder, parallel=True)
        self.res = [row[row >= 0] for row in I]

    def get_additional(self):
        s = self.index.stats
        return {"copies_per_point": s["copies_per_point"], "max_copies": s["max_copies"],
                "num_leaves": s["num_leaves"], "capped": int(s.get("capped", False))}

    def __str__(self):
        # the leaf count the index was built with (resolved from None after fit_ood)
        num_leaves = getattr(getattr(self, "index", None), "num_leaves", None) or self.num_leaves
        cap = "" if self.max_mean_copies is None else ", max_mean_copies=%g" % self.max_mean_copies
        return ("QA-ScaNN(centroids=%s, num_leaves=%s, p=%d, dims_per_block=%d, alpha=%g, avq_threshold=%g%s, "
                "leaves_to_search=%d, reorder=%d)") % (
            self.centroids,
            num_leaves,
            self.p,
            self.dims_per_block,
            self.alpha,
            self.avq_threshold,
            cap,
            self.leaves_to_search,
            self.reorder,
        )
