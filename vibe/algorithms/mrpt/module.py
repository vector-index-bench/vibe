import mrpt
import numpy as np

from ..base.module import BaseANN


class MRPT(BaseANN):
    def __init__(self, metric, count):
        if metric not in ["euclidean", "cosine", "normalized"]:
            raise NotImplementedError(f"MRPT does not support metric {metric}")

        self._metric = metric
        self._k = count

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if self._metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self._index_autotuned = mrpt.MRPTIndex(X)
        self._index_autotuned.build_autotune_sample(target_recall=None, k=self._k, n_test=1000)

    def fit_ood(self, X_train, X_learn, X_learn_neighbors):
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32)
        if X_learn.dtype != np.float32:
            X_learn = X_learn.astype(np.float32)

        if self._metric == "cosine":
            X_train /= np.linalg.norm(X_train, axis=1)[:, np.newaxis]
            X_learn /= np.linalg.norm(X_learn, axis=1)[:, np.newaxis]

        Q = X_learn[np.random.choice(X_learn.shape[0], size=2000, replace=False)]
        self._index_autotuned = mrpt.MRPTIndex(X_train)
        self._index_autotuned.build_autotune(target_recall=None, Q=Q, k=self._k)

    def set_query_arguments(self, target_recall):
        self._target_recall = target_recall
        self._index = self._index_autotuned.subset(target_recall)
        self._par = self._index.parameters()

    def query(self, v, n):
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        if self._metric == "cosine":
            v = v / np.linalg.norm(v)
        return self._index.ann(v)

    def __str__(self):
        str_template = "MRPT(target recall=%.3f, trees=%d, depth=%d, vote threshold=%d, estimated recall=%.3f)"
        return str_template % (
            self._target_recall,
            self._par["n_trees"],
            self._par["depth"],
            self._par["votes"],
            self._par["estimated_recall"],
        )
