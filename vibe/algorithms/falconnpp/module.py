import FalconnPP as fpp
import numpy as np

from ..base.module import BaseANN


class FalconnPP(BaseANN):
    def __init__(self, metric, numTables, numProj, bucketLimit, alpha, iProbes):
        if metric not in ["ip", "normalized", "cosine"]:
            raise NotImplementedError(f"FALCONN++ does not support metric {metric}")

        self.metric = metric
        self.numTables = numTables
        self.numProj = numProj
        self.bucketLimit = bucketLimit
        self.alpha = alpha
        self.iProbes = iProbes

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        numPoints, numDim = X.shape
        index = fpp.FalconnPP(numPoints, numDim)
        index.setIndexParam(self.numTables, self.numProj, self.bucketLimit, self.alpha, self.iProbes, 1)
        index.build(X.T)

        self.index = index

    def set_query_arguments(self, qProbes):
        self.qProbes = qProbes
        self.index.set_qProbes(qProbes)

    def query(self, v, n):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        q = np.expand_dims(v, axis=0).T.astype(np.float32)
        res = self.index.query(q, n)
        return res[0]

    def __str__(self):
        return "FalconnPP(numTables=%d, numProj=%d, bucketLimit=%d, alpha=%g, iProbes=%d, qProbes=%d)" % (
            self.numTables,
            self.numProj,
            self.bucketLimit,
            self.alpha,
            self.iProbes,
            self.qProbes,
        )
