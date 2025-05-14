import os
import struct
import tempfile
import numpy as np
import wrapper as pann

from ..base.module import BaseANN


def write_bin(filename, X):
    with open(filename, "wb") as f:
        nvecs, dim = X.shape
        f.write(struct.pack("<i", nvecs))
        f.write(struct.pack("<i", dim))
        X.flatten().tofile(f)


class ParlayANN(BaseANN):
    def __init__(self, metric, R, L, alpha):
        self.metric = metric
        self.R = R
        self.L = L
        self.alpha = alpha
        self.two_pass = True
        self.limit = 1000

    def fit(self, X):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

        self.index_dir = tempfile.mkdtemp(dir=os.getcwd())
        data_path = os.path.join(self.index_dir, "base.bin")
        save_path = os.path.join(self.index_dir, "index")
        write_bin(data_path, X)

        dtype = {"float32": "float", "uint8": "uint8"}[str(X.dtype)]

        if self.metric in ["cosine", "ip", "normalized"]:
            metric = "mips"
        else:
            metric = "Euclidian"  # not a typo

        if not os.path.exists(save_path):
            self.params = pann.build_vamana_index(
                metric, dtype, data_path, save_path, self.R, self.L, self.alpha, self.two_pass
            )

        self.index = pann.load_index(metric, dtype, data_path, save_path)

    def set_query_arguments(self, Q):
        self.Q = Q
        self.name = "ParlayANN(%d, %d, %g, %d)" % (self.R, self.L, self.alpha, self.Q)

    def query(self, v, k):
        if self.metric == "cosine":
            v = v / np.linalg.norm(v)
        return self.index.single_search(v, k, self.Q, True, self.limit)

    def batch_query(self, X, k):
        if self.metric == "cosine":
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        self.res, _ = self.index.batch_search(X, k, self.Q, True, self.limit)

    def get_batch_results(self):
        return self.res

    def __del__(self):
        import shutil

        if self.index_dir and os.path.exists(self.index_dir):
            try:
                shutil.rmtree(self.index_dir)
            except:
                pass
