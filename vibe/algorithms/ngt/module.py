import os
import subprocess
import tempfile
import numpy as np

import ngtpy

from ..base.module import BaseANN


class QG(BaseANN):
    def __init__(self, metric, edge, outdegree, indegree, max_edge, epsilon, sample):
        metrics = {"euclidean": "2", "cosine": "E", "normalized": "2"}
        if metric not in metrics:
            raise NotImplementedError(f"NGT-QG does not support metric {metric}")

        self.metric = metrics[metric]
        self.edge_size = edge
        self.outdegree = outdegree
        self.indegree = indegree
        self.max_edge_size = max_edge
        self.build_time_limit = 3
        self.epsilon = epsilon
        self.sample = sample
        self.dir = tempfile.mkdtemp(dir=os.getcwd())

    def fit(self, X):
        dim = X.shape[1]

        index_dir = self.dir
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self.edge_size, self.outdegree, self.indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self.edge_size))

        if not os.path.exists(anngIndex):
            args = [
                "ngt",
                "create",
                "-it",
                "-p1",
                "-b500",
                "-ga",
                "-of",
                "-D" + self.metric,
                "-d" + str(dim),
                "-E" + str(self.edge_size),
                "-S40",
                "-e" + str(self.epsilon),
                "-P0",
                "-B30",
                "-T" + str(self.build_time_limit),
                anngIndex,
            ]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=1, debug=False)
            idx.save()
            idx.close()

        if not os.path.exists(index):
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-E " + str(self.outdegree),
                "-o " + str(self.outdegree),
                "-i " + str(self.indegree),
                anngIndex,
                index,
            ]
            subprocess.call(args)

        if not os.path.exists(index + "/qg"):
            args = ["qbg", "create-qg", index]
            subprocess.call(args)
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self.sample),
                "-M6",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self.max_edge_size),
                index,
            ]
            subprocess.call(args)

        if os.path.exists(index + "/qg/grp"):
            self.index = ngtpy.QuantizedIndex(index, self.max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
        else:
            raise RuntimeError("QG: something went wrong.")

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        self.name = "QG-NGT(%d, %d, %d, %d, %1.3f, %1.3f, %1.3f)" % (
            self.edge_size,
            self.outdegree,
            self.indegree,
            self.max_edge_size,
            self.epsilon,
            epsilon,
            result_expansion,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def __del__(self):
        import shutil

        if self.dir and os.path.exists(self.dir):
            try:
                shutil.rmtree(self.dir)
            except:
                pass


class ONNG(BaseANN):
    def __init__(self, metric, edge, outdegree, indegree, search_edge, epsilon, refine):
        metrics = {"euclidean": "2", "cosine": "E", "ip": "i", "normalized": "i", "hamming": "h"}
        self.metric = metrics[metric]
        self.edge_size = edge
        self.outdegree = outdegree
        self.indegree = indegree
        self.edge_size_for_search = search_edge
        self.refine_enabled = bool(refine)
        self.tree_disabled = False
        self.build_time_limit = 3
        self.epsilon = epsilon
        self.dir = tempfile.mkdtemp(dir=os.getcwd())

    def fit(self, X):
        dim = X.shape[1]

        index_dir = self.dir
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self.edge_size, self.outdegree, self.indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self.edge_size))

        if not os.path.exists(anngIndex):
            args = [
                "ngt",
                "create",
                "-it",
                "-p1",
                "-b500",
                "-ga",
                "-o" + ("c" if X.dtype == np.uint8 else "f"),
                "-D" + self.metric,
                "-d" + str(dim),
                "-E" + str(self.edge_size),
                "-S" + str(self.edge_size_for_search),
                "-e" + str(self.epsilon),
                "-P0",
                "-B30",
                "-T" + str(self.build_time_limit),
                anngIndex,
            ]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=1, debug=False)
            if self.refine_enabled:
                idx.refine_anng(
                    epsilon=self.epsilon,
                    num_of_edges=self.edge_size,
                    num_of_explored_edges=self.edge_size_for_search,
                )
            idx.save()
            idx.close()

        if not os.path.exists(index):
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-o " + str(self.outdegree),
                "-i " + str(self.indegree),
                anngIndex,
                index,
            ]
            subprocess.call(args)

        if os.path.exists(index):
            self.index = ngtpy.Index(index, read_only=True, tree_disabled=self.tree_disabled)
            self.indexName = index
        else:
            raise RuntimeError("QG: something went wrong.")

    def set_query_arguments(self, parameters):
        epsilon, edge_size = parameters
        self.name = "ONNG-NGT(%d, %d, %d, %1.3f, %d, %1.3f)" % (
            self.edge_size,
            self.outdegree,
            self.indegree,
            self.epsilon,
            edge_size,
            epsilon,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, edge_size=edge_size)

    def query(self, v, n):
        return self.index.search(v, n, with_distance=False)

    def __del__(self):
        import shutil

        if self.dir and os.path.exists(self.dir):
            try:
                shutil.rmtree(self.dir)
            except:
                pass
