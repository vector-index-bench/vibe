import os
import subprocess
import tempfile
import time
import numpy as np

import ngtpy

from ..base.module import BaseANN


class QG(BaseANN):
    def __init__(self, metric, edge, outdegree, indegree, max_edge, epsilon, sample):
        metrics = {"euclidean": "2", "cosine": "E", "normalized": "2"}
        if metric not in metrics:
            raise NotImplementedError(f"NGT-QG does not support metric {metric}")

        self._metric = metrics[metric]
        self._edge_size = edge
        self._outdegree = outdegree
        self._indegree = indegree
        self._max_edge_size = max_edge
        self._build_time_limit = 3
        self._epsilon = epsilon
        self._sample = sample
        self._dir = tempfile.mkdtemp(dir=os.getcwd())

    def fit(self, X):
        dim = X.shape[1]

        index_dir = self._dir
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self._edge_size, self._outdegree, self._indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self._edge_size))

        if not os.path.exists(anngIndex):
            print("QG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-it",
                "-p1",
                "-b500",
                "-ga",
                "-of",
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + str(self._edge_size),
                "-S40",
                "-e" + str(self._epsilon),
                "-P0",
                "-B30",
                "-T" + str(self._build_time_limit),
                anngIndex,
            ]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=1, debug=False)
            idx.save()
            idx.close()
            print("QG: ANNG construction time(sec)=" + str(time.time() - t))

        if not os.path.exists(index):
            print("QG: degree adjustment")
            t = time.time()
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-E " + str(self._outdegree),
                "-o " + str(self._outdegree),
                "-i " + str(self._indegree),
                anngIndex,
                index,
            ]
            subprocess.call(args)
            print("QG: degree adjustment time(sec)=" + str(time.time() - t))

        if not os.path.exists(index + "/qg"):
            print("QG: create and append...")
            t = time.time()
            args = ["qbg", "create-qg", index]
            subprocess.call(args)
            print("QG: create qg time(sec)=" + str(time.time() - t))
            print("QB: build...")
            t = time.time()
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self._sample),
                "-M6",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self._max_edge_size),
                index,
            ]
            subprocess.call(args)
            print("QG: build qg time(sec)=" + str(time.time() - t))

        if os.path.exists(index + "/qg/grp"):
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print("QG: open time(sec)=" + str(time.time() - t))
        else:
            print("QG: something wrong.")

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        self.name = "QG-NGT(%s, %s, %s, %s, %s, %1.3f)" % (
            self._edge_size,
            self._outdegree,
            self._indegree,
            self._max_edge_size,
            epsilon,
            result_expansion,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def __del__(self):
        import shutil

        if self._dir and os.path.exists(self._dir):
            try:
                shutil.rmtree(self._dir)
            except:
                pass


class ONNG(BaseANN):
    def __init__(self, metric, edge, outdegree, indegree, search_edge, epsilon, refine):
        metrics = {"euclidean": "2", "cosine": "E", "ip": "i", "normalized": "i", "hamming": "h"}
        self._metric = metrics[metric]
        self._edge_size = edge
        self._outdegree = outdegree
        self._indegree = indegree
        self._edge_size_for_search = search_edge
        self._refine_enabled = bool(refine)
        self._tree_disabled = False
        self._build_time_limit = 3
        self._epsilon = epsilon
        self._dir = tempfile.mkdtemp(dir=os.getcwd())

    def fit(self, X):
        dim = X.shape[1]

        index_dir = self._dir
        index = os.path.join(index_dir, "ONNG-{}-{}-{}".format(self._edge_size, self._outdegree, self._indegree))
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self._edge_size))

        if not os.path.exists(anngIndex):
            print("ONNG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-it",
                "-p1",
                "-b500",
                "-ga",
                "-o" + ("c" if X.dtype == np.uint8 else "f"),
                "-D" + self._metric,
                "-d" + str(dim),
                "-E" + str(self._edge_size),
                "-S" + str(self._edge_size_for_search),
                "-e" + str(self._epsilon),
                "-P0",
                "-B30",
                "-T" + str(self._build_time_limit),
                anngIndex,
            ]
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=1, debug=False)
            print("ONNG: ANNG construction time(sec)=" + str(time.time() - t))
            t = time.time()
            if self._refine_enabled:
                idx.refine_anng(
                    epsilon=self._epsilon,
                    num_of_edges=self._edge_size,
                    num_of_explored_edges=self._edge_size_for_search,
                )
            print("ONNG: RNNG construction time(sec)=" + str(time.time() - t))
            idx.save()
            idx.close()

        if not os.path.exists(index):
            print("ONNG: degree adjustment")
            t = time.time()
            args = [
                "ngt",
                "reconstruct-graph",
                "-mS",
                "-o " + str(self._outdegree),
                "-i " + str(self._indegree),
                anngIndex,
                index,
            ]
            subprocess.call(args)
            print("QG: degree adjustment time(sec)=" + str(time.time() - t))

        if os.path.exists(index):
            t = time.time()
            self.index = ngtpy.Index(index, read_only=True, tree_disabled=self._tree_disabled)
            self.indexName = index
            print("ONNG: open time(sec)=" + str(time.time() - t))
        else:
            print("ONNG: something wrong.")

    def set_query_arguments(self, parameters):
        epsilon, edge_size = parameters
        self.name = "ONNG-NGT(%s, %s, %s, %s, %1.3f)" % (
            self._edge_size,
            self._outdegree,
            self._indegree,
            edge_size,
            epsilon,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, edge_size=edge_size)

    def query(self, v, n):
        return self.index.search(v, n, with_distance=False)

    def __del__(self):
        import shutil

        if self._dir and os.path.exists(self._dir):
            try:
                shutil.rmtree(self._dir)
            except:
                pass
