import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

import ngtpy

from ..base.module import BaseANN


class QG(BaseANN):
    def __init__(
        self,
        metric,
        edge,
        epsilon,
        indegree,
        outdegree,
        indegree_ext,
        outdegree_ext,
        max_edge,
        sample,
        leaf,
        seed,
        hop,
        refine_k,
        reconst="base",
    ):
        metrics = {"euclidean": "2", "cosine": "E", "normalized": "2"}
        if metric not in metrics:
            raise NotImplementedError(f"NGT-QG does not support metric {metric}")

        self.metric = metrics[metric]
        self.edge_size = int(edge)
        self.epsilon = float(epsilon)
        self.indegree = int(indegree)
        self.outdegree = int(outdegree)
        self.indegree_ext = int(indegree_ext)
        self.outdegree_ext = int(outdegree_ext)
        self.max_edge_size = int(max_edge)
        self.sample = int(sample)
        self.leaf = leaf
        self.seed = seed
        self.hop = hop
        self.refine_k = int(refine_k)
        self.reconst = reconst
        self.dir = tempfile.mkdtemp(dir=os.getcwd())

    def fit(self, X):
        dim = X.shape[1]

        index_dir = self.dir
        anngIndex = os.path.join(index_dir, "ANNG-" + str(self.edge_size))
        tempIndex = os.path.join(index_dir, "TEMP-" + str(self.edge_size))
        forestIndex = os.path.join(index_dir, "FOREST-" + str(self.edge_size))
        index = forestIndex

        print("QG: index=" + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print("QG: create ANNG")
            t = time.time()
            args = [
                "ngt",
                "create",
                "-p1",
                "-b500",
                "-ga",
                "-oauto",
                "-D" + self.metric,
                "-d" + str(dim),
                "-E" + str(self.edge_size),
                "-e" + str(self.epsilon),
                "-rd",
                "-L" + self.leaf,
                "-s" + self.seed,
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=1, build=False, debug=False)
            idx.save()
            idx.close()
            print("QG: build ANNG")
            idx = ngtpy.Index(path=anngIndex)
            idx.build_index()
            idx.save()
            idx.close()
            print("QG: ANNG construction time(sec)=" + str(time.time() - t))
            if self.refine_k >= 0:
                print("QG: create RANNG")
                t = time.time()
                args = [
                    "ngt",
                    "refine-anng",
                    "-e" + str(0.1 if self.refine_k == 0 else self.epsilon),
                    "-k-" + str(self.refine_k),
                    anngIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(anngIndex)
                os.rename(tempIndex, anngIndex)
                print("QG: RANNG construction time(sec)=" + str(time.time() - t))

        if not os.path.exists(index):
            print("QG: construct Forest")
            t = time.time()
            args = [
                "ngt",
                "construct-forest",
                "-EH",
                "-H" + self.hop,
                "-ms",
                "-Mg",
                "-o" + str(self.outdegree),
                "-i" + str(self.indegree),
                "-O" + str(self.outdegree_ext),
                "-I" + str(self.indegree_ext),
                "-e0.0",
                forestIndex,
                anngIndex,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: construct Forest time(sec)=" + str(time.time() - t))
            if self.reconst == "none":
                print("QG: degree adjustment none")
            elif self.reconst == "base":
                print("QG: degree adjustment")
                t = time.time()
                args = [
                    "ngt",
                    "reconstruct-graph",
                    "-mS",
                    "-sp",
                    forestIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(forestIndex)
                os.rename(tempIndex, forestIndex)
            else:
                args = [
                    "ngt",
                    "reconstruct-graph",
                    "-mS",
                    "-sp",
                    "-Ps",
                    "-R" + self.reconst,
                    forestIndex,
                    tempIndex,
                ]
                print(" ".join(args))
                subprocess.call(args)
                shutil.rmtree(forestIndex)
                os.rename(tempIndex, forestIndex)
            print("QG: degree adjustment time(sec)=" + str(time.time() - t))

        if not os.path.exists(index + "/qg"):
            print("QG:create and append...")
            t = time.time()
            args = [
                "qbg",
                "create-qg",
                "-R-:u",
                "-k0",
                index,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: create qg time(sec)=" + str(time.time() - t))
            print("QB: build...")
            t = time.time()
            args = [
                "qbg",
                "build-qg",
                "-o" + str(self.sample),
                "-M1",
                "-ib",
                "-I400",
                "-Gz",
                "-Pn",
                "-E" + str(self.max_edge_size),
                index,
            ]
            print(" ".join(args))
            subprocess.call(args)
            print("QG: build qg time(sec)=" + str(time.time() - t))

        if os.path.exists(index + "/qg/grp"):
            self.index = ngtpy.QuantizedIndex(index, self.max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
        else:
            raise RuntimeError("QG: something went wrong.")

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QG: result_expansion=" + str(result_expansion))
        print("QG: epsilon=" + str(epsilon))
        self.name = "QG-NGT(%s,%s,%s:%s,%s:%s,%s,%s,%s,%s,%s,%s,%s,%s)" % (
            self.edge_size,
            self.epsilon,
            self.outdegree,
            self.indegree,
            self.outdegree_ext,
            self.indegree_ext,
            self.max_edge_size,
            self.sample,
            self.leaf,
            self.seed,
            self.hop,
            self.refine_k,
            epsilon,
            result_expansion,
        )
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def __del__(self):
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
