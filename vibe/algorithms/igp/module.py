import os
import sys
import tempfile
import numpy as np

sys.path.append("/multi-vector-retrieval/build")
import IGP

from ..base.module import BaseANN


class IGPIndex(BaseANN):
    def __init__(self, metric, n_centroid, n_bit):
        self.metric = metric
        self.n_centroid = n_centroid
        self.n_bit = n_bit
        self.index = None
        self.index_path = None

    def fit(self, X):
        embeddings, counts = X
        n_vecs, vec_dim = embeddings.shape
        n_item = len(counts)

        self.index = IGP.DocRetrieval(
            item_n_vec_l=counts, n_item=n_item, vec_dim=vec_dim, n_centroid=self.n_centroid, n_bit=self.n_bit
        )

        self.index_path = tempfile.mkdtemp(prefix="igp_index_", dir=".")

        self._build_and_save_index(embeddings, counts)
        self._load_index()

    def _load_index(self):
        centroid_l = np.load(os.path.join(self.index_path, "centroid_l.npy"))
        vq_code_l = np.load(os.path.join(self.index_path, "vq_code_l.npy"))
        weight_l = np.load(os.path.join(self.index_path, "weight_l.npy"))
        residual_code_l = np.load(os.path.join(self.index_path, "residual_code_l.npy"))

        self.index.load_quantization_index(
            centroid_l=centroid_l, vq_code_l=vq_code_l, weight_l=weight_l, residual_code_l=residual_code_l
        )
        self.index.load_graph_index(os.path.join(self.index_path, "index.hnsw"))

    def _build_and_save_index(self, embeddings, counts):
        from .vq import vq_sq_ivf

        centroid_l, vq_code_l, weight_l, residual_code_l = vq_sq_ivf(
            train=embeddings, train_counts=counts, module=IGP, n_centroid=self.n_centroid, n_bit=self.n_bit
        )

        self.index.load_quantization_index(
            centroid_l=centroid_l, vq_code_l=vq_code_l, weight_l=weight_l, residual_code_l=residual_code_l
        )

        self.index.build_graph_index()

        self.index.save_graph_index(os.path.join(self.index_path, "index.hnsw"))
        np.save(os.path.join(self.index_path, "centroid_l.npy"), centroid_l)
        np.save(os.path.join(self.index_path, "vq_code_l.npy"), vq_code_l)
        np.save(os.path.join(self.index_path, "weight_l.npy"), weight_l)
        np.save(os.path.join(self.index_path, "residual_code_l.npy"), residual_code_l)

    def set_query_arguments(self, nprobe, probe_topk):
        self.nprobe = nprobe
        self.probe_topk = probe_topk

    def query(self, v, n):
        query_l = v.reshape(1, v.shape[0], v.shape[1])
        result = self.index.search(query_l=query_l, topk=n, nprobe=self.nprobe, probe_topk=self.probe_topk)
        return result[1][0]

    def batch_query(self, X, n):
        result = self.index.search(query_l=X, topk=n, nprobe=self.nprobe, probe_topk=self.probe_topk)
        self.res = result[1]

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return (
            f"IGP(n_centroid={self.n_centroid}, n_bit={self.n_bit}, nprobe={self.nprobe}, probe_topk={self.probe_topk})"
        )
