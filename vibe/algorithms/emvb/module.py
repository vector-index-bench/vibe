import os
import sys
import tempfile
import numpy as np
import faiss
from faiss.contrib.inspect_tools import get_invlist

sys.path.insert(0, "/emvb/build")
from emvb import DocumentScorer, search as _search

from ..base.module import BaseANN


class EMVB(BaseANN):
    def __init__(self, metric, ncentroids, m_pq, nbits):
        self.metric = metric
        self.ncentroids = ncentroids
        self.m_pq = m_pq
        self.nbits = nbits
        self.scorer = None
        self.index_path = None

    def fit(self, X):
        embeddings, counts = X
        n_vecs, d = embeddings.shape
        n_docs = len(counts)

        quantizer = faiss.IndexFlatL2(d)
        index_pq = faiss.IndexIVFPQ(quantizer, d, self.ncentroids, self.m_pq, self.nbits)
        index_pq.train(embeddings)
        index_pq.add(embeddings)

        residuals = np.zeros([index_pq.ntotal, index_pq.pq.M], dtype=np.uint8)
        all_indices = np.zeros([index_pq.ntotal], dtype=np.uint64)
        centroids = index_pq.quantizer.reconstruct_n(0, index_pq.nlist)
        centroids_to_pids = [None] * centroids.shape[0]

        emb2pid = np.zeros(index_pq.ntotal, dtype=np.int64)

        offset = 0
        for i in range(n_docs):
            emb2pid[offset : offset + counts[i]] = i
            offset += counts[i]

        for i in range(index_pq.nlist):
            ids, codes = get_invlist(index_pq.invlists, i)
            residuals[ids] = codes
            all_indices[ids] = i
            centroids_to_pids[i] = emb2pid[ids]

        self.index_path = tempfile.mkdtemp(prefix="emvb_index_", dir=".")

        with open(os.path.join(self.index_path, "centroids_to_pids.txt"), "w") as file:
            for centroids_list in centroids_to_pids:
                for x in centroids_list:
                    file.write(f"{x} ")
                file.write("\n")

        np.save(os.path.join(self.index_path, "residuals.npy"), residuals)
        np.save(os.path.join(self.index_path, "centroids.npy"), centroids)
        np.save(os.path.join(self.index_path, "index_assignment.npy"), all_indices)
        pq_centroids = faiss.vector_to_array(index_pq.pq.centroids)
        np.save(os.path.join(self.index_path, "pq_centroids.npy"), pq_centroids)
        np.save(os.path.join(self.index_path, "doclens.npy"), counts)

        self.scorer = DocumentScorer(os.path.join(self.index_path, "doclens.npy"), self.index_path, 32)

    def set_query_arguments(self, nprobe, thresh, thresh_query, out_second_stage, n_doc_to_score):
        self.nprobe = nprobe
        self.thresh = thresh
        self.thresh_query = thresh_query
        self.out_second_stage = out_second_stage
        self.n_doc_to_score = n_doc_to_score

    def query(self, v, n):
        results = _search(
            self.scorer, v, n, self.nprobe, self.thresh, self.thresh_query, self.out_second_stage, self.n_doc_to_score
        )
        return [idx for idx, score in results]

    def batch_query(self, X, n):
        self.res = np.zeros((X.shape[0], n), dtype=np.uint64)
        for i, v in enumerate(X):
            self.res[i] = self.query(v, n)

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return (
            "EMVB(ncentroids=%d, m_pq=%d, nbits=%d, nprobe=%d, thresh=%.1f, thresh_query=%.1f, out_second_stage=%d, n_doc_to_score=%d)"
            % (
                self.ncentroids,
                self.m_pq,
                self.nbits,
                self.nprobe,
                self.thresh,
                self.thresh_query,
                self.out_second_stage,
                self.n_doc_to_score,
            )
        )

    def __del__(self):
        import shutil

        if self.index_path and os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
            except:
                pass
