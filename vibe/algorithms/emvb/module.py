import os
import sys
import numpy as np
import faiss
from tqdm import tqdm
from faiss.contrib.inspect_tools import get_invlist

sys.path.insert(0, '/emvb/build')
from emvb import DocumentScorer, search as _search

from ..base.module import BaseANN


class EMVB(BaseANN):

    def __init__(self, metric, ncentroids, m_pq, nbits, nprobe, k_factor, q_factor):
        self.metric = metric
        self.ncentroids = ncentroids
        self.m_pq = m_pq
        self.nbits = nbits
        self.nprobe = nprobe
        self.k_factor = k_factor
        self.q_factor = q_factor
        self.scorer = None
        self.index_path = None

    def fit(self, X):
        embeddings, counts = X
        n_vecs, d = embeddings.shape

        quantizer = faiss.IndexFlatL2(d)
        index_jmpq = faiss.IndexIVFPQ(quantizer, d, self.ncentroids, self.m_pq, self.nbits)
        index_jmpq.train(embeddings)
        index_jmpq.add(embeddings)

        residuals = np.zeros([index_jmpq.ntotal, index_jmpq.pq.M], dtype=np.uint8)
        all_indices = np.zeros([index_jmpq.ntotal], dtype=np.uint64)
        centroids = index_jmpq.quantizer.reconstruct_n(0, index_jmpq.nlist)
        centroids_to_pids = [None] * centroids.shape[0]

        n_docs = len(counts)
        emb2pid = np.zeros(index_jmpq.ntotal, dtype=np.int64)
        offset = 0
        for i in range(n_docs):
            l = counts[i]
            emb2pid[offset: offset+l] = i
            offset = offset + l

        for i in tqdm(range(index_jmpq.nlist), desc="Extracting invlists", disable=True):
            ids, codes = get_invlist(index_jmpq.invlists, i)
            residuals[ids] = codes
            all_indices[ids] = i
            centroids_to_pids[i] = emb2pid[ids]

        import tempfile
        self.index_path = tempfile.mkdtemp(prefix="emvb_index_")

        with open(os.path.join(self.index_path, "centroids_to_pids.txt"), "w") as file:
            for centroids_list in centroids_to_pids:
                for x in centroids_list:
                    file.write(f"{x} ")
                file.write("\n")

        np.save(os.path.join(self.index_path, "residuals.npy"), residuals)
        np.save(os.path.join(self.index_path, "centroids.npy"), centroids)
        np.save(os.path.join(self.index_path, "index_assignment.npy"), all_indices)
        pq_centroids = faiss.vector_to_array(index_jmpq.pq.centroids)
        np.save(os.path.join(self.index_path, "pq_centroids.npy"), pq_centroids)
        np.save(os.path.join(self.index_path, "doclens.npy"), counts)

        self.scorer = DocumentScorer(os.path.join(self.index_path, "doclens.npy"), self.index_path, 32)

    def set_query_arguments(self, nprobe, k_factor, q_factor, rerank_size):
        self.nprobe = nprobe
        self.k_factor = k_factor
        self.q_factor = q_factor
        self.rerank_size = rerank_size

    def query(self, v, n):
        results = _search(
            self.scorer,
            v,
            n,
            self.nprobe,
            self.k_factor,
            self.q_factor,
            self.rerank_size,
            1000
        )
        return [idx for idx, score in results]

    def __str__(self):
        return (f"EMVB(ncentroids={self.ncentroids}, m_pq={self.m_pq}, nbits={self.nbits}, "
                f"nprobe={self.nprobe}, k_factor={self.k_factor}, q_factor={self.q_factor})")
