import copy

import glass
import numpy as np
import chamfer
import fde

from ..base.module import BaseANN

glass.set_num_threads(1)


class MUVERA(BaseANN):
    def __init__(self, metric, k_sim, dim_proj, r_reps, M):
        self.metric = metric
        self.k_sim = k_sim
        self.dim_proj = dim_proj
        self.r_reps = r_reps
        self.M = M
        self.index = None
        self.chamfer = None
        self.document_config = None
        self.query_config = None

    def fit(self, X):
        embeddings, counts = X

        self.document_config = fde.FDEConfig()
        self.document_config.dimension = embeddings.shape[1]
        self.document_config.num_repetitions = self.r_reps
        self.document_config.num_simhash_projections = self.k_sim
        self.document_config.projection_dimension = self.dim_proj
        self.document_config.fill_empty_partitions = True

        self.query_config = copy.copy(self.document_config)
        self.query_config.fill_empty_partitions = False

        muvera_embeddings = []
        start = 0
        for i, count in enumerate(counts):
            if i % 1000 == 0:
                print("MUVERA %d/%d" % (i, len(counts)))
            muvera_embeddings.append(
                fde.generate_document_fixed_dimensional_encoding(
                    embeddings[start : start + count], self.document_config
                )
            )
            start += count

        train = np.array(muvera_embeddings)

        index = glass.Index(index_type="HNSW", metric="IP", R=self.M, L=100, quant="FP32")
        graph = index.build(train)

        self.searcher = glass.Searcher(graph=graph, data=train, metric="IP", quantizer="FP32")
        self.searcher.optimize(num_threads=1)

        self.chamfer = chamfer.Chamfer(embeddings, counts)

    def set_query_arguments(self, ef, rerank):
        self.ef = ef
        self.rerank = rerank
        self.searcher.set_ef(ef)

    def query(self, v, n):
        q = self.muvera.process_query(v)

        candidates = self.searcher.search(q, self.rerank)[0]
        distances = self.chamfer.distance_to_indices(v, candidates)

        sorted_indices = np.argsort(distances)[:n]
        return candidates[sorted_indices]

    def batch_query(self, X, n):
        Q = []
        for doc in X:
            Q.append(fde.generate_query_fixed_dimensional_encoding(doc, self.query_config))
        Q = np.array(Q)

        candidates = self.searcher.batch_search(Q, self.rerank)[0]
        self.res = self.chamfer.batch_query_subset_fixed(X, n, candidates)

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "MUVERA(k_sim=%d, dim_proj=%d, r_reps=%d, M=%d, ef=%d, rerank=%d)" % (
            self.k_sim,
            self.dim_proj,
            self.r_reps,
            self.M,
            self.ef,
            self.rerank,
        )

