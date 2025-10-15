import glass
from fastembed.postprocess import Muvera
import numpy as np

from ..base.module import BaseANN

glass.set_num_threads(1)


class MUVERA(BaseANN):
    def __init__(self, metric, k_sim, dim_proj, r_reps):
        self.metric = metric
        self.k_sim = k_sim
        self.dim_proj = dim_proj
        self.r_reps = r_reps
        self.muvera = None
        self.index = None

    def fit(self, X):
        embeddings, counts = X

        self.muvera = Muvera(dim=128, k_sim=self.k_sim, dim_proj=self.dim_proj, r_reps=self.r_reps)

        muvera_embeddings = []
        start = 0
        for i, count in enumerate(counts):
            if i % 1000 == 0:
                print("MUVERA %d/%d" % (i, len(counts)))
            muvera_embeddings.append(self.muvera.process_document(embeddings[start : start + count]))
            start += count

        train = np.array(muvera_embeddings)

        index = glass.Index(index_type="HNSW", metric="IP", R=8, L=100, quant="SQ8U")
        graph = index.build(train)

        self.searcher = glass.Searcher(graph=graph, data=train, metric="IP", quantizer="SQ4U", refine_quant="FP32")
        self.searcher.optimize(num_threads=1)

    def set_query_arguments(self, ef):
        self.ef = ef
        self.searcher.set_ef(ef)

    def query(self, v, n):
        q = self.muvera.process_query(v)
        return self.searcher.search(q, n)[0]

    def batch_query(self, X, n):
        Q = []
        for doc in X:
            Q.append(self.muvera.process_query(doc))
        Q = np.array(Q)
        self.res = self.searcher.batch_search(Q, n)[0]

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "MUVERA(k_sim=%d, dim_proj=%d, r_reps=%d, ef=%d)" % (
            self.k_sim,
            self.dim_proj,
            self.r_reps,
            self.ef,
        )
