import os
import sys

sys.path.append(os.path.abspath("/mopbucket/build"))

import dessert_py
import lorann
import glass
import mopbucket_py as mop
import numpy as np

from ..base.module import BaseANN


class DESSERT(BaseANN):
    def __init__(self, metric, num_tables, hashes_per_table, n_clusters):
        self.metric = metric
        self.num_tables = num_tables
        self.hashes_per_table = hashes_per_table
        self.n_clusters = n_clusters
        self.index = None
        self.centroid_search = None
        self.centroids = None
        self.num_to_rerank = None

    def fit(self, X):
        embeddings, counts = X
        embedding_dim = embeddings.shape[1]

        if self.n_clusters > 8000:
            km = mop.kmeans_graph(embeddings, self.n_clusters, 1)
            km.init_random()
            for iter in range(8):
                num_changed, score = km.reassign()
                km.recompute_centroids()
            self.centroids = km.get_centroids()
            centroid_ids = km.get_clustering()
        else:
            kmeans = lorann.KMeans(n_clusters=self.n_clusters, iters=4, distance=lorann.IP)
            cluster_map = kmeans.train(embeddings, verbose=False)
            self.centroids = kmeans.get_centroids()

            centroid_ids = np.zeros(len(embeddings), dtype=np.uint32)
            for i, cluster in enumerate(cluster_map):
                for p in cluster:
                    centroid_ids[p] = i

        if self.n_clusters > 8000:
            index = glass.Index(index_type="HNSW", metric="L2", R=32, L=400, quant="FP32")
            graph = index.build(self.centroids)

            searcher = glass.Searcher(graph=graph, data=self.centroids, metric="L2", quantizer="FP32")
            searcher.optimize(num_threads=1)
            searcher.set_ef(100)
            self.centroid_search = lambda X, n: searcher.batch_search(X, n)[0].flatten()
        else:
            centroid_index = lorann.LorannIndex(self.centroids, 1, None, distance=lorann.IP)
            self.centroid_search = lambda X, n: centroid_index.exact_search(X, n).flatten()

        self.index = dessert_py.DocRetrieval(
            dense_input_dimension=embedding_dim,
            num_tables=self.num_tables,
            hashes_per_table=self.hashes_per_table,
            centroids=self.centroids,
        )

        start = 0
        for doc_id, count in enumerate(counts):
            doc_embeddings = embeddings[start : start + count]
            doc_centroids = centroid_ids[start : start + count]
            self.index.add_doc(
                doc_id=str(doc_id),
                doc_embeddings=doc_embeddings,
                doc_centroid_ids=doc_centroids,
            )
            start += count

    def set_query_arguments(self, num_probe_query, num_to_rerank):
        self.num_probe_query = num_probe_query
        self.num_to_rerank = num_to_rerank

    def query(self, v, n):
        query_centroid_ids = self.centroid_search(v, self.num_probe_query)
        doc_ids = self.index.query(
            v,
            top_k=n,
            num_to_rerank=self.num_to_rerank,
            query_centroid_ids=query_centroid_ids,
        )
        return np.array(doc_ids, dtype=np.uint32)

    def batch_query(self, X, n):
        self.res = np.zeros((X.shape[0], n), dtype=np.uint32)
        for i, v in enumerate(X):
            self.res[i] = self.query(v, n)

    def get_batch_results(self):
        return self.res

    def __str__(self):
        return "DESSERT(num_tables=%d, hashes_per_table=%d, n_clusters=%d, num_probe_query=%d, num_to_rerank=%d)" % (
            self.num_tables,
            self.hashes_per_table,
            self.n_clusters,
            self.num_probe_query,
            self.num_to_rerank,
        )
