import dessert_py
import lorann

from ..base.module import BaseANN


class DESSERT(BaseANN):

    def __init__(self, metric, num_tables, hashes_per_table, n_clusters, kmeans_iters=10):
        self.metric = metric
        self.num_tables = num_tables
        self.hashes_per_table = hashes_per_table
        self.n_clusters = n_clusters
        self.kmeans_iters = kmeans_iters
        self.index = None
        self.num_to_rerank = None

    def fit(self, X):
        embeddings, counts = X

        n_vecs, embedding_dim = embeddings.shape

        kmeans = lorann.KMeans(n_clusters=self.n_clusters, iters=self.kmeans_iters, distance=lorann.IP)
        cluster_map = kmeans.train(embeddings, verbose=False)
        centroids = kmeans.get_centroids()

        self.index = dessert_py.DocRetrieval(
            dense_input_dimension=embedding_dim,
            num_tables=self.num_tables,
            hashes_per_table=self.hashes_per_table,
            centroids=centroids,
        )

        start = 0
        for doc_id, count in enumerate(counts):
            doc_embeddings = embeddings[start:start + count]
            self.index.add_doc(
                doc_id=str(doc_id),
                doc_embeddings=doc_embeddings,
            )
            start += count

    def set_query_arguments(self, num_to_rerank):
        self.num_to_rerank = num_to_rerank

    def query(self, v, n):
        doc_ids = self.index.query(
            v,
            num_to_rerank=self.num_to_rerank,
            top_k=n,
        )
        return [int(doc_id) for doc_id in doc_ids]

    def __str__(self):
        return (f"DESSERT(num_tables={self.num_tables}, hashes_per_table={self.hashes_per_table}, "
                f"n_clusters={self.n_clusters}, num_to_rerank={self.num_to_rerank})")
