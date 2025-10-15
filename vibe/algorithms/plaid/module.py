import torch
from fast_plaid import search

from ..base.module import BaseANN


class PLAID(BaseANN):
    def __init__(self, metric, nbits, kmeans_iters):
        self.metric = metric
        self.nbits = nbits
        self.kmeans_iters = kmeans_iters
        self.index = None
        self.index_path = None

    def fit(self, X):
        embeddings, counts = X

        documents_embeddings = []
        start = 0
        for count in counts:
            doc_embeddings = embeddings[start : start + count]
            documents_embeddings.append(torch.tensor(doc_embeddings))
            start += count

        import tempfile

        self.index_path = tempfile.mkdtemp(prefix="plaid_index_")

        self.index = search.FastPlaid(index=self.index_path)
        self.index.create(documents_embeddings=documents_embeddings)

    def set_query_arguments(self, n_full_scores, n_ivf_probe):
        self.n_full_scores = n_full_scores
        self.n_ivf_probe = n_ivf_probe

    def query(self, v, n):
        queries_embeddings = torch.tensor(v).unsqueeze(0)

        scores = self.index.search(
            queries_embeddings=queries_embeddings,
            top_k=n,
        )[0]

        return [idx for idx, score in scores]

    def batch_query(self, X, n):
        scores = self.index.search(
            queries_embeddings=X,
            top_k=n,
        )

        self.res = []
        for res in scores:
            self.res.append([idx for idx, score in res])

    def get_batch_results(self):
        return np.array(self.res)

    def __str__(self):
        return "PLAID(nbits=%d, kmeans_iters=%d, n_full_scores=%d, n_ivf_probe=%d)" % (
            self.nbits,
            self.kmeans_iters,
            self.n_full_scores,
            self.n_ivf_probe,
        )

    def __del__(self):
        import shutil

        if self.index_path and os.path.exists(self.index_path):
            try:
                shutil.rmtree(self.index_path)
            except:
                pass
