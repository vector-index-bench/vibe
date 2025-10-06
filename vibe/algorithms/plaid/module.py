import torch
from fast_plaid import search

from ..base.module import BaseANN


class PLAID(BaseANN):

    def __init__(self, metric):
        self.metric = metric
        self.index = None
        self.index_path = None

    def fit(self, X):
        embeddings, counts = X

        documents_embeddings = []
        start = 0
        for count in counts:
            doc_embeddings = embeddings[start:start + count]
            documents_embeddings.append(torch.tensor(doc_embeddings))
            start += count

        import tempfile
        self.index_path = tempfile.mkdtemp(prefix="plaid_index_")

        self.index = search.FastPlaid(index=self.index_path)
        self.index.create(documents_embeddings=documents_embeddings)

    def set_query_arguments(self):
        pass

    def query(self, v, n):
        queries_embeddings = torch.tensor(v).unsqueeze(0)

        scores = self.index.search(
            queries_embeddings=queries_embeddings,
            top_k=n,
        )[0]

        return [idx for idx, score in scores]

    def __str__(self):
        return f"PLAID(metric={self.metric})"
