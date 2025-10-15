import numpy as np
import torch

from ..base.module import BaseANN


class Chamfer(BaseANN):
    def __init__(self, device):
        self.train_embeddings = None
        self.train_mask = None
        self.batch_results = None
        self.distances = None
        self.chamfer_scores = None
        self.device = device

    def pad_input(self, X):
        embeddings, counts = X

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if not isinstance(counts, np.ndarray):
            counts = np.asarray(counts, dtype=np.int64)
        counts = counts.astype(np.int64, copy=False)

        n_queries = len(counts)
        max_len = int(np.max(counts))
        dim = embeddings.shape[1]

        out = np.zeros((n_queries, max_len, dim), dtype=np.float32)

        start = 0
        for i, count in enumerate(counts):
            out[i, :count, :] = embeddings[start : start + count]
            start += count

        return out

    def fit(self, X):
        embeddings, counts = X
        self.train_embeddings = torch.from_numpy(self.pad_input((embeddings, counts))).to(self.device)
        counts_t = torch.as_tensor(counts, device=self.train_embeddings.device)

        self.train_mask = (
            torch.arange(self.train_embeddings.shape[1], device=counts_t.device)[None, :] < counts_t[:, None]
        )

        self.doc_mask_bias = torch.where(
            self.train_mask,
            torch.zeros(1, dtype=self.train_embeddings.dtype, device=self.train_embeddings.device),
            torch.full((), float("-inf"), dtype=self.train_embeddings.dtype, device=self.train_embeddings.device),
        )

    def batch_query_with_distances(self, X, n):
        return self.batch_query_padded_with_distances(self.pad_input(X), n)

    def batch_query_padded_with_distances(self, X, n):
        with torch.inference_mode():
            queries = torch.as_tensor(X)
            queries = queries.to(self.train_embeddings.device, dtype=self.train_embeddings.dtype)
            batch_size, _, _ = queries.shape

            query_mask = queries.abs().sum(dim=2) != 0

            docs_T = self.train_embeddings.transpose(1, 2)
            doc_bias = self.doc_mask_bias

            out_indices = torch.empty((batch_size, n), dtype=torch.long, device=queries.device)
            out_values = torch.empty((batch_size, n), dtype=queries.dtype, device=queries.device)

            for b in range(batch_size):
                q_active = queries[b][query_mask[b]]
                scores = torch.matmul(q_active, docs_T)
                scores = scores + doc_bias[:, None, :]

                max_per_q = scores.max(dim=2).values
                total_scores = max_per_q.sum(dim=1)
                del scores, max_per_q

                nearest = torch.topk(total_scores, n, largest=True)
                out_indices[b] = nearest.indices
                out_values[b] = nearest.values
                del nearest, total_scores

            return out_indices.detach().cpu().numpy(), (-out_values).detach().cpu().numpy()

    def batch_query(self, X, n):
        self.batch_results, _ = self.batch_query_with_distances(X, n)

    def batch_query_padded(self, X, n):
        self.batch_results, _ = self.batch_query_padded_with_distances(X, n)

    def get_batch_results(self):
        return self.batch_results

    def distance_to_indices(self, q, indices):
        with torch.inference_mode():
            qt = torch.as_tensor(q)
            qt = qt.to(self.train_embeddings.device, dtype=self.train_embeddings.dtype)

            q_mask = qt.abs().sum(dim=1) != 0
            q_active = qt[q_mask]

            idx = torch.as_tensor(indices, dtype=torch.long, device=self.train_embeddings.device)

            if int(idx.numel()) == 0:
                return []

            docs_T = self.train_embeddings.index_select(0, idx).transpose(1, 2)
            doc_bias = self.doc_mask_bias.index_select(0, idx)

            scores = torch.matmul(q_active.unsqueeze(0), docs_T)
            scores = scores + doc_bias[:, None, :]

            max_per_q = scores.max(dim=2).values
            total_scores = max_per_q.sum(dim=1)

            distances = (-total_scores).detach().cpu().numpy().astype(np.float32)
            return distances

    def __str__(self):
        return "Chamfer()"
