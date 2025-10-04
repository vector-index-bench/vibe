import numpy as np
import torch

from ..base.module import BaseANN


class Chamfer(BaseANN):
    def __init__(self, metric):
        self.metric = metric
        self.train_embeddings = None
        self.train_counts = None
        self.batch_results = None
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    def fit(self, X):
        """
        Fit the index with training data.

        Args:
            X: Tuple of (embeddings, counts) where:
                - embeddings: numpy array of shape (total_vectors, dim) containing all vectors concatenated
                - counts: numpy array of shape (n_docs,) containing the number of vectors per document
        """
        embeddings, counts = X
        self.train_embeddings = torch.from_numpy(embeddings).to(self.device)
        self.train_counts = torch.from_numpy(counts).to(self.device)

    def batch_query(self, X, n):
        """
        Perform batch query to find n nearest neighbors for each query.

        Args:
            X: Tuple of (embeddings, counts) where:
                - embeddings: numpy array of shape (total_vectors, dim) containing all query vectors concatenated
                - counts: numpy array of shape (n_queries,) containing the number of vectors per query
            n: Number of nearest neighbors to return
        """
        query_embeddings, query_counts = X
        query_embeddings = torch.from_numpy(query_embeddings).to(self.device)
        query_counts = torch.from_numpy(query_counts).to(self.device)

        n_queries = len(query_counts)
        n_train = len(self.train_counts)

        # Compute cumulative indices for splitting concatenated vectors
        query_splits = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), query_counts]), dim=0)
        train_splits = torch.cumsum(torch.cat([torch.tensor([0], device=self.device), self.train_counts]), dim=0)

        # Compute Chamfer distance for each query-document pair
        chamfer_distances = torch.zeros(n_queries, n_train, device=self.device)

        for i in range(n_queries):
            # Extract query vectors for this query
            q_start, q_end = query_splits[i], query_splits[i + 1]
            query_vecs = query_embeddings[q_start:q_end]  # shape: (n_query_vecs, dim)

            for j in range(n_train):
                # Extract train vectors for this document
                t_start, t_end = train_splits[j], train_splits[j + 1]
                train_vecs = self.train_embeddings[t_start:t_end]  # shape: (n_train_vecs, dim)

                # Compute pairwise distances between query and train vectors
                if self.metric == "cosine":
                    # Normalize vectors
                    query_vecs_norm = torch.nn.functional.normalize(query_vecs, p=2, dim=1)
                    train_vecs_norm = torch.nn.functional.normalize(train_vecs, p=2, dim=1)
                    # Cosine similarity -> cosine distance
                    similarities = torch.mm(query_vecs_norm, train_vecs_norm.t())
                    pairwise_dists = 1 - similarities
                elif self.metric == "euclidean":
                    # Compute pairwise Euclidean distances
                    pairwise_dists = torch.cdist(query_vecs, train_vecs, p=2)
                elif self.metric == "ip":
                    # Inner product -> negative distance (larger is better)
                    pairwise_dists = -torch.mm(query_vecs, train_vecs.t())
                else:
                    raise ValueError(f"Unsupported metric: {self.metric}")

                # Chamfer distance: min over train vecs + min over query vecs
                min_query_to_train = torch.min(pairwise_dists, dim=1)[0].mean()
                min_train_to_query = torch.min(pairwise_dists, dim=0)[0].mean()
                chamfer_distances[i, j] = min_query_to_train + min_train_to_query

        # Find top-n nearest neighbors for each query
        _, indices = torch.topk(chamfer_distances, k=n, dim=1, largest=False)
        self.batch_results = indices.cpu().numpy()

    def get_batch_results(self):
        """
        Return the results from the last batch query.

        Returns:
            numpy array of shape (n_queries, n) containing indices of nearest neighbors
        """
        return self.batch_results

    def __str__(self):
        return f"Chamfer(metric={self.metric})"
