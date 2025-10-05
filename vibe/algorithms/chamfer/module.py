import numpy as np
import torch

from ..base.module import BaseANN


class Chamfer(BaseANN):
    def __init__(self, metric):
        if metric != "chamfer":
            raise ValueError(f"Only 'chamfer' metric is supported, got: {metric}")

        self.metric = metric
        self.train_counts = None
        self.batch_results = None

    def fit(self, X):
        """
        Fit the index with training data.

        Args:
            X: Tuple of (embeddings, counts) where:
                - embeddings: numpy array of shape (total_vectors, dim) containing all vectors concatenated
                - counts: numpy array of shape (n_docs,) containing the number of vectors per document
        """
        embeddings, counts = X
        self.train_counts = counts

        n_docs = len(counts)
        max_doc_length = int(np.max(counts))
        embedding_dim = embeddings.shape[1]

        self.train_embeddings_padded = np.zeros((n_docs, max_doc_length, embedding_dim), dtype=np.float32)

        start_idx = 0
        for i, count in enumerate(counts):
            end_idx = start_idx + count
            self.train_embeddings_padded[i, :count, :] = embeddings[start_idx:end_idx]
            start_idx = end_idx

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
        n_queries = len(query_counts)
        n_train = len(self.train_counts)

        chamfer_scores = np.zeros((n_queries, n_train), dtype=np.float32)

        query_start_idx = 0
        for i, query_count in enumerate(query_counts):
            query_end_idx = query_start_idx + query_count
            query_vecs = query_embeddings[query_start_idx:query_end_idx]
            scores = np.matmul(query_vecs, self.train_embeddings_padded.transpose(0, 2, 1))
            max_scores_per_query_term = np.max(scores, axis=2)
            chamfer_scores[i] = np.sum(max_scores_per_query_term, axis=1)
            query_start_idx = query_end_idx

        indices = np.argsort(-chamfer_scores, axis=1)[:, :n]
        distances = -np.take_along_axis(chamfer_scores, indices, axis=1)

        self.batch_results = indices
        self.distances = distances
        self.chamfer_scores = chamfer_scores

    def batch_query_with_distances(self, X, n, return_avg_dist=False):
        """
        Perform batch query to find n nearest neighbors for each query, with optional average distances.

        Args:
            X: Tuple of (embeddings, counts) where:
                - embeddings: numpy array of shape (total_vectors, dim) containing all query vectors concatenated
                - counts: numpy array of shape (n_queries,) containing the number of vectors per query
            n: Number of nearest neighbors to return
            return_avg_dist: If True, return average distance to all training documents

        Returns:
            If return_avg_dist is False:
                Tuple of (indices, distances)
            If return_avg_dist is True:
                Tuple of ((indices, distances), avg_distances)
        """
        self.batch_query(X, n)
        indices = self.batch_results
        distances = self.distances

        if return_avg_dist:
            avg_distances = -np.mean(self.chamfer_scores, axis=1)
            return (indices, distances), avg_distances
        else:
            return indices, distances

    def get_batch_results(self):
        """
        Return the results from the last batch query.

        Returns:
            numpy array of shape (n_queries, n) containing indices of nearest neighbors
        """
        return self.batch_results

    def __str__(self):
        return f"Chamfer(metric={self.metric})"


class ChamferGPU(BaseANN):
    def __init__(self, metric):
        if metric != "chamfer":
            raise ValueError(f"Only 'chamfer' metric is supported, got: {metric}")
        self.metric = metric
        self.train_embeddings_padded = None
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

        n_docs = len(counts)
        max_doc_length = int(np.max(counts))
        embedding_dim = embeddings.shape[1]

        train_embeddings_padded = np.zeros((n_docs, max_doc_length, embedding_dim), dtype=np.float32)

        start_idx = 0
        for i, count in enumerate(counts):
            end_idx = start_idx + count
            train_embeddings_padded[i, :count, :] = embeddings[start_idx:end_idx]
            start_idx = end_idx

        self.train_embeddings_padded = torch.from_numpy(train_embeddings_padded).to(self.device)
        self.train_counts = counts

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
        n_queries = len(query_counts)
        n_train = len(self.train_counts)

        chamfer_scores = torch.zeros((n_queries, n_train), dtype=torch.float32, device=self.device)

        query_start_idx = 0
        for i, query_count in enumerate(query_counts):
            query_end_idx = query_start_idx + query_count
            query_vecs = query_embeddings[query_start_idx:query_end_idx]
            query_vecs_gpu = torch.from_numpy(query_vecs).to(self.device)
            scores = torch.matmul(query_vecs_gpu, self.train_embeddings_padded.transpose(1, 2))
            max_scores_per_query_term = torch.max(scores, dim=2)[0]
            chamfer_scores[i] = torch.sum(max_scores_per_query_term, dim=1)
            query_start_idx = query_end_idx

        distances, indices = torch.topk(chamfer_scores, k=n, dim=1, largest=True)
        distances = -distances

        self.batch_results = indices.cpu().numpy()
        self.distances = distances.cpu().numpy()
        self.chamfer_scores = chamfer_scores

    def batch_query_with_distances(self, X, n, return_avg_dist=False):
        """
        Perform batch query to find n nearest neighbors for each query, with optional average distances.

        Args:
            X: Tuple of (embeddings, counts) where:
                - embeddings: numpy array of shape (total_vectors, dim) containing all query vectors concatenated
                - counts: numpy array of shape (n_queries,) containing the number of vectors per query
            n: Number of nearest neighbors to return
            return_avg_dist: If True, return average distance to all training documents

        Returns:
            If return_avg_dist is False:
                Tuple of (indices, distances)
            If return_avg_dist is True:
                Tuple of ((indices, distances), avg_distances)
        """
        self.batch_query(X, n)
        indices = self.batch_results
        distances = self.distances

        if return_avg_dist:
            avg_distances = -torch.mean(self.chamfer_scores, dim=1).cpu().numpy()
            return (indices, distances), avg_distances
        else:
            return indices, distances

    def get_batch_results(self):
        """
        Return the results from the last batch query.

        Returns:
            numpy array of shape (n_queries, n) containing indices of nearest neighbors
        """
        return self.batch_results

    def __str__(self):
        return f"ChamferGPU(metric={self.metric})"
