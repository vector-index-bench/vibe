from multiprocessing.pool import ThreadPool


class BaseANN(object):
    """Base class for Approximate Nearest Neighbors (ANN) algorithms used in benchmarking."""

    def get_memory_usage(self):
        """Returns the current memory usage of this ANN algorithm instance in kilobytes.

        Returns:
            float: The current memory usage in kilobytes (for backwards compatibility), or None if
                this information is not available.
        """
        import psutil

        return psutil.Process().memory_full_info().uss / 1024

    def batch_query(self, X, n):
        """Performs multiple queries at once and lets the algorithm figure out how to handle it.

        The default implementation uses a ThreadPool to parallelize query processing.

        Args:
            X (numpy.array): An array of vectors to find the nearest neighbors of.
            n (int): The number of nearest neighbors to return for each query.
        Returns:
            None: self.get_batch_results() is responsible for retrieving batch result
        """
        pool = ThreadPool()
        self.res = pool.map(lambda q: self.query(q, n), X)

    def get_batch_results(self):
        """Retrieves the results of a batch query (from .batch_query()).

        Returns:
            numpy.array: An array of nearest neighbor results for each query in the batch.
        """
        return self.res

    def get_additional(self):
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """
        return {}

    def __str__(self):
        return self.name
