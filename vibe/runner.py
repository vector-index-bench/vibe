import argparse
import json
import logging
import subprocess
import os
import time
from typing import Tuple, List

from vibe.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm


def run_individual_query(
    algo: BaseANN, X_train, X_test, distance: str, count: int, run_count: int, batch: bool, gpu: bool
) -> Tuple[dict, list]:
    """Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_train (numpy.array): The training data.
        X_test (numpy.array): The testing data.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.
        gpu (bool): Flag to indicate whether to run in GPU mode or not.

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    from .distance import metrics

    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))

        def single_query(v) -> Tuple[float, List[Tuple[int, float]]]:
            """Executes a single query on an instantiated, ANN algorithm.

            Args:
                v (numpy.array): Vector to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            if prepared_queries:
                algo.prepare_query(v, count)
                start = time.perf_counter()
                algo.run_prepared_query()
                total = time.perf_counter() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.perf_counter()
                candidates = algo.query(v, count)
                total = time.perf_counter() - start

            # make sure all returned indices are unique
            # assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = [
                (int(idx), float(dist))
                for idx, dist in zip(candidates, metrics[distance].distance(v, X_train[candidates]))
            ]
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """Executes a batch of queries on an instantiated, ANN algorithm.

            Args:
                X (numpy.array): Array containing multiple vectors to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.perf_counter()
                algo.run_batch_query()
                total = time.perf_counter() - start
            else:
                start = time.perf_counter()
                algo.batch_query(X, count)
                total = time.perf_counter() - start
            results = algo.get_batch_results()
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(X))] * len(X)

            # make sure all returned indices are unique
            # for res in results:
            #     assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            candidates = [
                [
                    (int(idx), float(dist))
                    for idx, dist in zip(single_results, metrics[distance].distance(v, X_train[single_results]))
                ]
                for v, single_results in zip(X, results)
            ]
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch or gpu:
            results = batch_query(X_test)
        else:
            results = [single_query(x) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "gpu_mode": gpu,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str):
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading,
    and the file handle and the dimension of the dataset are returned.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    import h5py

    hdf5_filename = get_dataset_fn(dataset_name)
    hdf5_file = h5py.File(hdf5_filename, "r")

    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"])
    return hdf5_file, dimension


def load_and_transform_dataset(dataset_name: str):
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
    """
    import numpy

    D, dimension = get_dataset(dataset_name)
    train = numpy.array(D["train"])
    test = numpy.array(D["test"])
    distance = D.attrs["distance"]

    print(f"Got a train set of size ({train.shape[0]} * {dimension})")
    print(f"Got {len(test)} queries")

    return train, test, distance


def load_ood_data(dataset_name: str):
    import numpy

    D, dimension = get_dataset(dataset_name)
    if "learn" in D and "learn_neighbors" in D:
        learn = numpy.array(D["learn"])
        learn_neighbors = numpy.array(D["learn_neighbors"])
        return learn, learn_neighbors

    return None, None


def build_index(algo: BaseANN, constructor_name, X_train, X_learn, X_learn_neighbors) -> Tuple:
    """Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        constructor_name (str): Name of the constructor.
        X_train (Any): The training data.
        X_learn (Any): Sample from the query distribution.
        X_learn_neighbors (Any): Nearest neighbors of the query sample in the training data.

    Returns:
        Tuple: The build time and index size.
    """
    memory_usage_before = algo.get_memory_usage()
    t0 = time.time()

    if X_learn is None or X_learn_neighbors is None:
        if not hasattr(algo, "fit"):
            raise NotImplementedError(f"{constructor_name} supports only OOD datasets")
        algo.fit(X_train)
    else:
        algo.fit_ood(X_train, X_learn, X_learn_neighbors)

    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before

    print("Built index in", build_time)
    print("Index size: ", index_size)

    return build_time, index_size


def run(definition: Definition, dataset_name: str, count: int, run_count: int, batch: bool) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        count (int): The number of results to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    from .results import store_results

    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(algo, "set_query_arguments"), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    X_train, X_test, distance = load_and_transform_dataset(dataset_name)
    if definition.ood:
        X_learn, X_learn_neighbors = load_ood_data(dataset_name)
    else:
        X_learn, X_learn_neighbors = None, None

    try:
        if hasattr(algo, "supports_prepared_queries"):
            algo.supports_prepared_queries()

        build_time, index_size = build_index(algo, definition.constructor, X_train, X_learn, X_learn_neighbors)

        query_argument_groups = definition.query_argument_groups or [[]]  # Ensure at least one iteration

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)

            descriptor, results = run_individual_query(
                algo, X_train, X_test, distance, count, run_count, batch, definition.gpu
            )

            descriptor.update(
                {
                    "build_time": build_time,
                    "index_size": index_size,
                    "algo": definition.algorithm,
                    "dataset": dataset_name,
                }
            )

            store_results(dataset_name, count, definition, query_arguments, descriptor, results, batch, definition.gpu)
    finally:
        pass


def run_from_cmdline():
    """Calls the function `run` using arguments from the command line. See `ArgumentParser` for
    arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument(
        "--dataset",
        help="Dataset to benchmark on.",
        required=True,
    )
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "vibe.algorithms.lorann"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Lorann"', required=True)
    parser.add_argument(
        "--count", help="K: Number of nearest neighbours for the algorithm to return.", required=True, type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If included, the algorithm will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument(
        "--gpu",
        help="If included, the algorithm will be run in GPU mode.",
        action="store_true",
    )
    parser.add_argument(
        "--ood",
        help="If included, the algorithm will be run in OOD mode.",
        action="store_true",
    )
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["cosine", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        singularity_image=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
        ood=args.ood,
        gpu=args.gpu,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_singularity(
    definition: Definition,
    dataset: str,
    count: int,
    runs: int,
    timeout: int,
    batch: bool,
) -> None:
    """Runs `run_from_cmdline` within a Singularity container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    image = "images/" + definition.singularity_image + ".sif"

    cmd = []
    if timeout is not None:
        cmd += ["timeout", str(timeout)]
    cmd += ["singularity", "exec"]
    if definition.gpu:
        cmd += ["--nv"]
    cmd += [
        image,
        "python3",
        "-u",
        "run_algorithm.py",
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if definition.gpu:
        cmd += ["--gpu"]
    if batch:
        cmd += ["--batch"]
    if definition.ood:
        cmd += ["--ood"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1
    ) as process:
        logger = logging.getLogger(f"vibe.{process.pid}")
        logger.info(f"Started process with PID: {process.pid}")

        for line in process.stdout:
            logger.info(line.rstrip())

        return_code = process.wait()
        logger.info(f"Process completed with return code: {return_code}")

        for line in process.stderr:
            logger.error(line.rstrip())
