import argparse
import logging
import logging.config
import os
import glob
import random
import sys
from typing import List

from .definitions import Definition, InstantiationStatus, algorithm_status, get_definitions, list_algorithms
from .results import is_run
from .util import download, replace
from .workers import create_workers_and_execute

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("vibe")


def positive_int(input_str: str) -> int:
    """
    Validates if the input string can be converted to a positive integer.

    Args:
        input_str (str): The input string to validate and convert to a positive integer.

    Returns:
        int: The validated positive integer.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a positive integer.
    """
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer")

    return i


def filter_disabled_algorithms(definitions: List[Definition]) -> List[Definition]:
    """
    Excludes disabled algorithms from the given list of definitions.

    This function filters out the algorithm definitions that are marked as disabled in their `config.yml`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.

    Returns:
        List[Definition]: A list of algorithm definitions excluding any that are disabled.
    """
    return [d for d in definitions if not d.disabled]


def filter_algorithms_by_device(definitions: List[Definition], gpu: bool) -> List[Definition]:
    """
    Filters algorithms based on their device requirement (GPU or CPU).

    Args:
        definitions (List[Definition]): A list of algorithm definitions.
        gpu (bool): If True, filters for algorithms requiring GPU. If False, filters for algorithms not requiring GPU (CPU-only).

    Returns:
        List[Definition]: A list of algorithm definitions filtered according to the specified device requirement.
    """
    if gpu:
        return [d for d in definitions if d.gpu]
    else:
        return [d for d in definitions if not d.gpu]


def limit_algorithms(definitions: List[Definition], limit: int) -> List[Definition]:
    """
    Limits the number of algorithm definitions based on the given limit.

    If the limit is negative, all definitions are returned. For valid
    sampling, `definitions` should be shuffled before `limit_algorithms`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.
        limit (int): The maximum number of definitions to return.

    Returns:
        List[Definition]: A trimmed list of algorithm definitions.
    """
    return definitions if limit < 0 else definitions[:limit]


def filter_by_available_singularity_images(definitions):
    available_images = set([os.path.basename(x).replace(".sif", "") for x in glob.glob("images/*.sif")])
    missing_singularity_images = set(d.singularity_image for d in definitions).difference(available_images)

    if missing_singularity_images:
        logger.info(f"not all singularity images available, only: {available_images}")
        logger.info(f"missing singularity images: {missing_singularity_images}")
        definitions = [d for d in definitions if d.singularity_image in available_images]

    return definitions


def parse_dataset_string(s):
    """
    Parse dimension, distance, and data type from dataset strings like
    "landmark-nomic-vision-768-cosine" or "wiki-mxbai-1024-cosine-binary".

    Args:
        s: Input string to parse

    Returns:
        tuple: (dimension, distance, point_type)
    """
    parts = s.split("-")

    # Find the dimension
    dimension = None
    dimension_index = None

    for i, part in enumerate(parts):
        if part.isdigit():
            dimension = int(part)
            dimension_index = i
            break

    if dimension is None:
        raise ValueError(f"No dimension found in dataset name: {s}")

    # Get distance type, otherwise default to "normalized"
    if dimension_index + 1 < len(parts):
        distance = parts[dimension_index + 1]
    else:
        distance = "normalized"

    # Get data type if specified, otherwise default to "float"
    if dimension_index + 2 < len(parts):
        point_type = parts[dimension_index + 2]
    else:
        point_type = "float"

    return dimension, distance, point_type


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
    )
    parser.add_argument(
        "-k", "--count", default=100, type=positive_int, help="the number of near neighbours to search for"
    )
    parser.add_argument(
        "--definitions",
        metavar="FOLDER",
        help="base directory of algorithms. Algorithm definitions expected at 'FOLDER/*/config.yml'",
        default="vibe/algorithms",
    )
    parser.add_argument("--algorithm", metavar="NAME", help="run only the named algorithm", default=None)
    parser.add_argument("--module", metavar="NAME", help="run only algorithms in the named module", default=None)
    parser.add_argument(
        "--list-algorithms", help="print the names of all known algorithms and exit", action="store_true"
    )
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only the best result",
        default=5,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1 if no timeout should be set",
        default=4 * 3600,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, then will run everything locally (inside the same process) rather than using Singularity",
    )
    parser.add_argument("--gpu", action="store_true", help="If set, run the benchmark in GPU mode")
    parser.add_argument(
        "--max-n-algorithms", type=int, help="Max number of algorithms to run (just used for testing)", default=-1
    )
    parser.add_argument("--run-disabled", help="run algorithms that are disabled in algos.yml", action="store_true")
    parser.add_argument(
        "--parallelism", type=positive_int, help="Number of indexes to benchmark in parallel", default=1
    )

    args = parser.parse_args()

    if args.timeout == -1:
        args.timeout = None

    if args.gpu:
        args.parallelism = 1

    return args


def filter_already_run_definitions(
    definitions: List[Definition], dataset: str, count: int, gpu: bool, force: bool
) -> List[Definition]:
    """Filters out the algorithm definitions based on whether they have already been run or not.

    This function checks if there are existing results for each definition by constructing the
    result filename from the algorithm definition and the provided arguments. If there are no
    existing results or if the parameter `force=True`, the definition is kept. Otherwise, it is
    discarded.

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.
        dataset (str): The name of the dataset to load training points from.
        count (int): The number of near neighbours to search for (only used in file naming convention).
        gpu (bool): If set, run the benchmark in GPU mode (only used in file naming convention).
        force (bool): If set, re-run algorithms even if their results already exist.

    Returns:
        List[Definition]: A list of algorithm definitions that either have not been run or are
                          forced to be re-run.
    """
    filtered_definitions = []

    for definition in definitions:
        not_yet_run = [
            query_args
            for query_args in (definition.query_argument_groups or [[]])
            if force or not is_run(dataset, count, definition, query_args, gpu)
        ]

        if not_yet_run:
            definition = (
                replace(definition, query_argument_groups=not_yet_run)
                if definition.query_argument_groups
                else definition
            )
            filtered_definitions.append(definition)

    return filtered_definitions


def check_module_import_and_constructor(df: Definition) -> bool:
    """
    Verifies if the algorithm module can be imported and its constructor exists.

    This function checks if the module specified in the definition can be imported.
    Additionally, it verifies if the constructor for the algorithm exists within the
    imported module.

    Args:
        df (Definition): A definition object containing the module and constructor
        for the algorithm.

    Returns:
        bool: True if the module can be imported and the constructor exists, False
        otherwise.
    """
    status = algorithm_status(df)
    if status == InstantiationStatus.NO_CONSTRUCTOR:
        raise Exception(
            f"{df.module}.{df.constructor}({df.arguments}): error: the module '{df.module}' does not expose the named constructor"
        )
    if status == InstantiationStatus.NO_MODULE:
        logging.warning(
            f"{df.module}.{df.constructor}({df.arguments}): the module '{df.module}' could not be loaded; skipping"
        )
        return False

    return True


def main():
    args = parse_arguments()

    if args.list_algorithms:
        for algorithm in sorted(list_algorithms(args.definitions)):
            print(algorithm)
        sys.exit(0)

    if not os.path.exists("data"):
        os.mkdir("data")

    dimension, distance, point_type = parse_dataset_string(args.dataset)

    try:
        dataset_url = f"https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/{args.dataset}.hdf5"
        hdf5_filename = os.path.join("data", f"{args.dataset}.hdf5")
        download(dataset_url, hdf5_filename)
    except Exception:
        raise Exception(
            f"""Cannot download {args.dataset}. Make sure you have entered a valid dataset name.
           You can also try creating the dataset manually using create_dataset.sh"""
        )

    definitions: List[Definition] = get_definitions(
        dimension=dimension,
        point_type=point_type,
        distance_metric=distance,
        count=args.count,
        base_dir=args.definitions,
    )
    random.shuffle(definitions)

    definitions = filter_already_run_definitions(
        definitions,
        dataset=args.dataset,
        count=args.count,
        gpu=args.gpu,
        force=args.force,
    )

    if args.module:
        logger.info(f"running only algorithms for module {args.module}")
        definitions = [d for d in definitions if d.module.replace("vibe.algorithms.", "") == args.module]

    if args.algorithm:
        logger.info(f"running only {args.algorithm}")
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    if not args.local:
        definitions = filter_by_available_singularity_images(definitions)
    else:
        definitions = list(filter(check_module_import_and_constructor, definitions))

    definitions = filter_disabled_algorithms(definitions) if not args.run_disabled else definitions
    definitions = limit_algorithms(definitions, args.max_n_algorithms)
    definitions = filter_algorithms_by_device(definitions, args.gpu)

    if len(definitions) == 0:
        raise Exception("Nothing to run")
    else:
        logger.info(f"Order: {definitions}")

    create_workers_and_execute(definitions, args)
