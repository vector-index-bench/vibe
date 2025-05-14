import json
import os
import re
import traceback
from typing import Any, Optional, Tuple

from vibe.definitions import Definition


def is_run(
    dataset_name: Optional[str] = None,
    count: Optional[int] = None,
    definition: Optional[Definition] = None,
    query_arguments: Optional[Any] = None,
    batch_mode: bool = False,
    gpu_mode: bool = False,
) -> bool:
    file_path, search_parameters = build_result_filepath(dataset_name, count, definition, query_arguments, batch_mode, gpu_mode)
    return os.path.exists(file_path)


def build_result_filepath(
    dataset_name: str,
    count: int,
    definition: Optional[Definition] = None,
    query_arguments: Optional[Any] = None,
    batch_mode: bool = False,
    gpu_mode: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Constructs the filepath for storing the results.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition, optional): The definition of the algorithm.
        query_arguments (Any, optional): Additional arguments for the query.
        batch_mode (bool, optional): If true, the batch mode is activated.
        gpu_mode (bool, optional): If true, the GPU mode is activated.

    Returns:
        A tuple of (filepath, search_parameters) where
        filepath (str): The constructed filepath
        search_parameters (str): String description of the search parameters
    """
    suffix = ""
    if gpu_mode:
        suffix = "-gpu"
    elif batch_mode:
        suffix = "-batch"

    d = ["results", dataset_name, str(count)]
    search_parameters = None
    if definition:
        d.append(definition.algorithm + suffix)
        index_parameters = re.sub(r"\W+", "_", json.dumps(definition.arguments, sort_keys=True)).strip("_")
        search_parameters = re.sub(r"\W+", "_", json.dumps(query_arguments, sort_keys=True)).strip("_")
        d.append(index_parameters + ".hdf5")
    return os.path.join(*d), search_parameters


def store_results(dataset_name: str, count: int, definition: Definition, query_arguments: Any, attrs, results, batch, gpu):
    """
    Stores results for an algorithm (and hyperparameters) running against a dataset in a HDF5 file.

    Args:
        dataset_name (str): The name of the dataset.
        count (int): The count of records.
        definition (Definition): The definition of the algorithm.
        query_arguments (Any): Additional arguments for the query.
        attrs (dict): Attributes to be stored in the file.
        results (list): Results to be stored.
        batch (bool): If True, the batch mode is activated.
        gpu (bool): If true, the GPU mode is activated.
    """
    import h5py

    filename, search_parameters = build_result_filepath(dataset_name, count, definition, query_arguments, batch, gpu)
    directory, _ = os.path.split(filename)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    with h5py.File(filename, "a") as f:
        if search_parameters not in f:
            f.create_group(search_parameters)
        result_group = f[search_parameters]

        for k, v in attrs.items():
            result_group.attrs[k] = v
        times = result_group.create_dataset("times", (len(results),), "f")
        neighbors = result_group.create_dataset("neighbors", (len(results), count), "i")
        distances = result_group.create_dataset("distances", (len(results), count), "f")

        for i, (time, ds) in enumerate(results):
            times[i] = time
            neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
            distances[i] = [d for n, d in ds] + [float("inf")] * (count - len(ds))


def load_all_results(dataset: str, count: int, batch_mode: bool = False, gpu_mode: bool = False):
    """
    Loads all the results from the HDF5 files in the specified path.

    Args:
        dataset (str): The name of the dataset.
        count (int): The count of records.
        batch_mode (bool, optional): If True, the batch mode is activated.
        gpu_mode (bool, optional): If True, the GPU mode is activated.

    Yields:
        tuple: A tuple containing properties as a dictionary and an h5py file object.
    """
    import h5py

    result_directory, _ = build_result_filepath(dataset, count)
    for root, _, files in os.walk(result_directory):
        for filename in files:
            if os.path.splitext(filename)[-1] != ".hdf5":
                continue
            try:
                with h5py.File(os.path.join(root, filename), "r+") as f:
                    for group in f.keys():
                        g = f[group]
                        properties = dict(g.attrs)
                        if "gpu_mode" in properties and gpu_mode != properties["gpu_mode"]:
                            continue
                        if "batch_mode" in properties and batch_mode != properties["batch_mode"]:
                            continue
                        yield properties, g
            except Exception:
                print(f"Was unable to read {filename}")
                traceback.print_exc()