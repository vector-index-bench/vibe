import numpy as np
import pathlib
import argparse
import h5py
import polars as pl
from icecream import ic


def export_results(path):
    with h5py.File(path) as hfp:
        for query_params in hfp.keys():
            k = hfp[query_params].attrs["count"]
            dataset = hfp[query_params].attrs["dataset"]
            algo = hfp[query_params].attrs["algo"]
            params = hfp[query_params].attrs["name"] + "|" + query_params
            times = hfp[query_params]["times"][:]
            n_queries = len(times)
            recalls = hfp[query_params]["metrics"]["knn"]["recalls"][:] / k
            summary = dict(
                k=k,
                dataset=dataset,
                algorithm=algo,
                params=params,
                avg_time=times.mean(),
                qps=n_queries / times.sum(),
                recall=recalls.mean(),
            )
            detail = pl.DataFrame(
                dict(
                    dataset=dataset,
                    query_index=np.arange(n_queries),
                    k=k,
                    algorithm=algo,
                    params=params,
                    time=times,
                    recall=recalls,
                )
            )
            yield summary, detail


def export_all_results(path, output_summary, output_detail):
    summaries = []
    details = []

    for path in pathlib.Path(path).glob("**/*.hdf5"):
        for summary, detail in export_results(path):
            summaries.append(summary)
            details.append(detail)

    summary = pl.DataFrame(summaries)
    summary.write_parquet(output_summary)

    detail = pl.concat(details)
    detail.write_parquet(output_detail)


def compute_lid(distances, k):
    w = distances[min(len(distances) - 1, k - 1)]
    half_w = 0.5 * w

    distances = distances[:k]
    distances = distances[distances > 1e-5]

    small = distances[distances < half_w]
    large = distances[distances >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

    return -valid / s


def export_query_stats(data_dir, output_file):
    stats = []
    for path in pathlib.Path(data_dir).glob("**/*.hdf5"):
        with h5py.File(path) as hfp:
            name = path.name.replace(".hdf5", "")
            distances = hfp["distances"][:]
            avg_distances = hfp["avg_distances"][:]
            metrics = dict(dataset=name, query_index=np.arange(distances.shape[0]))
            for k in [1, 10, 100]:
                if k > 1:
                    metrics[f"lid{k}"] = np.array([compute_lid(ds, k) for ds in distances])
                metrics[f"rc{k}"] = avg_distances / distances[:, k - 1]
            stats.append(pl.DataFrame(metrics))
    stats = pl.concat(stats)
    stats.write_parquet(output_file)


def export_data_info(data_dir, output_file):
    stats = []
    for path in pathlib.Path(data_dir).glob("**/*.hdf5"):
        with h5py.File(path) as hfp:
            name = path.name.replace(".hdf5", "")
            n, d = hfp["train"][:].shape
            stats.append(dict(dataset=name, n=n, dimensions=d))
    stats = pl.DataFrame(stats)
    stats.write_parquet(output_file)


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--data", help="the path to the directory containing datasets", default="data")
    aparser.add_argument(
        "--output-summary", help="the path to the parquet file storing the summaries", default="summary.parquet"
    )
    aparser.add_argument(
        "--output-detail", help="the path to the file storing the detail of each query", default="detail.parquet"
    )
    aparser.add_argument(
        "--output-stats", help="the path to the file storing the statistics of each query", default="stats.parquet"
    )
    aparser.add_argument(
        "--output-info", help="the path to the file storing the statistics of each dataset", default="data-info.parquet"
    )

    args = aparser.parse_args()
    export_all_results(args.results, args.output_summary, args.output_detail)
    export_query_stats(args.data, args.output_stats)
    export_data_info(args.data, args.output_info)


if __name__ == "__main__":
    main()
