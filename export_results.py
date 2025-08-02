# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "h5py",
#     "scikit-learn",
#     "tqdm",
# ]
# ///

import numpy as np
import pathlib
import argparse
import h5py
import polars as pl
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict


def get_recall_values(dataset_distances, run_distances, count, epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = dataset_distances[i][count - 1] + epsilon
        recalls[i] = (run_distances[i][:count] <= t).sum()
    return recalls


def compute_metrics(path, data_dir):
    true_distances = {}
    for data_path in pathlib.Path(data_dir).glob("**/*.hdf5"):
        name = data_path.name.replace(".hdf5", "")
        with h5py.File(data_path) as hfp:
            true_distances[name] = hfp["distances"][:]

    with h5py.File(path, "r+") as hfp:
        for query_params in hfp.keys():
            dataset = hfp[query_params].attrs["dataset"]
            if "recalls" not in hfp[query_params]:
                hfp[query_params]["recalls"] = get_recall_values(
                    true_distances[dataset], hfp[query_params]["distances"], hfp[query_params].attrs["count"]
                )


def export_results(path, data_dir):
    try:
        with h5py.File(path, "r") as hfp:
            for query_params in hfp.keys():
                k = hfp[query_params].attrs["count"]
                dataset = hfp[query_params].attrs["dataset"]
                algo = hfp[query_params].attrs["algo"]
                params = hfp[query_params].attrs["name"] + "|" + query_params
                times = hfp[query_params]["times"][:]
                n_queries = len(times)
                recalls = hfp[query_params]["recalls"][:] / k
                qps = 1 / hfp[query_params].attrs["best_search_time"]
                summary = dict(
                    k=k,
                    dataset=dataset,
                    algorithm=algo,
                    params=params,
                    qps=qps,
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
                yield dataset, summary, detail
    except BlockingIOError:
        print(f"Unable to open {path} -- skipping")


def _process_file(file_path, data_dir):
    compute_metrics(file_path, data_dir)

    summaries = []
    details = defaultdict(list)

    for dataset, summary, detail in export_results(file_path, data_dir):
        summaries.append(summary)
        details[dataset].append(detail)

    return summaries, details


def export_all_results(path, data_dir, parallelism, output_summary, output_dir):
    root_path = pathlib.Path(path)
    data_dir = pathlib.Path(data_dir)
    output_summary = pathlib.Path(output_summary)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = list(root_path.glob("**/*.hdf5"))
    if not hdf5_files:
        print("No .hdf5 files found — nothing to do.")
        return

    summaries = []
    dataset_details = defaultdict(list)

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(_process_file, file_path, data_dir) for file_path in hdf5_files]

        with tqdm(total=len(futures), desc="Exporting results") as pbar:
            for future in concurrent.futures.as_completed(futures):
                file_summaries, details_map = future.result()
                summaries.extend(file_summaries)
                for dataset, detail_list in details_map.items():
                    dataset_details[dataset].extend(detail_list)

                pbar.update(1)

    for dataset, detail_frames in dataset_details.items():
        if detail_frames:
            detail_df = pl.concat(detail_frames)
            detail_df.write_parquet(output_dir / f"{dataset}__detail.parquet")

    if summaries:
        pl.DataFrame(summaries).write_parquet(output_summary)


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
    hdf5_files = list(pathlib.Path(data_dir).glob("**/*.hdf5"))

    stats = []
    with tqdm(total=len(hdf5_files), desc="Exporting query stats") as pbar:
        for path in hdf5_files:
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
            pbar.update(1)

    stats = pl.concat(stats)
    stats.write_parquet(output_file)


def export_data_info(data_dir, output_file):
    if output_file.exists():
        print(f"Output file {output_file} already exists -- skipping")
        return

    hdf5_files = list(pathlib.Path(data_dir).glob("**/*.hdf5"))

    stats = []
    with tqdm(total=len(hdf5_files), desc="Exporting dataset info") as pbar:
        for path in hdf5_files:
            with h5py.File(path) as hfp:
                name = path.name.replace(".hdf5", "")
                n, d = hfp["train"][:].shape
                stats.append(dict(dataset=name, n=n, dimensions=d))
            pbar.update(1)

    stats = pl.DataFrame(stats)
    stats.write_parquet(output_file)


def mahalanobis_distance_batch(V, Q):
    if V.ndim != 2:
        raise ValueError("Input matrix 'V' must be 2-dimensional (each row is a vector).")
    if Q.ndim != 2:
        raise ValueError("Input matrix 'Q' must be 2-dimensional.")
    if V.shape[1] != Q.shape[1]:
        raise ValueError(f"Dimension mismatch: V columns ({V.shape[1]}) must equal " f"Q columns ({Q.shape[1]}).")
    if Q.shape[0] < 2:
        raise ValueError("Input matrix 'Q' must have at least 2 samples (rows).")

    mu = np.mean(Q, axis=0)
    cov_matrix = np.cov(Q, rowvar=False, ddof=1)

    diff = V - mu

    if cov_matrix.ndim == 0:
        variance = cov_matrix.item()
        if np.isclose(variance, 0):
            is_zero_diff = np.all(np.isclose(diff, 0), axis=1)
            distances_sq = np.full(V.shape[0], np.inf)
            distances_sq[is_zero_diff] = 0.0
        else:
            inv_cov = 1.0 / variance
            distances_sq = (diff**2 * inv_cov).flatten()
    else:
        try:
            inv_cov_matrix = np.linalg.pinv(cov_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular and pseudo-inverse could not be computed.")

        temp = diff @ inv_cov_matrix
        distances_sq = np.sum(temp * diff, axis=1)

    negative_close_to_zero = (distances_sq < 0) & np.isclose(distances_sq, 0)
    distances_sq[negative_close_to_zero] = 0.0

    if np.any(distances_sq < 0):
        raise ValueError("Squared Mahalanobis distance is negative for some inputs")

    return np.sqrt(distances_sq)


def export_pca_and_mahalanobis(data_dir, output_file, sample_size=2000):
    if output_file.exists():
        print(f"Output file {output_file} already exists -- skipping")
        return

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    hdf5_files = list(pathlib.Path(data_dir).glob("**/*.hdf5"))

    pcas = []
    with tqdm(total=len(hdf5_files), desc="Exporting PCA and Mahalanobis data") as pbar:
        for path in hdf5_files:
            gen = np.random.default_rng(1234)
            with h5py.File(path) as hfp:
                name = path.name.replace(".hdf5", "")
                train = hfp["train"][:]
                test = hfp["test"][:]

                if name.endswith("-binary"):
                    train = np.unpackbits(train).reshape(train.shape[0], -1).astype(np.float32)
                    test = np.unpackbits(test).reshape(test.shape[0], -1).astype(np.float32)

                if train.dtype != np.float32:
                    train = train.astype(np.float32)
                    test = test.astype(np.float32)

                mahalanobis_sample_train = train[np.sort(gen.choice(train.shape[0], 100_000, replace=False))]
                train_sample_indices = np.sort(gen.choice(train.shape[0], sample_size, replace=False))
                train = train[train_sample_indices]

                data_to_data = mahalanobis_distance_batch(train, mahalanobis_sample_train)
                query_to_data = mahalanobis_distance_batch(test, mahalanobis_sample_train)

                mahalanobis_combined = np.concatenate((data_to_data, query_to_data))

                pca = PCA(n_components=2, random_state=1)
                scaler = StandardScaler()

                combined = np.vstack([train, test])
                combined_scaled = scaler.fit_transform(combined)
                combined_pca = pca.fit_transform(combined_scaled)
                df = pl.DataFrame(
                    dict(
                        dataset=name,
                        part=np.concatenate((np.repeat("train", train.shape[0]), np.repeat("test", test.shape[0]))),
                        x=combined_pca[:, 0],
                        y=combined_pca[:, 1],
                        mahalanobis_distance_to_data=mahalanobis_combined,
                    )
                )
                pcas.append(df)

            pbar.update(1)

    pcas = pl.concat(pcas)
    pcas.write_parquet(output_file)


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--data", help="the path to the directory containing datasets", default="data")
    aparser.add_argument("--output", help="the path to the output directory", default="results")
    aparser.add_argument("--parallelism", type=int, help="number of parallel processes to use", default=1)

    args = aparser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_summary = output_dir / "summary.parquet"
    output_stats = output_dir / "stats.parquet"
    output_info = output_dir / "data-info.parquet"
    output_pca_mahalanobis = output_dir / "data-pca-mahalanobis.parquet"

    export_all_results(args.results, args.data, args.parallelism, output_summary, output_dir)
    export_query_stats(args.data, output_stats)
    export_data_info(args.data, output_info)
    export_pca_and_mahalanobis(args.data, output_pca_mahalanobis)


if __name__ == "__main__":
    main()
