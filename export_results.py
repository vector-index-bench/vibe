# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "h5py",
#     "scikit-learn",
#     "tqdm",
# ]
# ///

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
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


def compute_metrics(path, true_distances):
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
                qps = (
                    hfp[query_params].attrs["best_qps"]
                    if "best_qps" in hfp[query_params].attrs
                    else 1 / hfp[query_params].attrs["best_search_time"]
                )
                # relative run-to-run spread of QPS across the repeated runs
                # ((max - min) / max; 0.0 for a single run). Old result files
                # without per-run values export null.
                if "all_qps" in hfp[query_params].attrs:
                    all_qps = np.asarray(hfp[query_params].attrs["all_qps"], dtype=float)
                    qps_rel_spread = (
                        float((all_qps.max() - all_qps.min()) / all_qps.max())
                        if all_qps.size > 0 and all_qps.max() > 0
                        else 0.0
                    )
                else:
                    qps_rel_spread = None
                build_time = hfp[query_params].attrs["build_time"]
                index_size = hfp[query_params].attrs["index_size"]
                summary = dict(
                    k=k,
                    dataset=dataset,
                    algorithm=algo,
                    params=params,
                    qps=qps,
                    qps_rel_spread=qps_rel_spread,
                    recall=recalls.mean(),
                    build_time=build_time,
                    index_size=index_size,
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
                        build_time=build_time,
                        index_size=index_size,
                    )
                )
                yield dataset, summary, detail
    except BlockingIOError:
        print(f"Unable to open {path} -- skipping")


def _process_file(file_path, data_dir, true_distances):
    try:
        compute_metrics(file_path, true_distances)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Invalid results file {file_path} -- skipping")
        return None, None

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

    true_distances = {}
    for data_path in pathlib.Path(data_dir).glob("**/*.hdf5"):
        try:
            name = data_path.name.replace(".hdf5", "")
            with h5py.File(data_path) as hfp:
                true_distances[name] = hfp["distances"][:]
        except Exception as e:
            print(f"Invalid dataset file {data_path}: {e}")
            continue

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallelism) as pool:
        futures = [pool.submit(_process_file, file_path, data_dir, true_distances) for file_path in hdf5_files]

        try:
            with tqdm(total=len(futures), desc="Exporting results") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    file_summaries, details_map = future.result()
                    if file_summaries is not None and details_map is not None:
                        summaries.extend(file_summaries)
                        for dataset, detail_list in details_map.items():
                            dataset_details[dataset].extend(detail_list)
                    pbar.update(1)
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            print("Interrupted! Cancelling remaining tasks…")
            raise

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
            try:
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
            except Exception as e:
                print(f"Skipping invalid HDF5 file {path}: {e}")
            pbar.update(1)

    if stats:
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
            try:
                with h5py.File(path) as hfp:
                    name = path.name.replace(".hdf5", "")
                    n, d = hfp["train"].shape
                    stats.append(dict(dataset=name, n=n, dimensions=d))
            except Exception as e:
                print(f"Invalid dataset file {path} -- skipping")
                continue
            pbar.update(1)

    stats = pl.DataFrame(stats)
    stats.write_parquet(output_file)


def mahalanobis_distance_batch(V, Q):
    if V.ndim != 2:
        raise ValueError("Input matrix 'V' must be 2-dimensional (each row is a vector).")
    if Q.ndim != 2:
        raise ValueError("Input matrix 'Q' must be 2-dimensional.")
    if V.shape[1] != Q.shape[1]:
        raise ValueError(f"Dimension mismatch: V columns ({V.shape[1]}) must equal Q columns ({Q.shape[1]}).")
    if Q.shape[0] < 2:
        raise ValueError("Input matrix 'Q' must have at least 2 samples (rows).")

    mu = np.mean(Q, axis=0, dtype=np.float64)
    diff = V.astype(np.float64, copy=False) - mu

    if Q.shape[1] == 1:
        variance = np.var(Q[:, 0], ddof=1, dtype=np.float64)
        if np.isclose(variance, 0):
            is_zero_diff = np.all(np.isclose(diff, 0), axis=1)
            distances_sq = np.full(V.shape[0], np.inf)
            distances_sq[is_zero_diff] = 0.0
        else:
            inv_cov = 1.0 / variance
            distances_sq = (diff**2 * inv_cov).flatten()
    else:
        cov_matrix = covariance_matrix(Q, mu)
        try:
            inv_cov_matrix = np.linalg.pinv(cov_matrix, hermitian=True)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular and pseudo-inverse could not be computed.")

        temp = diff @ inv_cov_matrix
        distances_sq = np.sum(temp * diff, axis=1)

    negative_close_to_zero = (distances_sq < 0) & np.isclose(distances_sq, 0)
    distances_sq[negative_close_to_zero] = 0.0

    if np.any(distances_sq < 0):
        raise ValueError("Squared Mahalanobis distance is negative for some inputs")

    return np.sqrt(distances_sq)


def diagonal_mahalanobis_distance_batch(V, mu, variance, batch_size=8192):
    inv_variance = np.zeros_like(variance, dtype=np.float32)
    np.divide(1.0, variance, out=inv_variance, where=~np.isclose(variance, 0))

    distances_sq = np.empty(V.shape[0], dtype=np.float64)
    for start in range(0, V.shape[0], batch_size):
        stop = min(start + batch_size, V.shape[0])
        diff = V[start:stop] - mu
        distances_sq[start:stop] = np.sum(diff * diff * inv_variance, axis=1, dtype=np.float64)

    return np.sqrt(distances_sq)


def estimate_diagonal_mahalanobis_stats(dataset, ranges, is_binary):
    total = None
    total_sq = None
    count = 0

    for start, stop in ranges:
        vectors = prepare_vectors(dataset[start:stop], is_binary)

        if total is None:
            total = np.zeros(vectors.shape[1], dtype=np.float64)
            total_sq = np.zeros(vectors.shape[1], dtype=np.float64)

        total += np.sum(vectors, axis=0, dtype=np.float64)
        total_sq += np.sum(vectors * vectors, axis=0, dtype=np.float64)
        count += vectors.shape[0]

    mu = total / count
    variance = (total_sq - count * mu * mu) / (count - 1)
    variance = np.maximum(variance, 0)

    return mu.astype(np.float32), variance.astype(np.float32)


def covariance_matrix(Q, mu, batch_size=2048):
    cov = np.zeros((Q.shape[1], Q.shape[1]), dtype=np.float64)
    for start in range(0, Q.shape[0], batch_size):
        stop = min(start + batch_size, Q.shape[0])
        centered = Q[start:stop].astype(np.float64, copy=True)
        centered -= mu
        cov += centered.T @ centered

    cov /= Q.shape[0] - 1
    return cov


def sample_row_ranges(row_count, sample_size, gen, block_size=1024):
    sample_size = min(row_count, sample_size)
    block_size = min(block_size, sample_size)
    block_count = (sample_size + block_size - 1) // block_size
    edges = np.linspace(0, row_count, block_count + 1, dtype=np.int64)

    ranges = []
    remaining = sample_size
    for block_index in range(block_count):
        take = min(block_size, remaining)
        low = edges[block_index]
        high = max(low, edges[block_index + 1] - take)
        start = low if high == low else gen.integers(low, high + 1)
        ranges.append((int(start), int(start + take)))
        remaining -= take

    return ranges


def read_hdf5_ranges(dataset, ranges):
    row_count = sum(stop - start for start, stop in ranges)
    rows = np.empty((row_count, *dataset.shape[1:]), dtype=dataset.dtype)

    offset = 0
    for start, stop in ranges:
        next_offset = offset + stop - start
        rows[offset:next_offset] = dataset[start:stop]
        offset = next_offset

    return rows


def read_hdf5_rows(dataset, indices, batch_size=8192):
    indices = np.asarray(indices)
    rows = np.empty((len(indices), *dataset.shape[1:]), dtype=dataset.dtype)

    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        rows[start:stop] = dataset[indices[start:stop]]

    return rows


def prepare_vectors(vectors, is_binary):
    if is_binary:
        return np.unpackbits(vectors, axis=1).astype(np.float32, copy=False)

    return vectors.astype(np.float32, copy=False)


def export_pca_and_mahalanobis(
    data_dir, output_file, sample_size=2000, mahalanobis_sample_size=100_000, mahalanobis_mode="full"
):
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

            try:
                with h5py.File(path) as hfp:
                    name = path.name.replace(".hdf5", "")
                    train_dataset = hfp["train"]
                    test = hfp["test"][:]
                    is_binary = name.endswith("-binary")

                    mahalanobis_sample_ranges = sample_row_ranges(
                        train_dataset.shape[0], mahalanobis_sample_size, gen
                    )
                    train_sample_ranges = sample_row_ranges(train_dataset.shape[0], sample_size, gen)

                    if mahalanobis_mode == "full":
                        mahalanobis_sample_train = read_hdf5_ranges(train_dataset, mahalanobis_sample_ranges)
                    else:
                        mahalanobis_mu, mahalanobis_variance = estimate_diagonal_mahalanobis_stats(
                            train_dataset, mahalanobis_sample_ranges, is_binary
                        )
                    train = read_hdf5_ranges(train_dataset, train_sample_ranges)
            except Exception:
                print(f"Invalid dataset file {path} -- skipping")
                continue

            train = prepare_vectors(train, is_binary)
            test = prepare_vectors(test, is_binary)
            combined = np.vstack([train, test])

            if mahalanobis_mode == "full":
                mahalanobis_sample_train = prepare_vectors(mahalanobis_sample_train, is_binary)
                mahalanobis_combined = mahalanobis_distance_batch(combined, mahalanobis_sample_train)
            else:
                mahalanobis_combined = diagonal_mahalanobis_distance_batch(
                    combined, mahalanobis_mu, mahalanobis_variance
                )

            pca = PCA(n_components=2, random_state=1, svd_solver="randomized")
            scaler = StandardScaler(copy=False)

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
            del combined, combined_scaled, combined_pca, train, test, mahalanobis_combined
            if mahalanobis_mode == "full":
                del mahalanobis_sample_train
            else:
                del mahalanobis_mu, mahalanobis_variance

            pbar.update(1)

    pcas = pl.concat(pcas)
    pcas.write_parquet(output_file)


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--data", help="the path to the directory containing datasets", default="data")
    aparser.add_argument("--output", help="the path to the output directory", default="results")
    aparser.add_argument("--parallelism", type=int, help="number of parallel processes to use", default=1)
    aparser.add_argument("--pca-sample-size", type=int, help="number of train rows to export for PCA", default=2000)
    aparser.add_argument(
        "--mahalanobis-sample-size",
        type=int,
        help="number of train rows used to estimate Mahalanobis distances",
        default=100_000,
    )
    aparser.add_argument(
        "--mahalanobis-mode",
        choices=("diagonal", "full"),
        default="full",
        help="full computes the exact covariance pseudo-inverse; diagonal is faster",
    )
    aparser.add_argument(
        "--skip-pca-mahalanobis",
        help="skip exporting data-pca-mahalanobis.parquet",
        action="store_true",
    )

    args = aparser.parse_args()
    if args.pca_sample_size < 1:
        aparser.error("--pca-sample-size must be at least 1")
    if args.mahalanobis_sample_size < 2:
        aparser.error("--mahalanobis-sample-size must be at least 2")

    output_dir = pathlib.Path(args.output)
    output_summary = output_dir / "summary.parquet"
    output_stats = output_dir / "stats.parquet"
    output_info = output_dir / "data-info.parquet"
    output_pca_mahalanobis = output_dir / "data-pca-mahalanobis.parquet"

    export_all_results(args.results, args.data, args.parallelism, output_summary, output_dir)
    export_query_stats(args.data, output_stats)
    export_data_info(args.data, output_info)
    if not args.skip_pca_mahalanobis:
        export_pca_and_mahalanobis(
            args.data,
            output_pca_mahalanobis,
            sample_size=args.pca_sample_size,
            mahalanobis_sample_size=args.mahalanobis_sample_size,
            mahalanobis_mode=args.mahalanobis_mode,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted by user (Ctrl+C).")
        sys.exit(130)
