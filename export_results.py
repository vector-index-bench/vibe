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
            yield dataset, summary, detail


def export_all_results(path, output_summary, output_dir):
    summaries = []

    for path in pathlib.Path(path).glob("**/*.hdf5"):
        try:
            for dataset, summary, detail in export_results(path):
                summaries.append(summary)
                detail.write_parquet(output_dir / f"{dataset}__detail.parquet")
        except:
            pass

    summary = pl.DataFrame(summaries)
    summary.write_parquet(output_summary)


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


def mahalanobis_distance_batch(V, Q):
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if V.ndim != 2:
        raise ValueError("Input matrix 'V' must be 2-dimensional (each row is a vector).")
    if Q.ndim != 2:
        raise ValueError("Input matrix 'Q' must be 2-dimensional.")
    if V.shape[1] != Q.shape[1]:
        raise ValueError(
                f"Dimension mismatch: V columns ({V.shape[1]}) must equal "
                f"Q columns ({Q.shape[1]})."
        )
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
            raise ValueError(
                    "Covariance matrix is singular and pseudo-inverse could not be computed."
            )

        temp = diff @ inv_cov_matrix
        distances_sq = np.sum(temp * diff, axis=1)

    negative_close_to_zero = (distances_sq < 0) & np.isclose(distances_sq, 0)
    distances_sq[negative_close_to_zero] = 0.0

    if np.any(distances_sq < 0):
            raise ValueError("Squared Mahalanobis distance is negative for some inputs")

    return np.sqrt(distances_sq)


def export_pca_and_mahalanobis(data_dir, output_file, sample_size=2000):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pcas = []
    for path in pathlib.Path(data_dir).glob("**/*.hdf5"):
        gen = np.random.default_rng(1234)
        with h5py.File(path) as hfp:
            name = path.name.replace(".hdf5", "")
            ic(name)
            train = hfp["train"][:]
            test = hfp["test"][:]

            mahalanobis_sample_train = train[np.sort(gen.choice(train.shape[0], 100_000, replace=False))]
            train_sample_indices = np.sort(gen.choice(train.shape[0], sample_size, replace=False))
            train = train[train_sample_indices]

            data_to_data = mahalanobis_distance_batch(train, mahalanobis_sample_train)
            query_to_data = mahalanobis_distance_batch(test, mahalanobis_sample_train)

            mahalanobis_combined = np.concatenate((data_to_data, query_to_data))

            pca = PCA(n_components=2, random_state=1)
            scaler = StandardScaler()

            combined = np.vstack([train, test])
            ic(combined.shape)
            combined_scaled = scaler.fit_transform(combined)
            combined_pca = pca.fit_transform(combined_scaled)
            ic(combined_pca.shape)
            df = pl.DataFrame(dict(
                dataset=name,
                part=np.concatenate((np.repeat("train", train.shape[0]), np.repeat("test", test.shape[0]))),
                x=combined_pca[:, 0],
                y=combined_pca[:, 1],
                mahalanobis_distance_to_data=mahalanobis_combined
            ))
            pcas.append(df)
            

    pcas = pl.concat(pcas)
    pcas.write_parquet(output_file)


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--data", help="the path to the directory containing datasets", default="data")
    aparser.add_argument("--output", help="the path to the output directory", default="results")

    args = aparser.parse_args()

    output_dir = pathlib.Path(args.output)
    output_summary = output_dir / "summary.parquet"
    output_stats = output_dir / "stats.parquet"
    output_info = output_dir / "data-info.parquet"
    output_pca_mahalanobis = output_dir / "data-pca-mahalanobis.parquet"

    export_all_results(args.results, output_summary, output_dir)
    export_query_stats(args.data, output_stats)
    export_data_info(args.data, output_info)
    export_pca_and_mahalanobis(args.data, output_pca_mahalanobis)


if __name__ == "__main__":
    main()
