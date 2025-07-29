# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "scipy",
#     "seaborn",
#     "scikit-learn",
#     "networkx"
# ]
# ///

import pathlib
import os
import math
import argparse
import itertools
from scipy.stats import wilcoxon
import polars as pl
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from vibe.definitions import get_definitions
from vibe.main import filter_disabled_algorithms, filter_algorithms_by_device

# The list of in-distribution datasets
ID_DATASETS = [
    "agnews-mxbai-1024-euclidean",
    "arxiv-nomic-768-normalized",
    "gooaq-distilroberta-768-normalized",
    "imagenet-clip-512-normalized",
    "landmark-nomic-768-normalized",
    "yahoo-minilm-384-normalized",
]
ID_DATASETS_ADDITIONAL = [
    "ccnews-nomic-768-normalized",
    "celeba-resnet-2048-cosine",
    "codesearchnet-jina-768-cosine",
    "glove-200-cosine",
    "landmark-dino-768-cosine",
    "simplewiki-openai-3072-normalized",
]
# The list of out of distribution datasets
OOD_DATASETS = [
    "coco-nomic-768-normalized",
    "laion-clip-512-normalized",
    "llama-128-ip",
    "imagenet-align-640-normalized",
    "yandex-200-cosine",
    "yi-128-ip",
]

sns.set_palette("tab10")


def radar_chart(
    data,
    theta,
    radius,
    ticks,
    ax=None,
    smooth=False,
    show_percentiles=False,
    shorten_labels=False,
    supporting_ip=True,
    theta_offset=0,
    **kwargs,
):
    from scipy.interpolate import make_interp_spline
    import numpy as np

    # Enforce the order of the datasets
    data = data.set_index("dataset").loc[ticks].reset_index()

    categories = data[theta].to_list()
    values = data[radius].to_list()
    types = data["dataset-type"].to_list()

    num_vars = len(categories)
    theta = np.linspace(0, 2 * np.pi, num_vars + 1, endpoint=True)
    values.append(values[0])

    if ax is None:
        ax = plt.gca()

    ax.set_theta_offset(theta_offset)

    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]

    datatype_palette = {"in-distribution": "tab:blue", "out-of-distribution": "tab:orange"}

    if smooth:
        theta_smooth = np.linspace(0, 2 * np.pi, 1000)
        values_smooth = make_interp_spline(theta, values, bc_type="periodic", k=3)(theta_smooth)
        ax.plot(theta_smooth, values_smooth, color="gray")
        ax.fill_between(theta_smooth, values_smooth, color="gray", alpha=0.3)
    else:
        ax.plot(theta, values, color="gray")
        ax.fill_between(theta, values, color="gray", alpha=0.3)

    for t, val, data_type in zip(theta, values, types):
        if val > 0.0:
            color = datatype_palette.get(data_type, "red")
            ax.scatter([t], [val], c=color, zorder=10)

    for ytick in yticks:
        ax.add_patch(plt.Circle((0, 0), ytick, transform=ax.transData._b, color="gray", alpha=0.1))

    for data_type, t, dataset in zip(types, theta[:-1], categories):
        if supporting_ip or "-ip" not in dataset:
            color = datatype_palette.get(data_type, "red")
            ax.axvline(t, c=color, linewidth=1, alpha=0.6, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    if show_percentiles:
        for t in yticks[:-1]:
            ax.annotate(xy=(np.pi / 2 - theta_offset, t), text=f"{t * 100}%", fontsize=7, va="center")
    if shorten_labels:
        xlabels = [t[:2] for t in ticks]
    else:
        xlabels = [t.split("-")[0] for t in ticks]
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(xlabels)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(theta[:-1])
    ax.grid(False)


def fastest_at(data, recall=0.9, k=100):
    return (
        data.filter(pl.col("k") == k)
        .filter(pl.col("recall") >= recall)
        .with_columns(pl.col("qps").rank(descending=True).over("dataset", "algorithm").alias("__tmp__"))
        .filter(pl.col("__tmp__") == 1)
        .select(pl.exclude("__tmp__"))
    )


def radar_at_recall_plot(
    out_dir,
    data,
    query_stats,
    recall,
    algorithms,
    ncols=5,
    height=4.5,
    k=100,
):
    data = data.filter(pl.col("dataset").is_in(ID_DATASETS + OOD_DATASETS))
    datasets = data["dataset"].unique().to_list()
    expected_combinations = pl.DataFrame({"dataset": datasets}).join(
        pl.DataFrame({"algorithm": algorithms}), how="cross"
    )

    supports_ip = data.filter(pl.col("dataset").str.contains("-ip"))["algorithm"].unique().to_list()

    plot_data = (
        data.filter(pl.col("k") == k)
        .filter(pl.col("recall") >= recall)
        .with_columns(pl.col("qps").rank(descending=True).over("dataset", "algorithm").alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("dataset", "algorithm", "params", "recall", "qps")
        .with_columns((pl.col("qps") / pl.col("qps").max().over("dataset")).alias("qps_frac"))
        .sort("qps", descending=True)
        .join(expected_combinations, on=["algorithm", "dataset"], how="right")
        .with_columns(
            pl.when(pl.col("qps").is_not_null())
            .then(pl.col("qps").map_elements(lambda x: f"{x:.0f}", return_dtype=pl.String))
            .otherwise(pl.lit("x"))
            .alias("label"),
            pl.when(pl.col("dataset").is_in(ID_DATASETS))
            .then(pl.lit("in-distribution"))
            .when(pl.col("dataset").is_in(OOD_DATASETS))
            .then(pl.lit("out-of-distribution"))
            .otherwise(pl.lit("unknown-type"))
            .alias("dataset-type"),
        )
        .with_columns(pl.col("qps", "recall", "qps_frac").fill_null(0))
        .filter(pl.col("algorithm").is_in(algorithms))
        .select("algorithm", "dataset", "qps_frac", "dataset-type")
    )

    avg_rc = query_stats.group_by("dataset").agg(pl.col("rc100").mean())

    dataset_order = (
        plot_data.select("dataset", "dataset-type")
        .unique()
        .join(avg_rc, on=["dataset"])
        .with_columns(~pl.col("dataset").str.contains("-ip").alias("is-ip"))
        .sort("dataset-type", "is-ip", "rc100", "dataset")
    )["dataset"].to_list()

    algorithm_order = (
        plot_data.with_columns(pl.col("qps_frac").rank(descending=True).over("dataset").alias("rank"))
        .group_by("algorithm")
        .agg(pl.col("rank").mean())
        .sort("rank")
    )["algorithm"].to_list()

    width = ncols * 2.25
    fig, axs = plt.subplots(
        figsize=(width, height),
        ncols=ncols,
        nrows=math.ceil(len(algorithm_order) / ncols),
        subplot_kw=dict(projection="polar"),
    )
    axs = [ax for sub in axs for ax in sub]
    for ax in axs:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    theta_offset = -1 / 3 * math.pi
    for algo, ax in zip(algorithm_order, axs[1:]):
        facet_data = plot_data.filter(pl.col("algorithm") == algo)
        radar_chart(
            facet_data.to_pandas(),
            theta="dataset",
            radius="qps_frac",
            ticks=dataset_order,
            ax=ax,
            shorten_labels=True,
            supporting_ip=algo in supports_ip,
            theta_offset=theta_offset,
        )
        ax.set_title(algo)

    # setup the legend
    legend_data = (
        plot_data.select("dataset", "dataset-type")
        .unique()
        .sort("dataset-type", "dataset")
        .with_columns(qps_frac=pl.lit(0.0))
    )
    radar_chart(
        legend_data.to_pandas(),
        theta="dataset",
        radius="qps_frac",
        ticks=dataset_order,
        show_percentiles=True,
        ax=axs[0],
        theta_offset=theta_offset,
    )

    plt.tight_layout()
    plt.savefig(out_dir / f"radar-{recall}.png", dpi=300)
    plt.close()


def compute_pareto(data, by=["algorithm", "dataset", "k"]):
    data = data.with_columns(
        pl.col("qps").over(partition_by=by, order_by="qps").rank("ordinal", descending=True).alias("rank_qps"),
        pl.col("recall").over(partition_by=by, order_by="recall").rank("ordinal", descending=True).alias("rank_recall"),
    )

    dominated = (
        data.join(data, on=by, how="inner")
        .filter(
            ((pl.col("rank_qps") > pl.col("rank_qps_right")) & (pl.col("rank_recall") >= pl.col("rank_recall_right")))
            | ((pl.col("rank_qps") >= pl.col("rank_qps_right")) & (pl.col("rank_recall") > pl.col("rank_recall_right")))
        )
        .select(pl.exclude("^*_right$"))
    )

    pareto = data.join(dominated, on=by + ["params"], how="anti")
    return pareto


def compute_pareto_direct(data, by=["algorithm", "dataset", "k"]):
    data_ranked = data.with_row_index("row_nr").with_columns(
        pl.col("qps").rank("ordinal", descending=True).over(by).alias("rank_qps"),
        pl.col("recall").rank("ordinal", descending=True).over(by).alias("rank_recall"),
    )

    right_data = data_ranked.select(by + ["row_nr", "rank_qps", "rank_recall"])

    comparison_join = data_ranked.join(right_data, on=by, how="inner", suffix="_right")

    is_dominated_flag = comparison_join.filter(
        pl.col("row_nr") != pl.col("row_nr_right")  # Don't compare a point to itself
    ).with_columns(
        is_dominated_by_right=(
            ((pl.col("rank_qps") > pl.col("rank_qps_right")) & (pl.col("rank_recall") >= pl.col("rank_recall_right")))
            | ((pl.col("rank_qps") >= pl.col("rank_qps_right")) & (pl.col("rank_recall") > pl.col("rank_recall_right")))
        )
    )

    dominance_summary = is_dominated_flag.group_by("row_nr").agg(
        pl.col("is_dominated_by_right").any().alias("is_dominated")
    )

    pareto_data = (
        data_ranked.join(dominance_summary, on="row_nr", how="left")
        .filter(pl.col("is_dominated").fill_null(False).not_())
        .select(pl.exclude(["row_nr", "rank_qps", "rank_recall", "is_dominated"]))
    )

    return pareto_data


def adjust_text(texts, height):
    inv_data_transform = plt.gca().transData.inverted()
    data_transform = plt.gca().transData

    def get_x_display(text):
        return data_transform.transform(text.get_position())[0]

    def get_y_display(text):
        return data_transform.transform(text.get_position())[1]

    texts = sorted(texts, reverse=True, key=lambda text: text.get_position()[1])
    for prev, text in zip(texts, texts[1:]):
        if get_y_display(prev) - get_y_display(text) < height:
            newpos = inv_data_transform.transform((get_x_display(text), get_y_display(prev) - height))
            text.set_y(newpos[1])

        pass


def pareto_plot(
    out_dir,
    data,
    pca_mahalanobis,
    datasets,
    algorithms,
    k: int = 100,
    xlim=(0.5, 1.0),
    ylim=(2e2, 1.4e4),
    *,
    figsize=(10, 6),
    separate_legend: bool = True,
):
    threshold_recall = 0.9

    plot_data = (
        data.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
    )

    pareto_data = compute_pareto_direct(plot_data)

    qps_over_threshold = (
        pareto_data.filter(pl.col("recall") >= threshold_recall)
        .group_by("algorithm")
        .agg(pl.col("qps").max().alias("best_qps_over_70"))
        .sort("best_qps_over_70", descending=True)
    )
    legend_order = qps_over_threshold["algorithm"].to_list()
    legend_order.extend([a for a in algorithms if a not in legend_order])
    rank = {alg: i for i, alg in enumerate(legend_order)}

    tab10 = mpl.colormaps["tab10"].colors
    tab20 = mpl.colormaps["tab20"].colors
    palette = tab20 + tab10

    algorithm_colors = dict(zip(algorithms, palette))
    algorithm_dashes = dict(zip(algorithms, sns._base.unique_dashes(len(algorithms))))
    algorithm_markers = dict(zip(algorithms, sns._base.unique_markers(len(algorithms))))

    joint_legend = isinstance(algorithms[0], str)

    def plot_lines(pdata, background=False, ax=None):
        if ax is None:
            ax = plt.gca()
        opacity = 0.2 if background else 1.0
        kwargs = dict(data=pdata, x="recall", y="qps", alpha=opacity)
        if background:
            kwargs["color"] = "gray"
        else:
            kwargs.update(
                hue="algorithm",
                style="algorithm",
                palette=algorithm_colors,
            )
        sns.lineplot(
            units="algorithm",
            lw=1.5,
            estimator=None,
            markers=algorithm_markers,
            legend=True,
            dashes=algorithm_dashes,
            ax=ax,
            **kwargs,
        )
        ax.grid(which="major", linewidth=0.5, color="lightgray", alpha=0.5)
        ax.grid(which="minor", axis="y", linewidth=0.5, color="lightgray", alpha=0.5)

    fig, axs = plt.subplots(1, len(datasets), figsize=figsize, sharex=True, sharey=True)
    if len(datasets) == 1:
        axs = [axs]

    if joint_legend:
        algorithms = [legend_order] * len(datasets)

    for dataset, ax, algos in zip(datasets, axs, algorithms):
        facet = pareto_data.filter(pl.col("dataset") == dataset)

        if facet.is_empty():
            raise ValueError(f"no results data for dataset {dataset}")

        plot_lines(facet, ax=ax)
        ax.set_xlim(xlim)
        ax.semilogy()

        title = "-".join(dataset.split("-")[:-2])
        if dataset.endswith("-binary"):
            title = "-".join(dataset.split("-")[:-3]) + "-binary"
        ax.set_title(title)

        ticks = np.arange(xlim[0], 1.01, 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.1f}" for x in ticks])

        if pca_mahalanobis is not None:
            pca_m_data = pca_mahalanobis.filter(pl.col("dataset") == dataset)
            inset_w, inset_h, gap = 0.3, 0.3, 0.05
            bounds = [
                [0.05, 0.05, inset_w, inset_h],
                [0.05 + inset_w + gap, 0.05, inset_w, inset_h],
            ]
            axins = [ax.inset_axes(bb) for bb in bounds]
            for a in axins:
                a.spines[:].set_visible(True)
                a.spines[:].set_color("black")
                a.set_xticks([])
                a.set_yticks([])
            for part in pca_m_data["part"].unique():
                pdata = pca_m_data.filter(pl.col("part") == part)
                axins[0].scatter(pdata["x"], pdata["y"], s=0.1)
            sns.kdeplot(
                pca_m_data,
                x="mahalanobis_distance_to_data",
                hue="part",
                fill=True,
                legend=False,
                ax=axins[1],
            )

    if joint_legend:
        handles, labels = axs[0].get_legend_handles_labels()

        # sort the legend entries by our rank
        sorted_pairs = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], float("inf")))
        handles, labels = zip(*sorted_pairs)

        if separate_legend:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            fig.tight_layout(pad=0.1, w_pad=1.08, h_pad=1.08)
            fig.savefig(out_dir / f"{'__'.join(datasets)}-qps-recall.png", dpi=300)
            plt.close(fig)

            plt.figure(figsize=(1.7, 2.5))
            plt.legend(handles, labels, frameon=False)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"{'__'.join(datasets)}-qps-recall-legend.png", dpi=300)
            plt.close()

        else:
            for ax in axs:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            legend_pad = 0.20
            fig.tight_layout(rect=[0, 0, 1 - legend_pad, 1], pad=0.1, w_pad=1.08, h_pad=1.08)
            fig.legend(handles, labels, loc="center right")
            fig.savefig(out_dir / f"{'__'.join(datasets)}-qps-recall.png", dpi=300)
            plt.close(fig)

    else:
        fig.tight_layout(pad=0.1, w_pad=1.08, h_pad=1.08)
        fig.savefig(out_dir / f"{'__'.join(datasets)}-qps-recall.png", dpi=300)
        plt.close(fig)


def split_difficulties_plot(
    out_dir,
    summary,
    detail,
    query_stats,
    recall,
    datasets,
    algorithms=["symphonyqg", "lorann", "glass", "ngt-qg"],
    k=100,
    easy_ptile=0.1,
    difficult_ptile=0.1,
):
    nqueries = query_stats.group_by("dataset").len("nqueries")

    actual_performance_data = (
        # Pick the relevant data
        summary.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
        # Compute the throughput and recall of each algorithm configuration
        # Select the fastest configuration with recall above the threshold,
        # for each algorithm and difficulty
        .filter(pl.col("recall") > recall)
        .with_columns(pl.col("qps").rank(descending=True).over(["dataset", "algorithm"]).alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("dataset", "algorithm", "params", "qps", "recall")
    )

    selected_queries = (
        query_stats.select("dataset", "query_index", "rc100")
        .with_columns(pl.col("rc100").rank("ordinal", descending=True).over("dataset").alias("rank_rc100"))
        .join(nqueries, on="dataset")
        .with_columns(
            pl.when(pl.col("rank_rc100") < easy_ptile * pl.col("nqueries"))
            .then(pl.lit("easy"))
            .when(pl.col("rank_rc100") >= (1 - difficult_ptile) * pl.col("nqueries"))
            .then(pl.lit("difficult"))
            .alias("difficulty")
        )
        .drop_nulls("difficulty")
    )

    plot_data = (
        # Pick the relevant data
        detail.filter(pl.col("dataset").is_in(datasets))
        .filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
        .join(actual_performance_data, on=["dataset", "algorithm", "params"], how="semi")
        # Select only the easy and difficult queries, for all algorithm's parameterizations
        .join(selected_queries, on=["dataset", "query_index"])
        .select(pl.exclude("k", "query_index", "rank_rc100", "nqueries"))
        # Compute the throughput and recall of each algorithm configuration
        # on these "virtual" workloads
        .group_by("dataset", "algorithm", "params", "difficulty")
        .agg(pl.col("recall").mean(), (1 / pl.col("time").mean()).alias("qps"))
        .select("dataset", "algorithm", "difficulty", "qps", "recall")
    )

    plot_data = plot_data.pivot(index=["dataset", "algorithm"], on="difficulty", values="recall").sort("difficult")

    height = 0.26 * plot_data.select("algorithm").n_unique()
    _, axs = plt.subplots(1, len(datasets), figsize=(8, height))
    if len(datasets) == 1:
        axs = [axs]

    def do_plot(pdata, ax):
        ax.hlines(range(pdata.shape[0]), xmin=pdata["easy"], xmax=pdata["difficult"], color="grey", alpha=0.4)
        ax.scatter(pdata["difficult"], pdata["algorithm"], zorder=2, clip_on=False, color="tab:blue", label="difficult")
        ax.scatter(pdata["easy"], pdata["algorithm"], zorder=2, clip_on=False, color="#c9a227", label="easy")

        ax.axvline(recall, color="lightgray", lw=1, zorder=-1)

        for algo in pdata["algorithm"].unique().to_list():
            xpos = pdata.filter(pl.col("algorithm") == algo)[["difficult", "easy"]].transpose().min()["column_0"][0]
            performance_easy, performance_difficult = tuple(
                pdata.filter(pl.col("algorithm") == algo)[["easy", "difficult"]].unpivot()["value"]
            )
            if xpos is not None:
                ax.annotate(
                    xy=(xpos, algo),
                    xytext=(-35, 0),
                    textcoords="offset points",
                    text=algo,
                    ha="right",
                    va="center",
                    size=9,
                )
            if performance_difficult is not None:
                ax.annotate(
                    xy=(performance_difficult, algo),
                    text=f"{performance_difficult:.2f}",
                    ha="right" if performance_easy > performance_difficult else "left",
                    va="center",
                    size=9,
                    color="steelblue",
                    xytext=(-10, 0) if performance_easy > performance_difficult else (10, 0),
                    textcoords="offset points",
                )
            if performance_easy is not None:
                ax.annotate(
                    xy=(performance_easy, algo),
                    text=f"{performance_easy:.2f}",
                    ha="left" if performance_easy > performance_difficult else "right",
                    va="center",
                    size=9,
                    color="#c9a227",
                    xytext=(10, 0) if performance_easy > performance_difficult else (-10, 0),
                    textcoords="offset points",
                )
        ax.axis("off")

    for dataset, ax in zip(datasets, axs):
        do_plot(plot_data.filter(pl.col("dataset") == dataset), ax)
        ax.set_title(dataset, pad=15)

    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    plt.savefig(out_dir / f"{'__'.join(datasets)}-split-performance.png", dpi=300)
    plt.close()


def plot_difficulty_ridgeline(out_dir, query_stats, x="rc100", log=True):
    # adapted from https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    from sklearn.neighbors import KernelDensity
    import numpy as np

    query_stats = (
        query_stats.filter(pl.col("dataset").is_in(ID_DATASETS + OOD_DATASETS))
        .filter(~pl.col("dataset").str.contains("-ip"))
        .filter(pl.col(x) >= 1)
        .with_columns(mean_x=pl.col(x).mean().over("dataset"))
        .with_columns(
            pl.when(pl.col("dataset").is_in(ID_DATASETS))
            .then(pl.lit("in-distribution"))
            .when(pl.col("dataset").is_in(OOD_DATASETS))
            .then(pl.lit("out-of-distribution"))
            .otherwise(pl.lit("unknown-type"))
            .alias("dataset-type"),
        )
        .sort("mean_x", descending=True)
    )

    if log:
        query_stats = query_stats.with_columns(pl.col(x).log())

    datasets = query_stats.group_by("dataset").agg(pl.col(x).median()).sort(x, descending=True)["dataset"].to_list()

    plt.figure(figsize=(8, 3))
    ax = plt.gca()

    maxx = 3.5
    minx = 0
    for i, dataset in enumerate(datasets):
        pdata = query_stats.filter(pl.col("dataset") == dataset)
        xvals = pdata[x].to_numpy()
        x_d = np.linspace(minx, maxx, 1000)

        kde = KernelDensity(bandwidth=0.05, kernel="gaussian")
        kde.fit(xvals[:, None])
        logprob = kde.score_samples(x_d[:, None])

        offset = (len(datasets) - i - 1) * 1.5
        color = "tab:blue" if dataset in ID_DATASETS else "tab:orange"
        ax.plot(x_d, offset + np.exp(logprob), color="#f0f0f0", lw=1, zorder=2 * i + 1)
        ax.fill_between(x_d, offset + np.exp(logprob), offset, alpha=1, zorder=2 * i, color=color)
        label = "-".join(dataset.split("-")[:-2])
        ax.annotate(label, (3.5, offset), ha="right", va="bottom", color=color)

    if log:
        ax.set_xlabel(f"log({x})")
    else:
        ax.set_xlabel(f"{x}")

    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_dir / f"distribution-{x}.png", dpi=300)
    plt.close()


def performance_gap_plot(out_dir, id_dataset, ood_dataset, summary, pca_mahalanobis_data, k=100, recall=0.9):
    """Plot the performance difference between in-distribution and out of distribution queries"""
    from matplotlib.gridspec import GridSpec

    pdata = summary.filter(pl.col("dataset").is_in([id_dataset, ood_dataset])).filter(pl.col("k") == k)
    if pdata.is_empty():
        raise ValueError("no results data found for performance gap plot")

    maindata = (
        fastest_at(pdata, recall)
        .with_columns(
            pl.when(pl.col("dataset") == id_dataset)
            .then(pl.lit("in-distribution"))
            .otherwise(pl.lit("out-of-distribution"))
            .alias("type")
        )
        .pivot(on="type", values="qps", index="algorithm")
        .drop_nulls()
        .sort("out-of-distribution", descending=False)
    )

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(2, 3)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_pca_id = fig.add_subplot(gs[0, 1])
    ax_pca_ood = fig.add_subplot(gs[1, 1])
    ax_mahalanobis_id = fig.add_subplot(gs[0, 2])
    ax_mahalanobis_ood = fig.add_subplot(gs[1, 2])
    ax_main.hlines(
        range(maindata.shape[0]),
        xmin=maindata["out-of-distribution"],
        xmax=maindata["in-distribution"],
        color="gray",
        zorder=-1,
    )
    ax_main.scatter(maindata["in-distribution"], maindata["algorithm"], color="tab:green")
    ax_main.scatter(maindata["out-of-distribution"], maindata["algorithm"], color="tab:purple")

    for algo in maindata["algorithm"].unique().to_list():
        xpos = (
            maindata.filter(pl.col("algorithm") == algo)[["in-distribution", "out-of-distribution"]]
            .transpose()
            .min()["column_0"][0]
        )
        performance_id, performance_ood = tuple(
            maindata.filter(pl.col("algorithm") == algo)[["in-distribution", "out-of-distribution"]].unpivot()["value"]
        )
        if xpos is not None:
            ax_main.annotate(
                xy=(xpos, algo), xytext=(-35, 0), textcoords="offset points", text=algo, ha="right", va="center", size=9
            )
        if performance_ood is not None:
            ax_main.annotate(
                xy=(performance_ood, algo),
                text=f"{performance_ood:.0f}",
                ha="right" if performance_id > performance_ood else "left",
                va="center",
                size=9,
                color="tab:purple",
                xytext=(-8, 0) if performance_id > performance_ood else (10, 0),
                textcoords="offset points",
            )
        if performance_id is not None:
            ax_main.annotate(
                xy=(performance_id, algo),
                text=f"{performance_id:.0f}",
                ha="left" if performance_id > performance_ood else "right",
                va="center",
                size=9,
                color="tab:green",
                xytext=(8, 0) if performance_id > performance_ood else (-10, 0),
                textcoords="offset points",
            )
    ax_main.axis("off")

    for dataname, ax_pca, ax_mahalanobis, color in zip(
        [id_dataset, ood_dataset],
        [ax_pca_id, ax_pca_ood],
        [ax_mahalanobis_id, ax_mahalanobis_ood],
        ["tab:green", "tab:purple"],
    ):
        pdata = pca_mahalanobis_data.filter(pl.col("dataset") == dataname, pl.col("part") == "train")
        ax_pca.scatter(pdata["x"], pdata["y"], s=0.1, c="tab:blue")
        sns.kdeplot(
            pdata, x="mahalanobis_distance_to_data", color="tab:blue", fill=True, legend=False, ax=ax_mahalanobis
        )

        pdata = pca_mahalanobis_data.filter(pl.col("dataset") == dataname, pl.col("part") == "test")
        ax_pca.scatter(pdata["x"], pdata["y"], s=0.1, c=color)
        sns.kdeplot(pdata, x="mahalanobis_distance_to_data", color=color, fill=True, legend=False, ax=ax_mahalanobis)
        ax_pca.axis("off")
        ax_mahalanobis.set_yticks([])
        ax_mahalanobis.set_xticks([])
        ax_mahalanobis.set_xlabel("")
        ax_mahalanobis.set_ylabel("")
        ax_mahalanobis.spines[:].set_visible(False)
        ax_mahalanobis.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.savefig(out_dir / f"performance-gap-{ood_dataset}.png", dpi=300)
    plt.close()


def paper(out_dir, summary, detail, query_stats, pca_mahalanobis):
    plot_difficulty_ridgeline(out_dir, query_stats)

    radar_at_recall_plot(
        out_dir,
        summary,
        query_stats,
        algorithms=[
            "lorann",
            "symphonyqg",
            "glass",
            "ngt-qg",
            "ngt-onng",
            "scann",
            "hnswlib",
            "ivfpqfs(faiss)",
            "vamana-lvq(svs)",
        ],
        recall=0.95,
    )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=None,
        datasets=["agnews-mxbai-1024-euclidean", "landmark-nomic-768-normalized"],
        algorithms=[
            "lorann",
            "symphonyqg",
            "glass",
            "ngt-qg",
            "scann",
            "ngt-onng",
            "hnswlib",
            "ivfpqfs(faiss)",
            "pynndescent",
        ],
        xlim=(0.7, 1.0),
        ylim=(2e2, 1.1e4),
        figsize=(8, 3),
        separate_legend=True,
    )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=None,
        datasets=["imagenet-clip-512-normalized", "landmark-nomic-768-normalized"],
        algorithms=[
            "lorann",
            "symphonyqg",
            "glass",
            "ngt-qg",
            "scann",
            "ngt-onng",
            "hnswlib",
            "ivfpqfs(faiss)",
            "pynndescent",
        ],
        xlim=(0.7, 1.0),
        ylim=(5e2, 1.1e4),
        figsize=(8, 3),
        separate_legend=True,
    )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=pca_mahalanobis,
        datasets=["yi-128-ip", "llama-128-ip"],
        algorithms=[
            "roargraph",
            "mlann-rf",
            "lorann",
            "ivf(faiss)",
            "scann",
            "ivfpqfs(faiss)",
            "hnswlib",
            "glass",
            "ngt-onng",
        ],
        xlim=(0, 1),
        figsize=(8, 3),
        separate_legend=True,
    )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=pca_mahalanobis,
        datasets=["yandex-200-cosine", "laion-clip-512-normalized"],
        algorithms=["roargraph", "lorann", "symphonyqg", "scann", "ngt-qg", "hnswlib", "glass"],
        xlim=(0.5, 1),
        figsize=(8, 3),
        separate_legend=True,
    )

    pareto_plot(
        out_dir,
        summary,
        pca_mahalanobis=None,
        datasets=["agnews-mxbai-1024-hamming-binary", "agnews-mxbai-1024-euclidean"],
        algorithms=[
            ["ngt-onng", "ivf(faiss)", "pynndescent", "hnsw(faiss)"],
            ["cuvs-cagra", "cuvs-ivfpq", "faiss-gpu-ivf", "cuvs-ivf", "ggnn"],
        ],
        xlim=(0.7, 1),
        ylim=(3e2, 3e5),
        figsize=(8, 3),
        separate_legend=True,
    )

    split_difficulties_plot(
        out_dir,
        summary,
        detail,
        query_stats,
        0.90,
        datasets=["arxiv-nomic-768-normalized", "landmark-nomic-768-normalized"],
    )


def latency_difference_table(
    data,
    detail,
    algorithms,
    recall,
    dataset,
    k=100,
):
    """\
    Checks whether the differences in latencies are statistically significant or not,
    by performing the Wilcoxon paired test.
    """
    data = data.filter(pl.col("dataset") == dataset).filter(pl.col("algorithm").is_in(algorithms))

    configs = (
        fastest_at(data, recall, k).select("dataset", "algorithm", "params").sort("dataset", "algorithm", "params")
    )

    stats = detail.filter(pl.col("k") == k, pl.col("dataset") == dataset).join(
        configs, on=["dataset", "algorithm", "params"]
    )

    def get_times(stats, algorithm):
        times = (
            stats.filter(pl.col("algorithm") == algorithm)
            .group_by("query_index")
            .agg(pl.col("time").mean())
            .sort("query_index")
        )["time"]
        return times.to_numpy()

    p_values = []
    for a, b in itertools.combinations(algorithms, 2):
        atimes = get_times(stats, a)
        btimes = get_times(stats, b)
        if len(atimes) == 0 or len(btimes) == 0:
            continue
        # Do the pairwise test
        test = wilcoxon(atimes - btimes)
        p_values.append(
            dict(algorithm_a=a, algorithm_b=b, latency_a=atimes.mean(), latency_b=btimes.mean(), p_value=test.pvalue)
        )
    if len(p_values) == 0:
        return pl.DataFrame(schema=["algorithm_a", "algorithm_b", "latency_a", "latency_b", "p_value", "dataset"])
    return pl.DataFrame(p_values).sort("p_value").with_columns(pl.lit(dataset).alias("dataset"))


def holm_bonferroni(table, p_value_col):
    """Correct the p-values of the given table, where each row is a statistical test."""
    table = table.sort(p_value_col)
    sorted_p_values = table[p_value_col].to_numpy()
    corrected_p_values = np.maximum.accumulate(sorted_p_values * np.arange(len(sorted_p_values), 0, -1))
    corrected_p_values = np.minimum(corrected_p_values, 1)
    table = table.with_columns(pl.Series(name=p_value_col, values=corrected_p_values))
    return table


def latency_difference_plot(summary, detail, recall, datasets, algorithms, output, k=100, significance_level=0.05):
    try:
        import networkx
    except ImportError:
        raise ImportError("latency_difference_plot requires networkx")
    tests = []
    for dataset in datasets:
        df = latency_difference_table(summary, detail, algorithms, recall, dataset, k=k)
        tests.append(df)
    tests = holm_bonferroni(pl.concat(tests), "p_value")
    print(
        tests.filter(pl.col("p_value") < 0.01).shape[0], "tests out of", tests.shape[0], "are statistically significant"
    )

    graphs = dict()
    for dataset in datasets:
        G = networkx.Graph()
        for algo in algorithms:
            G.add_node(algo)
        graphs[dataset] = G
    for test in tests.rows(named=True):
        if (
            test["p_value"] >= significance_level
            and test["algorithm_a"] in algorithms
            and test["algorithm_b"] in algorithms
        ):
            graphs[test["dataset"]].add_edge(test["algorithm_a"], test["algorithm_b"])

    for dataset in datasets:
        G = graphs[dataset]
        groups = [c for c in networkx.find_cliques(G) if len(c) > 1]
        plt.figure(figsize=(8, 2))
        pdata = tests.filter(pl.col("dataset") == dataset)
        pdata = (
            pl.concat(
                [
                    pdata.select(algorithm="algorithm_a", latency="latency_a"),
                    pdata.select(algorithm="algorithm_b", latency="latency_b"),
                ]
            )
            .unique()
            .sort("latency")
        )

        algos = pdata["algorithm"].to_numpy()
        times = pdata["latency"].to_numpy()
        minx, maxx = times[0], times[-1]
        span = maxx - minx
        for x in [minx, maxx]:
            label = f"{x * 1000:.3} ms"
            plt.annotate(label, xy=(x, 0), xytext=(x, 0.15), ha="center")
            plt.plot((x, x), (0, 0.05), c="black", lw=0.5)
        plt.plot((minx, maxx), (0, 0), c="black")
        minx = minx - 0.1 * span
        maxx = maxx + 0.1 * span

        offset = 0.2
        baseline = -0.3
        for i, a in enumerate(algos):
            if i < len(algos) // 2:
                y = baseline - i * offset
                plt.annotate(a, xy=(minx - 0.02 * span, y), ha="right")
                plt.plot((minx, times[i], times[i]), (y, y, 0), lw=0.5, c="black")
            else:
                y = baseline - (len(algos) - i - 1) * offset
                plt.annotate(a, xy=(maxx + 0.02 * span, y), ha="left")
                plt.plot((maxx, times[i], times[i]), (y, y, 0), lw=0.5, c="black")

        offset = 0.1
        baseline = -0.1
        for i, group in enumerate(groups):
            gstart = pdata.filter(pl.col("algorithm").is_in(group))["latency"].min()
            gend = pdata.filter(pl.col("algorithm").is_in(group))["latency"].max()
            y = baseline - i * offset
            plt.plot((gstart - 0.005 * span, gend + 0.005 * span), (y, y), lw=3, c="black")

        plt.title(dataset)
        plt.gca().set_axis_off()
        plt.tight_layout()
        plt.savefig(pathlib.Path(output) / f"latency-critdiff-{dataset}-{recall}.png")


if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument("--results", help="the path to the directory containing results", default="results")
    aparser.add_argument("--output", help="the path to the output directory", default="plots")
    aparser.add_argument(
        "--plot-type",
        help="type of plot (pareto, radar, difficulty, performance-gap, split-difficulties, critdiff)",
        default="pareto",
    )
    aparser.add_argument("--dataset", help="dataset", default="agnews-mxbai-1024-euclidean")
    aparser.add_argument("--selected", help="plot results for only selected algorithms", action="store_true")
    aparser.add_argument("--gpu", help="plot results for GPU algorithms", action="store_true")
    aparser.add_argument("--pca", help="add PCA plot (only applicable for pareto plot)", action="store_true")
    aparser.add_argument("--recall", help="recall level for plot", default=0.95)
    aparser.add_argument("--count", help="number of nearest neighbors (k) to use", default=100)

    args = aparser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data_dir = pathlib.Path(args.results)
    out_dir = pathlib.Path(args.output)

    normalize_names = pl.col("dataset")

    summary = pl.read_parquet(data_dir / "summary.parquet").with_columns(normalize_names)
    detail = pl.concat([pl.read_parquet(path) for path in data_dir.glob("*__detail.parquet")]).with_columns(
        normalize_names
    )
    query_stats = pl.read_parquet(data_dir / "stats.parquet").with_columns(normalize_names)
    pca_mahalanobis = pl.read_parquet(data_dir / "data-pca-mahalanobis.parquet").with_columns(normalize_names)

    datasets = args.dataset.split(",")
    count = int(args.count)
    recall = float(args.recall)

    if args.selected:
        algorithms = [
            "lorann",
            "symphonyqg",
            "glass",
            "ngt-qg",
            "ngt-onng",
            "scann",
            "hnswlib",
            "ivfpqfs(faiss)",
            "vamana-lvq(svs)",
        ]
    else:
        if datasets[0].endswith("-binary"):
            point_type = "binary"
            distance_metric = "hamming"
        elif datasets[0].endswith("-uint8"):
            point_type = "uint8"
            distance_metric = "euclidean"
        else:
            point_type = "float"
            distance_metric = "normalized"

        definitions = get_definitions(
            dimension=None,
            point_type=point_type,
            distance_metric=distance_metric,
            count=count,
            base_dir="vibe/algorithms",
        )

        definitions = filter_disabled_algorithms(definitions)
        definitions = filter_algorithms_by_device(definitions, args.gpu)

        algorithms = list(sorted(set(definition.algorithm for definition in definitions)))

    if args.plot_type == "pareto":
        if args.gpu:
            ylim = (2e3, 3e5)
        else:
            ylim = (2e2, 1.1e4)

        pareto_plot(
            out_dir,
            summary,
            pca_mahalanobis=pca_mahalanobis if args.pca else None,
            datasets=datasets,
            algorithms=algorithms,
            k=count,
            xlim=(0.7, 1.0),
            ylim=ylim,
            separate_legend=False,
        )
    elif args.plot_type == "radar":
        radar_at_recall_plot(
            out_dir,
            summary,
            query_stats,
            algorithms=algorithms,
            height=len(algorithms) / 2,
            recall=recall,
            k=count,
        )
    elif args.plot_type == "difficulty":
        plot_difficulty_ridgeline(out_dir, query_stats)
    elif args.plot_type == "performance-gap":
        if len(datasets) != 2:
            raise ValueError("plot type performance-gap requires two datasets")

        performance_gap_plot(
            out_dir,
            datasets[0],
            datasets[1],
            summary,
            pca_mahalanobis,
            k=count,
            recall=recall,
        )
    elif args.plot_type == "split-difficulties":
        split_difficulties_plot(
            out_dir,
            summary,
            detail,
            query_stats,
            recall,
            datasets=datasets,
            k=count,
        )
    elif args.plot_type == "critdiff":
        latency_difference_plot(
            summary, detail, float(args.recall), ID_DATASETS + OOD_DATASETS, output=args.output, algorithms=algorithms, k=count
        )
    elif args.plot_type == "paper":
        paper(out_dir, summary, detail, query_stats, pca_mahalanobis)
    else:
        raise ValueError(f"invalid plot type: {args.plot_type}")
