# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "adjusttext",
#     "icecream",
#     "polars",
#     "pyarrow",
#     "scipy",
#     "seaborn",
# ]
# ///

import pathlib
import sys
import math

import polars as pl
import seaborn as sns
from icecream import ic
from matplotlib import pyplot as plt

SCRIPT_PATH = pathlib.Path(sys.argv[0])
OUT_DIR = SCRIPT_PATH.parent
# The list of in-distribution datasets
ID_DATASETS = [
    "ccnews-distilroberta-768-cosine",
    "celeba-mobilenet-1280-cosine",
    "coco-nomic-768-normalized",
    "gooaq-nomic-768-normalized",
    "imagenet-clip-512-cosine",
    "landmark-dino-768-cosine",
    "wiki-openai-1536-cosine",
]
# The list of out of distribution datasets
OOD_DATASETS = ["llama-f-128-ip", "llama-g-128-ip", "laion-512-normalized", "yandex-200-ip"]

sns.set_palette("tab10")

algorithm_family = pl.DataFrame(
    {
        "annoy": "Tree",
        "faiss-ivf": "Clustering",
        "faiss-ivfpqfs": "Clustering",
        "flatnav": "Graph",
        "glass": "Graph",
        "hnsw(faiss)": "Graph",
        "hnswlib": "Graph",
        "hnswq(faiss)": "Graph",
        "lorann": "Clustering",
        "mlann-pca": "Other",
        "mlann-rf": "Other",
        "mrpt": "Tree",
        "ngt-onng": "Graph",
        "ngt-qg": "Graph",
        "nsg(faiss)": "Graph",
        "puffinn": "Hashing",
        "pynndescent": "Graph",
        "roargraph": "Graph",
        "scann": "Clustering",
        "symphonyqg": "Graph",
        "vamana(diskann)": "Graph",
        "vamana(parlayann)": "Graph",
        "vamana(svs)": "Graph",
        "vamana-leanvec(svs)": "Graph",
        "vamana-lvq(svs)": "Graph",
    }
).unpivot(variable_name="algorithm", value_name="family")

FAMILY_COLORS = dict(Tree="tab:orange", Clustering="tab:blue", Graph="tab:red", Hashing="tab:purple", Other="tab:brown")
ALGORITHM_COLORS = dict(lorann="tab:orange", glass="tab:blue", symphonyqg="tab:red")


def radar_chart(
    data, theta, radius, ticks, ax=None, smooth=False, show_percentiles=False, shorten_labels=False, **kwargs
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

    for data_type, t in zip(types, theta[:-1]):
        color = datatype_palette.get(data_type, "red")
        ax.axvline(t, c=color, linewidth=1, alpha=0.6, zorder=5)

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    if show_percentiles:
        for t in yticks[:-1]:
            ax.annotate(xy=(np.pi / 2, t), text=f"{t * 100}%", va="center")
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
    data,
    recall,
    algorithms=[
        "lorann",
        "symphonyqg",
        "glass",
        "ngt-qg",
        "scann",
        "vamana(parlayann)",
        "flatnav",
        "hnsw(faiss)",
        "pynndescent",
    ],
    ncols=5,
    height=4.5,
    k=100,
):
    datasets = data["dataset"].unique().to_list()
    expected_combinations = pl.DataFrame({"dataset": datasets}).join(
        pl.DataFrame({"algorithm": algorithms}), how="cross"
    )

    plot_data = (
        data.filter(pl.col("algorithm").is_in(algorithms))
        .filter(pl.col("k") == k)
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
        .select("algorithm", "dataset", "qps_frac", "dataset-type")
    )

    dataset_order = (plot_data.select("dataset", "dataset-type").unique().sort("dataset-type", "dataset"))[
        "dataset"
    ].to_list()

    algorithm_order = (
        plot_data.with_columns(pl.col("qps_frac").rank(descending=True).over("dataset").alias("rank"))
        .group_by("algorithm")
        .agg(pl.col("rank").mean())
        .sort("rank")
    )["algorithm"].to_list()

    width = height * ncols * 0.5
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

    for algo, ax in zip(algorithm_order, axs[1:]):
        facet_data = plot_data.filter(pl.col("algorithm") == algo)
        radar_chart(
            facet_data.to_pandas(), theta="dataset", radius="qps_frac", ticks=dataset_order, ax=ax, shorten_labels=True
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
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"radar-{recall}.png", dpi=300)
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
    data,
    dataset=None,
    algorithms=[
        "lorann",
        "glass",
        "symphonyqg",
        "flatnav",
        "scann",
        "ngt-qg",
        "hnsw(faiss)",
        "pynndescent",
        "vamana(parlayann)",
        "puffinn",
        "annoy",
    ],
    highlight_algorithms=[
        "lorann",
        "glass",
        "symphonyqg",
    ],
    k=100,
    with_annotations=False,
):
    if dataset is None:
        for dataset in data["dataset"].unique().to_list():
            pareto_plot(data, dataset, algorithms, highlight_algorithms, k, with_annotations)
        return

    plot_data = (
        data.filter(pl.col("dataset") == pl.lit(dataset))
        .filter(pl.col("k") == k)
        .join(algorithm_family, on="algorithm")
    )
    pareto_data = compute_pareto(plot_data)
    best_in_family = (
        pareto_data.filter(pl.col("recall") >= 0.8)
        .filter(pl.col("qps") == pl.col("qps").max().over("algorithm"))
        .select(pl.col("algorithm"), is_best=pl.col("qps") == pl.col("qps").max().over("family"))
    )
    pareto_data = pareto_data.join(best_in_family, on="algorithm")

    def plot_lines(pdata, background=False):
        opacity = 0.2 if background else 1.0
        kwargs = dict(
            data=pdata,
            x="recall",
            y="qps",
            alpha=opacity,
        )
        if background:
            kwargs["color"] = "gray"
        else:
            kwargs["hue"] = "algorithm"
            kwargs["palette"] = ALGORITHM_COLORS
        sns.lineplot(units="algorithm", estimator=None, **kwargs)
        if not background:
            sns.scatterplot(legend=False, **kwargs)

    plt.figure(figsize=(4,3))
    plot_lines(pareto_data.filter(~pl.col("algorithm").is_in(highlight_algorithms)), background=True)
    plot_lines(pareto_data.filter(pl.col("algorithm").is_in(highlight_algorithms)))

    if with_annotations:
        texts = [
            plt.annotate(
                xy=(0.5, row[0]),
                xytext=(0.45, row[0]),
                text=row[1],
                arrowprops=dict(relpos=(1.0, 0.5), arrowstyle="-"),
                ha="right",
                fontsize=8,
            )
            for row in fastest_at(pareto_data, recall=0.5).select("qps", "algorithm").sort("qps").iter_rows()
        ]

    plt.xlim(0.5, 1.0)
    plt.semilogy()
    plt.gca().secondary_yaxis("right")
    if with_annotations:
        plt.gca().secondary_yaxis("right").set_ylabel("qps")
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().get_yaxis().set_visible(False)

    plt.title(dataset)
    sns.move_legend(plt.gca(), "lower left")

    plt.tight_layout()
    if with_annotations:
        adjust_text(texts, 14)
    plt.savefig(OUT_DIR / f"{dataset}-qps-recall.png", dpi=300)
    plt.close()


def rank_at_recall_plot(data, recall, dataset=None, k=100):
    # TODO: shall we do some linear interpolation for the
    # configurations that sport a much higher recall than required?
    if dataset is None:
        for dataset in data["dataset"].unique().to_list():
            rank_at_recall_plot(data, recall, dataset, k)
        return

    algorithm_names = data.select("algorithm").unique()

    plot_data = (
        data.filter(pl.col("dataset") == pl.lit(dataset))
        .filter(pl.col("k") == k)
        .filter(pl.col("recall") >= recall)
        .with_columns(pl.col("qps").rank(descending=True).over("dataset", "algorithm").alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("dataset", "algorithm", "params", "recall", "qps")
        .join(algorithm_names, on="algorithm", how="right")
        .with_columns(
            pl.when(pl.col("qps").is_not_null())
            .then(pl.col("qps").map_elements(lambda x: f"{x:.0f}", return_dtype=pl.String))
            .otherwise(pl.lit("x"))
            .alias("label")
        )
        .with_columns(pl.col("qps", "recall").fill_null(0))
        .join(algorithm_family, on="algorithm")
        .sort("qps", descending=True)
    )
    maxx = plot_data["qps"].max() * 1.1

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=plot_data, hue="family", palette=FAMILY_COLORS, x="qps", y="algorithm")
    for row in plot_data.to_dicts():
        color = "tab:red" if row["label"] == "x" else "black"
        plt.text(row["qps"] + 0.01 * maxx, row["algorithm"], row["label"], color=color, va="center", size=8)

    plt.title(f"{dataset} | min. recall {recall}")
    plt.xlim(0, maxx)
    ax.spines[["top", "bottom", "right"]].set_visible(False)
    for tick in ax.get_xticks():
        if tick > 0:
            plt.axvline(tick, color="white", linewidth=1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{dataset}-ranking-{recall}.png")
    plt.close()


def split_difficulties_plot(
    summary, detail, query_stats, recall, dataset=None, k=100, easy_ptile=0.1, difficult_ptile=0.1
):
    if dataset is None:
        for dataset in summary["dataset"].unique().to_list():
            split_difficulties_plot(summary, detail, query_stats, recall, dataset, k)
        return

    algorithm_names = summary["algorithm"].unique().to_list()

    nqueries = query_stats.group_by("dataset").len("nqueries")

    # FIXME: some queries have a RC score smaller than 1
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
        # Pick the relevnt data
        detail.filter(pl.col("dataset") == pl.lit(dataset))
        .filter(pl.col("k") == k)
        # Select only the easy and difficult queries, for all algorithm's parameterizations
        .join(selected_queries, on=["dataset", "query_index"])
        .select(pl.exclude("dataset", "k", "query_index", "rank_rc100", "nqueries"))
        # Compute the throughput and recall of each algorithm configuration
        # on these "virtual" workloads
        .group_by("algorithm", "params", "difficulty")
        .agg(pl.col("recall").mean(), (1 / pl.col("time").mean()).alias("qps"))
        # Select the fastest configuration with recall above the threshold,
        # for each algorithm and difficulty
        .filter(pl.col("recall") > recall)
        .with_columns(pl.col("qps").rank(descending=True).over("difficulty", "algorithm").alias("qps_rank"))
        .filter(pl.col("qps_rank") == 1)
        .select("algorithm", "difficulty", "qps")
        .pivot(index="algorithm", on="difficulty", values="qps")
        .sort("difficult")
    )

    height = 0.24 * plot_data.select("algorithm").n_unique()
    plt.figure(figsize=(8, height))
    plt.hlines(range(plot_data.shape[0]), xmin=plot_data["easy"], xmax=plot_data["difficult"], color="grey", alpha=0.4)
    plt.scatter(plot_data["difficult"], plot_data["algorithm"], zorder=2, color="steelblue", label="difficult")
    plt.scatter(plot_data["easy"], plot_data["algorithm"], zorder=2, color="#c9a227", label="easy")
    # plt.legend()

    xrange = plot_data["difficult"].max() - plot_data["difficult"].min()

    for algo in plot_data["algorithm"].unique().to_list():
        xpos = plot_data.filter(pl.col("algorithm") == algo)[["difficult", "easy"]].transpose().min()["column_0"][0]
        qps_easy, qps_difficult = tuple(
            plot_data.filter(pl.col("algorithm") == algo)[["easy", "difficult"]].unpivot()["value"]
        )
        if xpos is not None:
            plt.annotate(
                xy=(xpos, algo), xytext=(-35, 0), textcoords="offset points", text=algo, ha="right", va="center", size=9
            )
        if qps_difficult is not None:
            plt.annotate(
                xy=(qps_difficult, algo),
                text=f"{qps_difficult:.0f}",
                ha="right",
                va="center",
                size=9,
                color="steelblue",
                xytext=(-10, 0),
                textcoords="offset points",
            )
        if qps_easy is not None:
            plt.annotate(
                xy=(qps_easy, algo),
                text=f"{qps_easy:.0f}",
                ha="left",
                va="center",
                size=9,
                color="#c9a227",
                xytext=(10, 0),
                textcoords="offset points",
            )

    plt.gca().spines[["top", "bottom", "right", "left"]].set_visible(False)
    plt.title(f"{dataset} | min. recall {recall}")
    # plt.axis("off")
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{dataset}-split-performance.png")
    plt.close()


def plot_difficulty_ridgeline(query_stats, x="rc100", log=True):
    ic(query_stats)
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette(palette="viridis")

    query_stats = query_stats.with_columns(mean_x=pl.col(x).mean().over("dataset")).sort("mean_x", descending=True)
    if log:
        query_stats = query_stats.with_columns(pl.col(x).log())
    labels = query_stats.group_by("dataset").agg(pl.col(x).mean()).sort(x, descending=True)["dataset"].to_list()
    print(query_stats)
    g = sns.FacetGrid(
        query_stats,
        row="dataset",
        aspect=15,
        # hue="mean_x",
        height=0.75,
        sharey=True,
        palette=pal,
    )

    g.map_dataframe(sns.kdeplot, x=x, bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map_dataframe(sns.kdeplot, x=x, bw_adjust=1, clip_on=False, color="w", lw=2)
    g.map(plt.axhline, y=0, lw=1, clip_on=False)

    for ax, label in zip(g.axes.flat, labels):
        ax.text(
            5,
            0.1,
            label,
            ha="right",
            # fontweight='semibold',
            fontsize=12,
            color="black",
        )

    g.fig.subplots_adjust(hspace=-0.8)

    g.set_ylabels("")
    if log:
        g.set_xlabels(f"log({x})")
    else:
        g.set_xlabels(f"{x}")
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"distribution-{x}.png")
    plt.close()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "website/results"
    data_dir = pathlib.Path(data_dir)

    summary = pl.read_parquet(data_dir / "summary.parquet")
    detail = pl.read_parquet(data_dir / "detail.parquet")
    query_stats = pl.read_parquet(data_dir / "stats.parquet")

    # plot_difficulty_ridgeline(query_stats)
    pareto_plot(summary)
    # rank_at_recall_plot(summary, 0.8)
    # split_difficulties_plot(summary, detail, query_stats, 0.8)
    # radar_at_recall_plot(summary.filter(~pl.col("algorithm").is_in(["mlann-rf", "mlann-pca", "roargraph"])), 0.9)
