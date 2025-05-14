# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "polars",
# ]
# ///
import polars as pl
import sys
from pathlib import Path

all_data = pl.read_parquet(sys.argv[1])
out_dir = Path(sys.argv[2])

for dataset in all_data["dataset"].unique():
    print(dataset)
    data = all_data.filter(pl.col("dataset") == dataset)
    data.write_parquet(out_dir / f"{dataset}__detail.parquet")


