import argparse

from vibe.datasets import DATASETS
from vibe.runner import get_dataset_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--extract-path", metavar="NAME", help="path to extract datasets to", default=None)
    args = parser.parse_args()
    fn = get_dataset_fn(args.dataset)

    if args.extract_path:
        DATASETS[args.dataset](fn, args.extract_path)
    else:
        DATASETS[args.dataset](fn)
