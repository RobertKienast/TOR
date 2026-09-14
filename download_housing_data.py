#!/usr/bin/env python3
"""Sequentially prefetch California Housing into every benchmark cache scope."""

import argparse
from pathlib import Path

from sklearn.datasets import fetch_california_housing


DEFAULT_DATA_ROOT = Path("/srv/scratch/z5591496/newStart/data")
HOUSING_SCOPES = ("housing_small", "housing_medium", "housing", "housing_test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root used by open_l2o_problems.py (default: {DEFAULT_DATA_ROOT})",
    )
    args = parser.parse_args()

    cache_root = args.data_root / "_task_data_cache"
    for scope in HOUSING_SCOPES:
        target = cache_root / scope
        target.mkdir(parents=True, exist_ok=True)
        print(f"Fetching California Housing -> {target}", flush=True)
        bunch = fetch_california_housing(
            data_home=str(target),
            download_if_missing=True,
        )
        if bunch.data.shape != (20640, 8) or bunch.target.shape != (20640,):
            raise RuntimeError(
                f"Unexpected California Housing shape in {target}: "
                f"X={bunch.data.shape}, y={bunch.target.shape}"
            )
        print(f"  ready: {bunch.data.shape[0]} rows, {bunch.data.shape[1]} features", flush=True)

    print("All Housing cache scopes are ready.", flush=True)


if __name__ == "__main__":
    main()
