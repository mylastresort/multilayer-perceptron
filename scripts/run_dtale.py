#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import dtale
import pandas as pd


def parse_header_value(value: str) -> int | None | Literal["infer"]:
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    if lowered == "infer":
        return "infer"
    try:
        return int(lowered)
    except ValueError as exc:
        raise ValueError("--header must be an integer, infer, or none") from exc


def parse_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names:
        raise ValueError("--names must include at least one column name")
    return names


def default_feature_names() -> list[str]:
    base_features = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]
    stats = ["mean", "se", "extreme"]
    expanded = [f"{feature}_{stat}" for feature in base_features for stat in stats]
    return ["ID", "Diagnosis", *expanded]


def load_dataframe(
    data_path: Path,
    skiprows: int | None,
    header: int | None | Literal["infer"],
    names: list[str] | None,
) -> pd.DataFrame:
    return pd.read_csv(data_path, skiprows=skiprows, header=header, names=names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-Tale for a CSV dataset")
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=Path("data/data.csv"),
        help="Path to CSV dataset (default: data/data.csv)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="D-Tale host")
    parser.add_argument("--port", type=int, default=40000, help="D-Tale port")
    parser.add_argument("--skiprows", type=int, default=1, help="Rows to skip")
    parser.add_argument(
        "--header",
        default="none",
        help="Header row index, infer, or none (default: none)",
    )
    parser.add_argument("--names", default=None, help="Comma-separated column names")
    parser.add_argument(
        "--use-default-schema",
        action="store_true",
        help="Use ID, Diagnosis, and the expanded 30-feature schema",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    header = parse_header_value(args.header)
    names = default_feature_names() if args.use_default_schema else parse_names(args.names)

    df = load_dataframe(args.data_path, args.skiprows, header, names)
    instance = dtale.show(
        df,
        host=args.host,
        port=args.port,
        subprocess=False,
        open_browser=False,
    )

    print(f"D-Tale URL: {instance.main_url()}")
    print("Press Ctrl+C to stop D-Tale.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping D-Tale...")


if __name__ == "__main__":
    main()