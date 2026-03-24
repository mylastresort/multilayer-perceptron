#!/usr/bin/env python3
"""Generate a pandas profiling report for a dataset file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from ydata_profiling import ProfileReport
except ImportError:
    # Backward compatibility for older environments.
    from pandas_profiling import ProfileReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pandas profiling report for a dataset."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to the dataset file (csv, parquet, json, xlsx, xls).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML report path (example: reports/profile.html).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where profile_report.html will be written.",
    )
    parser.add_argument(
        "--title",
        default="Dataset Profile Report",
        help="Title shown in the generated report.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Enable minimal mode for large datasets.",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=None,
        help="Number of rows to skip before parsing data.",
    )
    parser.add_argument(
        "--header",
        default="infer",
        help="Header row index (e.g. 0) or 'none' for no header. Default: infer.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated column names to use.",
    )
    args = parser.parse_args()

    if args.output is not None and args.output_dir is not None:
        parser.error("Use either --output or --output-dir, not both.")

    return args


def resolve_output_path(output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        return output
    if output_dir is not None:
        return output_dir / "profile_report.html"
    return Path("profile_report.html")


def parse_header_value(header: str) -> int | None | str:
    value = header.strip().lower()
    if value == "none":
        return None
    if value == "infer":
        return "infer"
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("--header must be an integer, 'infer', or 'none'.") from exc


def parse_names_value(names: str | None) -> list[str] | None:
    if names is None:
        return None
    parsed = [name.strip() for name in names.split(",") if name.strip()]
    if not parsed:
        raise ValueError("--names must contain at least one non-empty column name.")
    return parsed


def load_dataframe(
    data_path: Path,
    *,
    skiprows: int | None,
    header: int | None | str,
    names: list[str] | None,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    read_options = {
        "skiprows": skiprows,
        "header": header,
        "names": names,
    }
    read_options = {k: v for k, v in read_options.items() if v is not None}

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path, **read_options)
    if suffix == ".parquet":
        if read_options:
            raise ValueError(
                "--skiprows/--header/--names are only supported for .csv and .xlsx/.xls files."
            )
        return pd.read_parquet(data_path)
    if suffix == ".json":
        if read_options:
            raise ValueError(
                "--skiprows/--header/--names are only supported for .csv and .xlsx/.xls files."
            )
        return pd.read_json(data_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(data_path, **read_options)

    raise ValueError(
        "Unsupported file type. Supported extensions: .csv, .parquet, .json, .xlsx, .xls"
    )


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output, args.output_dir)
    header_value = parse_header_value(args.header)
    names_value = parse_names_value(args.names)

    df = load_dataframe(
        args.data_path,
        skiprows=args.skiprows,
        header=header_value,
        names=names_value,
    )

    report = ProfileReport(df, title=args.title, explorative=True, minimal=args.minimal)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_file(output_path)
    print(f"Report generated at: {output_path.resolve()}")


if __name__ == "__main__":
    main()
