"""Filesystem persistence for generated datasets."""

from pathlib import Path

import pandas as pd


def write_csv_dataset(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def read_csv_dataset(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {path.stem: pd.read_csv(path) for path in sorted(input_dir.glob("*.csv"))}

