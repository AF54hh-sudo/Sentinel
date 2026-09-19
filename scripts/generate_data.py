"""Generate and validate the Phase 2 CSV dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.config import get_settings
from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.data.loader import write_csv_dataset
from sentinel.data.validation import validate_dataset
from sentinel.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    config = GenerationConfig(seed=args.seed if args.seed is not None else settings.random_seed)
    tables = AMXTechDataGenerator(config).generate()
    report = validate_dataset(tables)
    report.raise_for_errors()
    output_dir = args.output_dir or settings.data_dir
    write_csv_dataset(tables, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "rows": report.row_counts, "warnings": report.warnings}, indent=2))


if __name__ == "__main__":
    main()
