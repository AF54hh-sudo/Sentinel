"""Run and persist the AMX Tech Phase 8 cloud-cost anomaly analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from sentinel.anomaly import (
    CloudCostAnomalyConfig,
    detect_cloud_cost_anomalies,
    evaluate_detection_period,
)
from sentinel.config import get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.eda.cleaning import clean_dataset


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _evaluation_period(path: Path) -> tuple[str, str]:
    """Read the evaluation-only period; this function is never imported by runtime code."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario = next(
        item for item in payload["scenarios"] if item["id"] == "cloud_gpu_anomaly"
    )
    start, end = scenario["period"].split("/", maxsplit=1)
    return start, end


def _load_tables(source: str) -> dict[str, Any]:
    if source == "csv":
        return read_csv_dataset(Path("data/generated"))
    engine = create_database_engine(get_settings().database_url)
    try:
        return read_database_tables(engine)
    finally:
        engine.dispose()


def main(source: str = "database") -> int:
    tables, cleaning = clean_dataset(_load_tables(source))

    config = CloudCostAnomalyConfig()
    detections, isolation_forest, report = detect_cloud_cost_anomalies(
        tables["cloud_costs"], config
    )
    expected_start, expected_end = _evaluation_period(
        Path("data/ground_truth/scenarios.yaml")
    )
    report["evaluation_only_ground_truth"] = {
        flag: evaluate_detection_period(
            detections,
            expected_start,
            expected_end,
            flag_column=flag,
        )
        for flag in ("iqr_anomaly", "isolation_forest_anomaly", "strong_anomaly")
    }

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_schema_version": 1,
        "company": "AMX Tech",
        "model": isolation_forest,
        "configuration": asdict(config),
        "iqr_ratio_bounds": report["iqr_ratio_bounds"],
        "model_input": "log(observed daily metric / centered rolling-median baseline)",
        "final_rule": "strong outer-IQR outlier AND Isolation Forest anomaly",
        "intended_use": "Retrospective daily company-wide GPU-cost anomaly review.",
    }
    joblib.dump(artifact, output_dir / "cloud_cost_anomaly_detector.joblib")
    (output_dir / "cloud_cost_anomaly_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    detections.to_csv(output_dir / "cloud_cost_anomaly_scores.csv", index=False)

    evaluation = report["evaluation_only_ground_truth"]["strong_anomaly"]
    print(
        json.dumps(
            {
                "company": "AMX Tech",
                "metric": config.metric,
                "strong_anomalies": report["counts"]["final_strong_anomalies"],
                "detected_dates": evaluation["detected_dates"],
                "ground_truth_precision": evaluation["precision"],
                "ground_truth_recall": evaluation["recall"],
                "output_dir": str(output_dir),
                "source": source,
                "duplicates_removed": cleaning.duplicate_events_removed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("database", "csv"), default="database")
    raise SystemExit(main(parser.parse_args().source))
