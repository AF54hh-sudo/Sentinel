"""Train, evaluate, and persist the AMX Tech Phase 7 churn model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.eda.cleaning import clean_dataset
from sentinel.features.customer_features import ChurnSnapshotConfig
from sentinel.ml.training import ChurnTrainingConfig, train_churn_model


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


def main() -> int:
    engine = create_database_engine(get_settings().database_url)
    try:
        tables, cleaning = clean_dataset(read_database_tables(engine))
    finally:
        engine.dispose()
    model, report, ranking = train_churn_model(
        tables,
        snapshot_config=ChurnSnapshotConfig(),
        training_config=ChurnTrainingConfig(),
    )
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_schema_version": 1,
        "company": "AMX Tech",
        "model": model,
        "threshold": report["threshold_selection"]["selected_threshold"],
        "snapshot": report["snapshot"],
        "features": report["features"],
        "selected_model": report["selected_model"],
        "training_cohort_size": report["cohort"]["customers"],
        "intended_use": (
            "Rank customers active at the documented snapshot by churn risk within "
            "the documented prediction horizon."
        ),
    }
    joblib.dump(artifact, output_dir / "churn_model.joblib")
    (output_dir / "churn_model_report.json").write_text(
        json.dumps(_json_safe({"company": "AMX Tech", **report}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    ranking.to_csv(output_dir / "churn_test_predictions.csv", index=False)
    print(
        json.dumps(
            {
                "company": "AMX Tech",
                "selected_model": report["selected_model"],
                "held_out_average_precision": report["held_out_test_metrics"][
                    "average_precision"
                ],
                "held_out_recall": report["held_out_test_metrics"]["recall"],
                "output_dir": str(output_dir),
                "duplicates_removed": cleaning.duplicate_events_removed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
