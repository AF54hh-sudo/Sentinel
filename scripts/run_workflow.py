"""Run one AMX Tech question through the bounded Phase 13 workflow."""

from __future__ import annotations

import argparse
import json

from sentinel.agents import run_workflow
from sentinel.dashboard import load_dashboard_bundle


def main(question: str, source: str = "csv") -> int:
    bundle = load_dashboard_bundle(source)
    result = run_workflow(bundle, question)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Free-form business question")
    parser.add_argument("--source", choices=("csv", "database"), default="csv")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.question, arguments.source))
