"""Convert one AMX Tech business question into a validated Phase 12 intent."""

from __future__ import annotations

import argparse
import json

from sentinel.dashboard import load_dashboard_bundle
from sentinel.llm import QuestionContext, create_intent_client


def main(question: str, source: str = "csv") -> int:
    bundle = load_dashboard_bundle(source)
    start_date, end_date = bundle.date_bounds
    context = QuestionContext(
        data_start_date=start_date,
        data_end_date=end_date,
        regions=list(bundle.regions),
        plans=list(bundle.plans),
        industries=sorted(bundle.tables["customers"].industry.dropna().unique()),
    )
    result = create_intent_client().understand_question(question, context)
    output = {
        "provider": result.provider,
        "model": result.model,
        "response_id": result.response_id,
        "intent": result.intent.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Free-form business question")
    parser.add_argument("--source", choices=("csv", "database"), default="csv")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.question, arguments.source))
