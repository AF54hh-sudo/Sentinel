"""Build a citation-validated AMX Tech business report for one question."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentinel.agents import run_workflow
from sentinel.dashboard import load_dashboard_bundle
from sentinel.rag import retrieve_documents
from sentinel.reporting import (
    build_report_from_workflow,
    create_report_narrator,
    render_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Free-form business question")
    parser.add_argument("--source", choices=("csv", "database"), default="csv")
    parser.add_argument(
        "--with-documents",
        action="store_true",
        help="Retrieve and include pgvector document context",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the offline evidence-copying narrator instead of hosted report narration",
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_dashboard_bundle(args.source)
    workflow = run_workflow(bundle, args.question)
    documents = retrieve_documents(args.question) if args.with_documents else []
    narrator = None if args.deterministic else create_report_narrator()
    report = build_report_from_workflow(
        workflow,
        documents=documents,
        narrator=narrator,
    )
    markdown = render_report_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Saved validated report to {args.output}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
