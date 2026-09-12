#!/usr/bin/env python3
"""Summarize a completed manual manuscript-image validation worksheet."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys


DEFAULT_INPUT = Path("results/qc/image_validation/v1")
DEFAULT_OUTPUT = Path(
    "results/qc/image_validation/v1/review_summary.json"
)

ALLOWED_STATUSES = {
    "PASS",
    "TRANSCRIPTION_DISAGREEMENT",
    "PARSER_OR_ALIGNMENT_ISSUE",
    "UNRESOLVED",
}


def read_csv(path: Path):
    if not path.is_file():
        raise ValueError(f"Missing worksheet: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_statuses(rows, field, label):
    invalid = []
    blank = []

    for row_number, row in enumerate(rows, start=2):
        value = row.get(field, "").strip()
        if not value:
            blank.append(row_number)
        elif value not in ALLOWED_STATUSES:
            invalid.append((row_number, value))

    if invalid:
        raise ValueError(
            f"{label}: invalid {field} values: {invalid}"
        )

    return blank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        pages = read_csv(args.input_dir / "pages.csv")
        loci = read_csv(args.input_dir / "loci.csv")

        blank_pages = validate_statuses(
            pages,
            "review_status",
            "pages.csv",
        )
        blank_loci = validate_statuses(
            loci,
            "check_status",
            "loci.csv",
        )

        page_counts = Counter(
            row["review_status"].strip()
            for row in pages
            if row["review_status"].strip()
        )
        locus_counts = Counter(
            row["check_status"].strip()
            for row in loci
            if row["check_status"].strip()
        )

        complete = not blank_pages and not blank_loci
        parser_issues = (
            page_counts["PARSER_OR_ALIGNMENT_ISSUE"]
            + locus_counts["PARSER_OR_ALIGNMENT_ISSUE"]
        )
        unresolved = (
            page_counts["UNRESOLVED"]
            + locus_counts["UNRESOLVED"]
        )

        if not complete:
            checkpoint = "INCOMPLETE"
        elif parser_issues:
            checkpoint = "FAIL_REQUIRES_PIPELINE_REVIEW"
        elif unresolved:
            checkpoint = "INCOMPLETE_UNRESOLVED"
        else:
            checkpoint = "PASS_WITH_DOCUMENTED_TRANSCRIPTION_VARIATION"

        summary = {
            "schema_version": "1.0",
            "checkpoint": checkpoint,
            "complete": complete,
            "scope": (
                "manual train-only cross-check against manuscript images"
            ),
            "page_rows": len(pages),
            "locus_rows": len(loci),
            "blank_page_status_rows": blank_pages,
            "blank_locus_status_rows": blank_loci,
            "page_status_counts": dict(sorted(page_counts.items())),
            "locus_status_counts": dict(sorted(locus_counts.items())),
            "parser_or_alignment_issue_count": parser_issues,
            "unresolved_count": unresolved,
            "locked_test_accessed": False,
        }

        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print("=" * 72)
        print("MANUSCRIPT-IMAGE VALIDATION REVIEW SUMMARY")
        print("=" * 72)
        print(f"Pages:       {len(pages)}")
        print(f"Locus checks:{len(loci):>5}")
        print()
        print("Page statuses:")
        for key, count in sorted(page_counts.items()):
            print(f"  {key:<30} {count}")
        print("Locus statuses:")
        for key, count in sorted(locus_counts.items()):
            print(f"  {key:<30} {count}")
        print()
        print(f"Checkpoint: {checkpoint}")
        print(f"Saved:      {args.output}")

        if checkpoint.startswith("FAIL"):
            return 2
        if checkpoint.startswith("INCOMPLETE"):
            return 1
        return 0

    except Exception as exc:
        print(f"IMAGE-VALIDATION SUMMARY ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
