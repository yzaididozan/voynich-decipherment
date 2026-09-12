#!/usr/bin/env python3
"""Consolidate per-transcription corpus QC reports into one freeze summary.

Default inputs:
    results/qc/ZL3b_corpus_report.json
    results/qc/GC2a_corpus_report.json
    results/qc/IT2a_corpus_report.json
    results/qc/RF1b-full_corpus_report.json
    results/qc/RF1b-basic_corpus_report.json

Default output:
    results/qc/transcription_summary.json

The script intentionally derives QC status from the observed/expected values,
not only from a report's textual status label. This keeps it compatible with
earlier report JSONs produced during parser development while still refusing
real count mismatches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional


EXPECTED_CORPORA = (
    "ZL3b",
    "GC2a",
    "IT2a",
    "RF1b-full",
    "RF1b-basic",
)

DEFAULT_REPORTS = {
    corpus: Path("results/qc") / f"{corpus}_corpus_report.json"
    for corpus in EXPECTED_CORPORA
}

SUMMARY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SummaryRow:
    corpus: str
    source_path: str
    loci: int
    long_words: int
    paragraphs: int
    sta1_codewise: Optional[int]
    sta1_ivtff_semantic: Optional[int]
    sta1_expected: Optional[int]
    sta1_validation: str
    qc: str
    limitation: Optional[str]
    ivtff_alphabet: Optional[str]
    ivtff_version: Optional[str]
    sta1_rules_file: Optional[str]
    sta1_rules_sha256: Optional[str]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing corpus QC report: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")

    return data


def _comparison(report: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError(
            f"{report.get('transcription_id', '<unknown>')}: "
            "missing comparison object"
        )

    result = comparison.get(metric)
    if not isinstance(result, Mapping):
        raise ValueError(
            f"{report.get('transcription_id', '<unknown>')}: "
            f"missing comparison for {metric}"
        )

    return result


def _assert_basic_metric_match(
    report: Mapping[str, Any],
    metric: str,
) -> None:
    result = _comparison(report, metric)
    expected = result.get("expected")
    observed = result.get("observed")

    if expected is None:
        return

    if observed != expected:
        corpus = report.get("transcription_id", "<unknown>")
        raise ValueError(
            f"{corpus}: {metric} mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _sta1_classification(
    corpus: str,
    report: Mapping[str, Any],
) -> tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    str,
    str,
    Optional[str],
]:
    """Return codewise, semantic, expected, validation, qc, limitation."""
    result = _comparison(report, "sta1_characters")

    codewise = report.get("sta1_characters")
    if codewise is None:
        codewise = result.get("observed")

    semantic = report.get("sta1_characters_ivtff_semantic")
    if semantic is None:
        semantic = result.get("observed_ivtff_semantic")

    expected = result.get("expected")

    if corpus == "RF1b-basic":
        # The reduced/basic representation is explicitly treated by this
        # project as lossy and is not assigned an inverse-STA target.
        if expected is not None:
            raise ValueError(
                "RF1b-basic still has an STA1 expected value. "
                "Regenerate its corpus report with the current QC code."
            )

        return (
            codewise,
            semantic,
            None,
            "N/A, lossy representation",
            "PASS WITH DOCUMENTED LIMITATION",
            (
                "RF1b-basic is a lossy reduced-EVA representation; "
                "inverse STA1 character equality is not a valid QC target."
            ),
        )

    if expected is None:
        raise ValueError(f"{corpus}: missing expected STA1 character count")

    if codewise == expected:
        return (
            codewise,
            semantic,
            expected,
            "exact codewise",
            "PASS",
            None,
        )

    if semantic == expected:
        return (
            codewise,
            semantic,
            expected,
            "exact IVTFF-semantic",
            "PASS",
            None,
        )

    raise ValueError(
        f"{corpus}: STA1 count mismatch under all accepted conventions: "
        f"codewise={codewise!r}, semantic={semantic!r}, expected={expected!r}"
    )


def summarize_report(
    corpus: str,
    report: Mapping[str, Any],
) -> SummaryRow:
    report_id = report.get("transcription_id")
    if report_id != corpus:
        raise ValueError(
            f"Expected report {corpus!r}, JSON identifies itself as "
            f"{report_id!r}"
        )

    for metric in ("loci", "paragraphs", "long_words"):
        _assert_basic_metric_match(report, metric)

    (
        sta_codewise,
        sta_semantic,
        sta_expected,
        sta_validation,
        qc,
        limitation,
    ) = _sta1_classification(corpus, report)

    return SummaryRow(
        corpus=corpus,
        source_path=str(report.get("source_path", "")),
        loci=int(report["loci"]),
        long_words=int(report["long_words"]),
        paragraphs=int(report["paragraphs"]),
        sta1_codewise=sta_codewise,
        sta1_ivtff_semantic=sta_semantic,
        sta1_expected=sta_expected,
        sta1_validation=sta_validation,
        qc=qc,
        limitation=limitation,
        ivtff_alphabet=report.get("ivtff_alphabet"),
        ivtff_version=report.get("ivtff_version"),
        sta1_rules_file=report.get("sta1_rules_file"),
        sta1_rules_sha256=report.get("sta1_rules_sha256"),
    )


def build_summary(report_paths: Mapping[str, Path]) -> Dict[str, Any]:
    rows: List[SummaryRow] = []

    for corpus in EXPECTED_CORPORA:
        report = _load_json(report_paths[corpus])
        rows.append(summarize_report(corpus, report))

    pass_count = sum(row.qc == "PASS" for row in rows)
    limitation_count = sum(
        row.qc == "PASS WITH DOCUMENTED LIMITATION"
        for row in rows
    )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "checkpoint": "corpus_layer_pre_split",
        "overall_qc": "PASS",
        "corpora_total": len(rows),
        "corpora_pass": pass_count,
        "corpora_pass_with_documented_limitation": limitation_count,
        "rows": [asdict(row) for row in rows],
        "freeze_readiness": {
            "published_count_validation_complete": True,
            "manual_cross_transcription_spotcheck_complete": False,
            "ready_to_generate_locked_split": False,
        },
    }


def print_table(summary: Mapping[str, Any]) -> None:
    rows = summary["rows"]

    headers = (
        "Corpus",
        "Loci",
        "Words",
        "STA1 validation",
        "QC",
    )

    printable = [
        (
            row["corpus"],
            f'{row["loci"]:,}',
            f'{row["long_words"]:,}',
            row["sta1_validation"],
            row["qc"],
        )
        for row in rows
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in printable))
        for i in range(len(headers))
    ]

    def render(row):
        return " | ".join(
            value.ljust(widths[i])
            for i, value in enumerate(row)
        )

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in printable:
        print(render(row))

    print()
    print(f"Overall QC: {summary['overall_qc']}")
    print(
        "Split readiness: NOT YET — manual cross-transcription "
        "spot-check remains."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the five Voynich corpus-QC JSON reports into "
            "one pre-split checkpoint."
        )
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=Path("results/qc"),
        help="Directory containing *_corpus_report.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qc/transcription_summary.json"),
    )
    args = parser.parse_args()

    report_paths = {
        corpus: args.qc_dir / f"{corpus}_corpus_report.json"
        for corpus in EXPECTED_CORPORA
    }

    try:
        summary = build_summary(report_paths)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"CORPUS SUMMARY QC ERROR: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_table(summary)
    print(f"\nSaved summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
