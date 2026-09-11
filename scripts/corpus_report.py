#!/usr/bin/env python3
"""Generate the first reproducible corpus-QC report for an IVTFF file.

Primary use:

    python scripts/corpus_report.py \
        data/raw/zl/ZL3b-n.txt \
        --transcription-id ZL3b

For ZL3b, this script compares directly reproducible structural counts with
the source-published values:

    loci       5,385
    paragraphs   740
    long words 36,278

The published character total (157,304) is explicitly *not* reproduced by
this script because that number is defined after conversion to the STA1
alphabet. Counting raw EVA code points would be a different measurement and
would create a false validation result. STA1 character validation should be
implemented as a separate QC stage.

"Long words" follows the source definition: uncertain word spaces (`,` in
IVTFF) do not separate words. Confident spaces (`.`) and drawing-intrusion
word spaces (`<->`, `<~>`) do.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

# Allow direct execution from the repository root:
#     python scripts/corpus_report.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ivtff import LocusRecord, load_ivtff


PUBLISHED_COUNTS = {
    "ZL3b": {
        "loci": 5385,
        "paragraphs": 740,
        "long_words": 36278,
        # Published after STA1 conversion; intentionally not computed here.
        "sta1_characters": 157304,
    },
    "GC2a": {
        "loci": 5367,
        "paragraphs": 775,
        "long_words": 38242,
        "sta1_characters": 157096,
    },
    "IT2a": {
        "loci": 5215,
        "paragraphs": 772,
        "long_words": 37919,
        "sta1_characters": 155292,
    },
    "RF1b-full": {
        "loci": 5385,
        "paragraphs": None,
        "long_words": 37848,
        "sta1_characters": 157254,
    },
    "RF1b-basic": {
        "loci": 5385,
        "paragraphs": None,
        "long_words": 37848,
        "sta1_characters": 157254,
    },
}


# Inline IVTFF markup that is metadata rather than transliterated text.
_TEXT_TAG_RE = re.compile(r"<@[A-Z]=[^>]>")
_FREE_COMMENT_RE = re.compile(r"<![^>]*>")


def _text_for_long_word_count(text: str) -> str:
    """Return text suitable for source-style 'long word' counting.

    This is intentionally narrow:
    - paragraph markers are removed;
    - text tags/free comments are removed;
    - drawing interruptions become confident boundaries;
    - uncertain spaces (comma) remain inside a long word;
    - alternative readings, ligatures, high-ASCII encodings, and unreadable
      symbols remain represented because they belong to the transliteration.
    """
    text = text.replace("<%>", "").replace("<$>", "")
    text = _TEXT_TAG_RE.sub("", text)
    text = _FREE_COMMENT_RE.sub("", text)
    text = text.replace("<->", ".").replace("<~>", ".")
    return text


def count_long_words(records: Iterable[LocusRecord]) -> int:
    """Count words while *not* treating uncertain comma spaces as boundaries."""
    total = 0

    for record in records:
        text = _text_for_long_word_count(record.text_normalized)

        # IVTFF '.' is a confident word space.  Commas intentionally stay
        # inside spans for the "long words" statistic.
        for span in text.split("."):
            span = span.strip()
            if span:
                total += 1

    return total


@dataclass
class CorpusReport:
    transcription_id: str
    source_path: str
    file_size_bytes: int
    ivtff_alphabet: Optional[str]
    ivtff_version: Optional[str]
    ivtff_options: Optional[str]

    loci: int
    pages: int
    physical_leaves_observed: int
    paragraphs: int
    long_words: int

    uncertain_space_occurrences: int
    alternative_reading_occurrences: int
    drawing_intrusion_occurrences: int
    high_ascii_occurrences: int
    unknown_single_occurrences: int
    unknown_run_occurrences: int

    currier_counts: dict
    scribe_counts: dict
    section_counts: dict
    locus_type_counts: dict

    published_targets: Optional[dict]
    comparison: Optional[dict]


def _sorted_counter(values) -> dict:
    counter = Counter(values)
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda x: str(x))
    }


def build_report(
    source: Path,
    transcription_id: str,
    *,
    strict: bool = True,
) -> CorpusReport:
    records = load_ivtff(
        source,
        transcription_id=transcription_id,
        strict=strict,
    )

    if not records:
        raise RuntimeError(f"No locus records parsed from {source}")

    all_text = "\n".join(r.text_normalized for r in records)

    loci = len(records)
    pages = len({r.folio for r in records})
    physical_leaves = {
        r.physical_leaf for r in records if r.physical_leaf is not None
    }
    paragraphs = sum(1 for r in records if r.paragraph_start)
    long_words = count_long_words(records)

    targets = PUBLISHED_COUNTS.get(transcription_id)

    comparison = None
    if targets is not None:
        comparison = {}
        for metric, observed in {
            "loci": loci,
            "paragraphs": paragraphs,
            "long_words": long_words,
        }.items():
            expected = targets.get(metric)
            if expected is None:
                comparison[metric] = {
                    "observed": observed,
                    "expected": None,
                    "delta": None,
                    "status": "not_published",
                }
            else:
                delta = observed - expected
                comparison[metric] = {
                    "observed": observed,
                    "expected": expected,
                    "delta": delta,
                    "status": "PASS" if delta == 0 else "MISMATCH",
                }

        comparison["sta1_characters"] = {
            "observed": None,
            "expected": targets.get("sta1_characters"),
            "delta": None,
            "status": "PENDING_STA1_CONVERSION",
        }

    return CorpusReport(
        transcription_id=transcription_id,
        source_path=str(source),
        file_size_bytes=source.stat().st_size,
        ivtff_alphabet=records[0].ivtff_alphabet,
        ivtff_version=records[0].ivtff_version,
        ivtff_options=records[0].ivtff_options,
        loci=loci,
        pages=pages,
        physical_leaves_observed=len(physical_leaves),
        paragraphs=paragraphs,
        long_words=long_words,
        uncertain_space_occurrences=all_text.count(","),
        alternative_reading_occurrences=len(re.findall(r"\[[^\]]+\]", all_text)),
        drawing_intrusion_occurrences=all_text.count("<->") + all_text.count("<~>"),
        high_ascii_occurrences=len(re.findall(r"@\d{3};", all_text)),
        unknown_single_occurrences=len(re.findall(r"(?<!\?)\?(?!\?)", all_text)),
        unknown_run_occurrences=len(re.findall(r"\?{3,}", all_text)),
        currier_counts=_sorted_counter(
            r.currier if r.currier is not None else "UNSET"
            for r in records
        ),
        scribe_counts=_sorted_counter(
            r.scribe if r.scribe is not None else "UNSET"
            for r in records
        ),
        section_counts=_sorted_counter(
            r.section if r.section is not None else "UNSET"
            for r in records
        ),
        locus_type_counts=_sorted_counter(r.locus_type for r in records),
        published_targets=targets,
        comparison=comparison,
    )


def print_report(report: CorpusReport) -> None:
    print("=" * 72)
    print(f"IVTFF CORPUS REPORT — {report.transcription_id}")
    print("=" * 72)
    print(f"Source:              {report.source_path}")
    print(f"File size:           {report.file_size_bytes:,} bytes")
    print(f"IVTFF alphabet:      {report.ivtff_alphabet}")
    print(f"IVTFF version:       {report.ivtff_version}")
    print(f"IVTFF options:       {report.ivtff_options}")
    print()
    print("STRUCTURE")
    print(f"  Loci:              {report.loci:,}")
    print(f"  Pages with text:   {report.pages:,}")
    print(f"  Physical leaves:   {report.physical_leaves_observed:,}")
    print(f"  Paragraphs:        {report.paragraphs:,}")
    print(f"  Long words:        {report.long_words:,}")
    print()
    print("IVTFF FEATURES")
    print(f"  Uncertain spaces:  {report.uncertain_space_occurrences:,}")
    print(f"  Alt. readings:     {report.alternative_reading_occurrences:,}")
    print(f"  Drawing breaks:    {report.drawing_intrusion_occurrences:,}")
    print(f"  High-ASCII codes:  {report.high_ascii_occurrences:,}")
    print(f"  Single ?:          {report.unknown_single_occurrences:,}")
    print(f"  ??? runs:          {report.unknown_run_occurrences:,}")

    if report.comparison is not None:
        print()
        print("PUBLISHED-COUNT CHECK")
        for metric in ("loci", "paragraphs", "long_words", "sta1_characters"):
            result = report.comparison[metric]
            observed = result["observed"]
            expected = result["expected"]
            status = result["status"]

            obs_text = "pending" if observed is None else f"{observed:,}"
            exp_text = "n/a" if expected is None else f"{expected:,}"
            print(
                f"  {metric:16s} observed={obs_text:>9s} "
                f"expected={exp_text:>9s}  {status}"
            )

    for title, values in (
        ("CURRIER", report.currier_counts),
        ("SCRIBES", report.scribe_counts),
        ("SECTIONS", report.section_counts),
        ("LOCUS TYPES", report.locus_type_counts),
    ):
        print()
        print(title)
        for key, value in values.items():
            print(f"  {key:24s} {value:,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible IVTFF corpus-QC report."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--transcription-id",
        required=True,
        help="Dataset ID from data/manifest.yaml, e.g. ZL3b",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save the complete report as JSON.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow parser to skip unrecognised data lines (not recommended for QC).",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero if a directly reproducible published count mismatches.",
    )
    args = parser.parse_args()

    report = build_report(
        args.source,
        args.transcription_id,
        strict=not args.non_strict,
    )

    print_report(report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved JSON report: {args.json}")

    if args.fail_on_mismatch and report.comparison:
        mismatches = [
            key
            for key, result in report.comparison.items()
            if result["status"] == "MISMATCH"
        ]
        if mismatches:
            print(
                "\nQC FAILURE: published-count mismatch in "
                + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
