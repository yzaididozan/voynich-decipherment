#!/usr/bin/env python3
"""Generate reproducible corpus-QC reports for IVTFF transliterations.

Example:

    python scripts/corpus_report.py \
        data/raw/zl/ZL3b-n.txt \
        --transcription-id ZL3b \
        --json results/qc/ZL3b_corpus_report.json

The report validates published locus, paragraph, long-word, and STA1
character counts. STA1 character validation is an explicit derived stage:
the lossless IVTFF parser remains unchanged, while ``src.data.sta1`` converts
native transliteration text using locally stored official bitrans rules.

For STA1, the report preserves three named measures: codewise STA1,
IVTFF-semantic unreadables (literal native ??? is one unknown sequence), and
an all-adjacent-Z1-collapse diagnostic. If a file contains unreadable syntax
outside the IVTFF-defined forms, the semantic measure is reported unavailable
rather than guessed. A published value is considered
reproduced if it exactly matches either of the first two documented
conventions; the matched convention is recorded explicitly. The lossy
RF1b-basic representation is not assigned an inverse-STA character target.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ivtff import LocusRecord, load_ivtff
from src.data.sta1 import (
    DEFAULT_RULES_DIR,
    STA1Error,
    count_records_as_sta1,
)


PUBLISHED_COUNTS = {
    "ZL3b": {
        "loci": 5385,
        "paragraphs": 740,
        "long_words": 36278,
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
        # Basic Eva is a lossy simplification of RF's STA representation.
        # The single published RF character total therefore is not a valid
        # inverse-conversion target for RF1b-er.
        "sta1_characters": None,
    },
}


_TEXT_TAG_RE = re.compile(r"<@[A-Z]=[^>]>")
_FREE_COMMENT_RE = re.compile(r"<![^>]*>")


def _text_for_long_word_count(text: str) -> str:
    """Return text suitable for the published 'long word' convention."""
    text = text.replace("<%>", "").replace("<$>", "")
    text = _TEXT_TAG_RE.sub("", text)
    text = _FREE_COMMENT_RE.sub("", text)
    text = text.replace("<->", ".").replace("<~>", ".")
    return text


def count_long_words(records: Iterable[LocusRecord]) -> int:
    """Count words without treating uncertain comma-spaces as boundaries."""
    total = 0

    for record in records:
        text = _text_for_long_word_count(record.text_normalized)
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
    sta1_characters: int
    sta1_characters_ivtff_semantic: Optional[int]
    sta1_ivtff_semantic_valid: bool
    sta1_ivtff_semantic_error: Optional[str]
    sta1_characters_collapsed_unknown_runs: int
    sta1_unknown_collapse_delta: int

    sta1_rules_file: str
    sta1_rules_sha256: str

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
    sta1_rules_dir: Path = DEFAULT_RULES_DIR,
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

    # Explicit derived representation. Missing/incompatible rule files are a
    # QC error rather than silently reverting to raw EVA/v101 character count.
    sta1 = count_records_as_sta1(
        records,
        rules_dir=sta1_rules_dir,
        strict=True,
    )

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

        sta_expected = targets.get("sta1_characters")
        if sta_expected is None:
            comparison["sta1_characters"] = {
                "observed": sta1.characters,
                "observed_ivtff_semantic": (
                    sta1.characters_ivtff_semantic
                ),
                "expected": None,
                "delta": None,
                "status": "not_applicable_lossy_representation",
                "matched_convention": None,
            }
        elif sta1.characters == sta_expected:
            comparison["sta1_characters"] = {
                "observed": sta1.characters,
                "observed_ivtff_semantic": (
                    sta1.characters_ivtff_semantic
                ),
                "expected": sta_expected,
                "delta": 0,
                "status": "PASS_CODEWISE_STA1",
                "matched_convention": "codewise_sta1",
            }
        elif (
            sta1.characters_ivtff_semantic is not None
            and sta1.characters_ivtff_semantic == sta_expected
        ):
            comparison["sta1_characters"] = {
                "observed": sta1.characters,
                "observed_ivtff_semantic": (
                    sta1.characters_ivtff_semantic
                ),
                "expected": sta_expected,
                "delta": 0,
                "status": "PASS_IVTFF_SEMANTIC",
                "matched_convention": "ivtff_unknown_sequence",
            }
        else:
            comparison["sta1_characters"] = {
                "observed": sta1.characters,
                "observed_ivtff_semantic": (
                    sta1.characters_ivtff_semantic
                ),
                "expected": sta_expected,
                "delta": sta1.characters - sta_expected,
                "status": "MISMATCH",
                "matched_convention": None,
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
        sta1_characters=sta1.characters,
        sta1_characters_ivtff_semantic=(
            sta1.characters_ivtff_semantic
        ),
        sta1_ivtff_semantic_valid=sta1.ivtff_semantic_valid,
        sta1_ivtff_semantic_error=sta1.ivtff_semantic_error,
        sta1_characters_collapsed_unknown_runs=(
            sta1.characters_collapsed_unknown_runs
        ),
        sta1_unknown_collapse_delta=sta1.unknown_collapse_delta,
        sta1_rules_file=sta1.rule_file,
        sta1_rules_sha256=sta1.rule_sha256,
        uncertain_space_occurrences=all_text.count(","),
        alternative_reading_occurrences=len(
            re.findall(r"\[[^\]]+\]", all_text)
        ),
        drawing_intrusion_occurrences=(
            all_text.count("<->") + all_text.count("<~>")
        ),
        high_ascii_occurrences=len(
            re.findall(r"@\d{3};", all_text)
        ),
        unknown_single_occurrences=len(
            re.findall(r"(?<!\?)\?(?!\?)", all_text)
        ),
        unknown_run_occurrences=len(
            re.findall(r"\?{3,}", all_text)
        ),
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
        locus_type_counts=_sorted_counter(
            r.locus_type for r in records
        ),
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
    print(f"  STA1 characters (codewise): {report.sta1_characters:,}")
    if report.sta1_characters_ivtff_semantic is None:
        print("  STA1 characters (IVTFF semantic): unavailable")
    else:
        print(
            "  STA1 characters (IVTFF semantic): "
            f"{report.sta1_characters_ivtff_semantic:,}"
        )
    print(
        "  STA1 chars (all adjacent Z1 collapsed, diagnostic): "
        f"{report.sta1_characters_collapsed_unknown_runs:,}"
    )
    print(
        "  Unknown-run delta: "
        f"{report.sta1_unknown_collapse_delta:,}"
    )
    print()
    print("STA1 DERIVATION")
    print(f"  Rules file:        {report.sta1_rules_file}")
    print(f"  Rules SHA-256:     {report.sta1_rules_sha256}")
    if not report.sta1_ivtff_semantic_valid:
        print("  IVTFF semantic:    unavailable")
        print(
            "  Semantic reason:   "
            f"{report.sta1_ivtff_semantic_error}"
        )
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
        for metric in ("loci", "paragraphs", "long_words"):
            result = report.comparison[metric]
            observed = result["observed"]
            expected = result["expected"]
            status = result["status"]

            obs_text = "n/a" if observed is None else f"{observed:,}"
            exp_text = "n/a" if expected is None else f"{expected:,}"
            print(
                f"  {metric:16s} observed={obs_text:>9s} "
                f"expected={exp_text:>9s}  {status}"
            )

        sta_result = report.comparison["sta1_characters"]
        sta_obs = sta_result["observed"]
        sta_sem = sta_result.get("observed_ivtff_semantic")
        sta_exp = sta_result["expected"]
        sta_status = sta_result["status"]

        sem_text = "n/a" if sta_sem is None else f"{sta_sem:,}"
        exp_text = "n/a" if sta_exp is None else f"{sta_exp:,}"
        print(
            f"  {'sta1_characters':16s} codewise={sta_obs:>9,} "
            f"semantic={sem_text:>9s} "
            f"expected={exp_text:>9s}  "
            f"{sta_status}"
        )

        matched = sta_result.get("matched_convention")
        if matched:
            print(f"    matched convention: {matched}")

        print()
        print(
            "  STA1 conventions are reported separately; no count is "
            "silently rewritten to force a match."
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
        description="Generate a reproducible IVTFF + STA1 corpus-QC report."
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
        "--sta1-rules-dir",
        type=Path,
        default=DEFAULT_RULES_DIR,
        help=(
            "Directory containing official STA-Eva_def.bit, "
            "STA-EvaT_def.bit, and STA-v101_def.bit files "
            f"(default: {DEFAULT_RULES_DIR})"
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help=(
            "Allow IVTFF parser to skip unrecognised data lines. "
            "Not recommended for QC."
        ),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero if any published count mismatches.",
    )
    args = parser.parse_args()

    try:
        report = build_report(
            args.source,
            args.transcription_id,
            strict=not args.non_strict,
            sta1_rules_dir=args.sta1_rules_dir,
        )
    except STA1Error as exc:
        print(f"STA1 QC ERROR: {exc}", file=sys.stderr)
        return 2

    print_report(report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                asdict(report),
                indent=2,
                sort_keys=True,
            )
            + "\n",
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
