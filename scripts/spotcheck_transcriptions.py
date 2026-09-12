#!/usr/bin/env python3
"""Generate a deterministic manual cross-transcription spot-check sheet.

This is intentionally a *manual-review* artifact. The script aligns records
by IVTFF locus ID across ZL3b, GC2a, IT2a, RF1b-full, and RF1b-basic, selects
a diverse deterministic sample from loci shared by all five corpora, checks
metadata consistency automatically, and writes the source text side by side.

Default output:
    results/qc/manual_spotcheck.json
    results/qc/manual_spotcheck.md

A human should then inspect each selected locus and fill in the review
checkbox/notes in the Markdown file. The train/validation/test split should
not be frozen until this review is complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ivtff import LocusRecord, load_ivtff


DEFAULT_SEED = 40814041438
DEFAULT_SAMPLE_SIZE = 15

CORPUS_PATHS = {
    "ZL3b": Path("data/raw/zl/ZL3b-n.txt"),
    "GC2a": Path("data/raw/gc/GC2a-n.txt"),
    "IT2a": Path("data/raw/it/IT2a-n.txt"),
    "RF1b-full": Path("data/raw/rf/RF1b-e.txt"),
    "RF1b-basic": Path("data/raw/rf/RF1b-er.txt"),
}

CORPUS_ORDER = tuple(CORPUS_PATHS)


@dataclass(frozen=True)
class SpotcheckEntry:
    selection_rank: int
    locus: str
    folio: str
    section: Optional[str]
    scribe: Optional[str]
    currier: Optional[str]
    locus_type: str
    metadata_consistent: bool
    metadata_differences: Tuple[str, ...]
    records: Mapping[str, Mapping[str, object]]


def _index(records: Sequence[LocusRecord], corpus: str) -> Dict[str, LocusRecord]:
    index: Dict[str, LocusRecord] = {}
    duplicates: List[str] = []

    for record in records:
        if record.locus in index:
            duplicates.append(record.locus)
        index[record.locus] = record

    if duplicates:
        raise ValueError(
            f"{corpus}: duplicate locus IDs encountered: "
            + ", ".join(sorted(set(duplicates))[:10])
        )

    return index


def _metadata_differences(
    records: Mapping[str, LocusRecord],
) -> Tuple[str, ...]:
    """Compare alignment metadata while allowing unavailable values."""
    fields = (
        "folio",
        "line",
        "locus_type",
        "physical_leaf",
        "recto_verso",
        "foldout_panel",
        "section",
        "scribe",
        "currier",
    )

    differences: List[str] = []

    for field in fields:
        values = {
            corpus: getattr(record, field)
            for corpus, record in records.items()
        }

        present = {
            value
            for value in values.values()
            if value is not None
        }

        if len(present) > 1:
            rendered = ", ".join(
                f"{corpus}={values[corpus]!r}"
                for corpus in CORPUS_ORDER
            )
            differences.append(f"{field}: {rendered}")

    return tuple(differences)


def _coverage_signature(record: LocusRecord) -> Tuple[str, str, str, str]:
    return (
        record.section or "UNSET",
        record.scribe or "UNSET",
        record.currier or "UNSET",
        record.locus_type,
    )


def select_diverse_loci(
    zl_index: Mapping[str, LocusRecord],
    shared_loci: Sequence[str],
    *,
    sample_size: int,
    seed: int,
) -> List[str]:
    """Greedily maximize metadata coverage with deterministic tie-breaking."""
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    if sample_size > len(shared_loci):
        raise ValueError(
            f"Requested {sample_size} loci but only "
            f"{len(shared_loci)} are shared by all corpora"
        )

    rng = random.Random(seed)
    candidates = list(shared_loci)
    rng.shuffle(candidates)

    covered_sections = set()
    covered_scribes = set()
    covered_curriers = set()
    covered_types = set()

    selected: List[str] = []

    while len(selected) < sample_size:
        best_locus = None
        best_score = None

        for position, locus in enumerate(candidates):
            if locus in selected:
                continue

            record = zl_index[locus]
            section, scribe, currier, locus_type = _coverage_signature(record)

            score = (
                8 * (section not in covered_sections)
                + 5 * (scribe not in covered_scribes)
                + 3 * (currier not in covered_curriers)
                + 2 * (locus_type not in covered_types)
                - position / max(1, len(candidates))
            )

            if best_score is None or score > best_score:
                best_score = score
                best_locus = locus

        assert best_locus is not None
        selected.append(best_locus)

        record = zl_index[best_locus]
        section, scribe, currier, locus_type = _coverage_signature(record)
        covered_sections.add(section)
        covered_scribes.add(scribe)
        covered_curriers.add(currier)
        covered_types.add(locus_type)

    return selected


def build_spotcheck(
    *,
    paths: Mapping[str, Path],
    sample_size: int,
    seed: int,
) -> Dict[str, object]:
    indexes: Dict[str, Dict[str, LocusRecord]] = {}

    for corpus in CORPUS_ORDER:
        path = paths[corpus]
        if not path.is_file():
            raise FileNotFoundError(f"{corpus}: missing transcription {path}")

        records = load_ivtff(
            path,
            transcription_id=corpus,
            strict=True,
        )
        indexes[corpus] = _index(records, corpus)

    shared = set(indexes[CORPUS_ORDER[0]])
    for corpus in CORPUS_ORDER[1:]:
        shared &= set(indexes[corpus])

    shared_loci = sorted(shared)
    selected = select_diverse_loci(
        indexes["ZL3b"],
        shared_loci,
        sample_size=sample_size,
        seed=seed,
    )

    entries: List[SpotcheckEntry] = []

    for rank, locus in enumerate(selected, start=1):
        aligned = {
            corpus: indexes[corpus][locus]
            for corpus in CORPUS_ORDER
        }

        differences = _metadata_differences(aligned)
        zl = aligned["ZL3b"]

        records_payload = {
            corpus: {
                "transcription_id": record.transcription_id,
                "source_line_number": record.source_line_number,
                "source_line_numbers": list(record.source_line_numbers),
                "folio": record.folio,
                "locus": record.locus,
                "section": record.section,
                "scribe": record.scribe,
                "currier": record.currier,
                "locus_type": record.locus_type,
                "text_raw": record.text_raw,
                "text_normalized": record.text_normalized,
                "original": record.original,
            }
            for corpus, record in aligned.items()
        }

        entries.append(
            SpotcheckEntry(
                selection_rank=rank,
                locus=locus,
                folio=zl.folio,
                section=zl.section,
                scribe=zl.scribe,
                currier=zl.currier,
                locus_type=zl.locus_type,
                metadata_consistent=not differences,
                metadata_differences=differences,
                records=records_payload,
            )
        )

    section_counts = Counter(
        entry.section or "UNSET"
        for entry in entries
    )
    scribe_counts = Counter(
        entry.scribe or "UNSET"
        for entry in entries
    )
    currier_counts = Counter(
        entry.currier or "UNSET"
        for entry in entries
    )

    return {
        "schema_version": "1.0",
        "checkpoint": "manual_cross_transcription_spotcheck",
        "seed": seed,
        "sample_size": sample_size,
        "shared_loci_total": len(shared_loci),
        "selection_method": (
            "deterministic greedy metadata-coverage sample from locus IDs "
            "present in all five transcriptions"
        ),
        "corpora": list(CORPUS_ORDER),
        "coverage": {
            "sections": dict(sorted(section_counts.items())),
            "scribes": dict(sorted(scribe_counts.items())),
            "currier": dict(sorted(currier_counts.items())),
        },
        "automatic_metadata_check": {
            "all_selected_loci_consistent": all(
                entry.metadata_consistent
                for entry in entries
            ),
            "entries_with_differences": sum(
                not entry.metadata_consistent
                for entry in entries
            ),
        },
        "manual_review": {
            "status": "PENDING",
            "reviewer_instruction": (
                "Inspect all selected loci side by side. Confirm that each "
                "record refers to the same manuscript locus and that source "
                "text/provenance have not been shifted or misaligned. Record "
                "any discrepancy in the Markdown review sheet."
            ),
        },
        "entries": [asdict(entry) for entry in entries],
    }


def write_markdown(
    payload: Mapping[str, object],
    path: Path,
) -> None:
    lines: List[str] = [
        "# Manual Cross-Transcription Spot-Check",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Sample size: `{payload['sample_size']}`",
        f"- Shared loci available: `{payload['shared_loci_total']}`",
        "- Review status: **PENDING**",
        "",
        "For each locus, verify that all five records refer to the same "
        "manuscript location and that provenance/text alignment is plausible. "
        "Do not judge whether the transcriptions *agree* glyph-for-glyph; "
        "transcription disagreement is expected evidence.",
        "",
    ]

    entries = payload["entries"]

    for entry in entries:
        lines.extend(
            [
                f"## {entry['selection_rank']:02d}. `{entry['locus']}`",
                "",
                f"- Folio: `{entry['folio']}`",
                f"- Section: `{entry['section']}`",
                f"- Scribe: `{entry['scribe']}`",
                f"- Currier: `{entry['currier']}`",
                f"- Locus type: `{entry['locus_type']}`",
                (
                    "- Automatic metadata consistency: "
                    f"**{'PASS' if entry['metadata_consistent'] else 'REVIEW'}**"
                ),
            ]
        )

        if entry["metadata_differences"]:
            lines.append("- Automatic differences:")
            for difference in entry["metadata_differences"]:
                lines.append(f"  - `{difference}`")

        lines.extend(["", "### Transcriptions", ""])

        for corpus in CORPUS_ORDER:
            record = entry["records"][corpus]
            lines.extend(
                [
                    f"**{corpus}** — source line "
                    f"`{record['source_line_number']}`",
                    "",
                    "```text",
                    str(record["text_normalized"]),
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "- [ ] Same manuscript locus/provenance confirmed",
                "- [ ] No apparent one-line / one-locus alignment shift",
                "- [ ] Differences look like transcription differences, "
                "not parser corruption",
                "- Reviewer notes:",
                "",
                "---",
                "",
            ]
        )

    lines.extend(
        [
            "## Completion",
            "",
            "- [ ] All selected loci reviewed",
            "- [ ] Any discrepancies investigated and resolved",
            "- [ ] `transcription_summary.json` updated to mark manual "
            "spot-check complete before split generation",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic manual alignment spot-check across "
            "the five Voynich transcriptions."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("results/qc/manual_spotcheck.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("results/qc/manual_spotcheck.md"),
    )

    for corpus, default_path in CORPUS_PATHS.items():
        option = "--" + corpus.lower().replace("-", "_")
        parser.add_argument(
            option,
            type=Path,
            default=default_path,
        )

    args = parser.parse_args()

    paths = {
        "ZL3b": args.zl3b,
        "GC2a": args.gc2a,
        "IT2a": args.it2a,
        "RF1b-full": args.rf1b_full,
        "RF1b-basic": args.rf1b_basic,
    }

    try:
        payload = build_spotcheck(
            paths=paths,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"SPOT-CHECK ERROR: {exc}", file=sys.stderr)
        return 1

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(payload, args.markdown)

    auto = payload["automatic_metadata_check"]
    print(
        f"Shared loci across all five corpora: "
        f"{payload['shared_loci_total']:,}"
    )
    print(f"Selected for manual review: {payload['sample_size']}")
    print(
        "Automatic metadata differences: "
        f"{auto['entries_with_differences']}"
    )
    print(f"Saved JSON: {args.json}")
    print(f"Saved review sheet: {args.markdown}")
    print(
        "\nManual review is still required; this script does not mark "
        "the corpus layer frozen."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
