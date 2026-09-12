#!/usr/bin/env python3
"""Generate a deterministic TRAIN-ONLY manuscript-image validation sample.

The goal is manual QC against Yale Beinecke MS 408 images before additional
model-development decisions. This script never reads validation.txt or
test_LOCKED.txt.

Outputs
-------
results/qc/image_validation/v1/pages.csv
    One row per sampled manuscript page, including metadata/uncertainty
    diagnostics and blank reviewer fields.

results/qc/image_validation/v1/loci.csv
    Representative loci from each sampled page (first/middle/last plus the
    most markup-rich locus when distinct), with the parsed transcription to
    compare visually against the manuscript image.

results/qc/image_validation/v1/README.md
    Manual review protocol.

results/qc/image_validation/v1/sample_manifest.json
    Frozen provenance, selection rules, and selected page IDs.

The sample is selected only from the frozen TRAIN physical leaves to protect
the locked test from visual inspection.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import (
    parse_train_leaf_ids,
    select_train_records,
    sha256_file,
)
from src.data.ivtff import LocusRecord, load_ivtff


EXPECTED_ZL3B_SHA256 = (
    "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
)
EXPECTED_TRAIN_SHA256 = (
    "5bb5232d5e211a4bb90800f259294c554c1c9e2005548055ebaf94e782e3cc00"
)

DEFAULT_SOURCE = Path("data/raw/zl/ZL3b-n.txt")
DEFAULT_TRAIN = Path("data/splits/v1/train.txt")
DEFAULT_CONFIG = Path("configs/qc/image_validation_v1.json")
DEFAULT_OUTPUT = Path("results/qc/image_validation/v1")

YALE_SOURCE_URL = (
    "https://beinecke.library.yale.edu/beinecke/collections/"
    "beinecke-cipher-voynich-manuscript"
)

_ALT_RE = re.compile(r"\[[^\]]+:[^\]]+\]")
_DRAWING_RE = re.compile(r"<(?:-|~)>")
_UNKNOWN_RE = re.compile(r"\?+")
_HIGH_ASCII_RE = re.compile(r"@\d+;")


@dataclass(frozen=True)
class PageCandidate:
    folio: str
    leaf_group: str
    physical_leaf: str
    recto_verso: str
    curriers: Tuple[str, ...]
    scribes: Tuple[str, ...]
    sections: Tuple[str, ...]
    locus_types: Tuple[str, ...]
    locus_count: int
    uncertain_spaces: int
    alternative_readings: int
    unknown_question_marks: int
    drawing_interruptions: int
    high_ascii_codes: int
    markup_richness: int

    def feature_set(self) -> Set[str]:
        features = set()
        features.update(f"currier:{x}" for x in self.curriers)
        features.update(f"scribe:{x}" for x in self.scribes)
        features.update(f"section:{x}" for x in self.sections)
        features.update(f"locus_type:{x}" for x in self.locus_types)

        if self.uncertain_spaces:
            features.add("markup:uncertain_space")
        if self.alternative_readings:
            features.add("markup:alternative_reading")
        if self.unknown_question_marks:
            features.add("markup:unknown_glyph")
        if self.drawing_interruptions:
            features.add("markup:drawing_interruption")
        if self.high_ascii_codes:
            features.add("markup:high_ascii")

        if self.markup_richness == 0:
            features.add("profile:ordinary_no_markup")
        else:
            features.add("profile:markup_present")

        if any(char.isdigit() for char in self.folio[2:]):
            features.add("page:foldout_or_panel")

        return features


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch.\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}\n"
            "Refusing to sample from a non-frozen input."
        )
    return observed


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported image-validation config schema")
    if int(data.get("sample_size", -1)) != 15:
        raise ValueError("image-validation sample_size must be exactly 15")
    if int(data.get("seed", -1)) != 40814041438:
        raise ValueError("Unexpected image-validation seed")
    if int(data.get("max_pages_per_leaf_group", -1)) != 1:
        raise ValueError("max_pages_per_leaf_group must be exactly 1")
    if int(data.get("loci_per_page", -1)) != 4:
        raise ValueError("loci_per_page must be exactly 4")

    return data


def atomic_leaf_group(record: LocusRecord) -> str:
    if record.folio == "fRos" or record.physical_leaf in (85, 86):
        return "f85+f86"
    if record.physical_leaf is None:
        raise ValueError(
            f"No train leaf group for {record.folio} {record.locus}"
        )
    return f"f{record.physical_leaf}"


def markup_diagnostics(text: str) -> dict:
    unknown_runs = _UNKNOWN_RE.findall(text)
    return {
        "uncertain_spaces": text.count(","),
        "alternative_readings": len(_ALT_RE.findall(text)),
        "unknown_question_marks": sum(len(x) for x in unknown_runs),
        "drawing_interruptions": len(_DRAWING_RE.findall(text)),
        "high_ascii_codes": len(_HIGH_ASCII_RE.findall(text)),
    }


def build_page_candidates(
    records: Sequence[LocusRecord],
) -> Tuple[List[PageCandidate], Dict[str, List[LocusRecord]]]:
    by_page: Dict[str, List[LocusRecord]] = defaultdict(list)

    for record in records:
        by_page[record.folio].append(record)

    candidates = []

    for folio, page_records in sorted(by_page.items()):
        groups = {atomic_leaf_group(r) for r in page_records}
        if len(groups) != 1:
            raise ValueError(
                f"Page {folio} unexpectedly maps to multiple leaf groups: "
                f"{sorted(groups)}"
            )
        leaf_group = next(iter(groups))

        diag = Counter()
        for record in page_records:
            diag.update(markup_diagnostics(record.text_normalized))

        markup_richness = (
            diag["uncertain_spaces"]
            + 2 * diag["alternative_readings"]
            + 2 * diag["unknown_question_marks"]
            + diag["drawing_interruptions"]
        )

        physical_values = sorted(
            {
                str(r.physical_leaf)
                for r in page_records
                if r.physical_leaf is not None
            }
        )
        if folio == "fRos":
            physical_text = "85+86"
        else:
            physical_text = ",".join(physical_values)

        rv_values = sorted(
            {
                str(r.recto_verso)
                for r in page_records
                if r.recto_verso
            }
        )

        candidates.append(
            PageCandidate(
                folio=folio,
                leaf_group=leaf_group,
                physical_leaf=physical_text,
                recto_verso="|".join(rv_values),
                curriers=tuple(
                    sorted({r.currier or "UNSET" for r in page_records})
                ),
                scribes=tuple(
                    sorted({r.scribe or "UNSET" for r in page_records})
                ),
                sections=tuple(
                    sorted({r.section or "UNSET" for r in page_records})
                ),
                locus_types=tuple(
                    sorted({r.locus_type or "UNSET" for r in page_records})
                ),
                locus_count=len(page_records),
                uncertain_spaces=diag["uncertain_spaces"],
                alternative_readings=diag["alternative_readings"],
                unknown_question_marks=diag["unknown_question_marks"],
                drawing_interruptions=diag["drawing_interruptions"],
                high_ascii_codes=diag["high_ascii_codes"],
                markup_richness=markup_richness,
            )
        )

    return candidates, by_page


def feature_weights(candidates: Sequence[PageCandidate]) -> Dict[str, float]:
    """Weight rare representativeness features more strongly."""
    counts = Counter()
    for candidate in candidates:
        counts.update(candidate.feature_set())

    weights = {}
    for feature, count in counts.items():
        if feature.startswith("markup:"):
            weights[feature] = 4.0 / math.sqrt(count)
        elif feature.startswith("page:"):
            weights[feature] = 4.0 / math.sqrt(count)
        elif feature.startswith("profile:"):
            weights[feature] = 2.0 / math.sqrt(count)
        elif feature.startswith("scribe:"):
            weights[feature] = 3.0 / math.sqrt(count)
        elif feature.startswith("section:"):
            weights[feature] = 3.0 / math.sqrt(count)
        elif feature.startswith("currier:"):
            weights[feature] = 2.5 / math.sqrt(count)
        else:
            weights[feature] = 1.0 / math.sqrt(count)

    return weights


def select_representative_pages(
    candidates: Sequence[PageCandidate],
    *,
    sample_size: int,
    seed: int,
    max_pages_per_leaf_group: int,
) -> List[PageCandidate]:
    if sample_size > len(candidates):
        raise ValueError(
            f"Requested {sample_size} pages but only "
            f"{len(candidates)} train pages are available"
        )

    rng = random.Random(seed)
    tie_break = {
        candidate.folio: rng.random()
        for candidate in candidates
    }

    weights = feature_weights(candidates)
    selected: List[PageCandidate] = []
    covered: Set[str] = set()
    leaf_counts = Counter()

    remaining = list(candidates)

    while len(selected) < sample_size:
        eligible = [
            c
            for c in remaining
            if leaf_counts[c.leaf_group] < max_pages_per_leaf_group
        ]
        if not eligible:
            raise ValueError(
                "Unable to fill sample under max_pages_per_leaf_group"
            )

        scored = []
        for candidate in eligible:
            features = candidate.feature_set()
            novelty = sum(
                weights[feature]
                for feature in features
                if feature not in covered
            )

            # Once broad coverage is achieved, favor informative markup while
            # still allowing ordinary pages via the feature system.
            markup_bonus = math.log1p(candidate.markup_richness) * 0.08
            locus_bonus = math.log1p(candidate.locus_count) * 0.02

            score = novelty + markup_bonus + locus_bonus

            scored.append(
                (
                    score,
                    candidate.markup_richness,
                    candidate.locus_count,
                    tie_break[candidate.folio],
                    candidate,
                )
            )

        _, _, _, _, chosen = max(scored, key=lambda x: x[:-1])

        selected.append(chosen)
        covered.update(chosen.feature_set())
        leaf_counts[chosen.leaf_group] += 1
        remaining.remove(chosen)

    return selected


def locus_markup_score(record: LocusRecord) -> int:
    diag = markup_diagnostics(record.text_normalized)
    return (
        diag["uncertain_spaces"]
        + 2 * diag["alternative_readings"]
        + 2 * diag["unknown_question_marks"]
        + diag["drawing_interruptions"]
    )


def representative_loci(
    records: Sequence[LocusRecord],
    *,
    limit: int,
) -> List[LocusRecord]:
    ordered = sorted(
        records,
        key=lambda r: (
            r.line,
            r.source_line_number,
            r.locus,
        ),
    )
    if not ordered:
        return []

    indices = {0, len(ordered) // 2, len(ordered) - 1}
    richest_index = max(
        range(len(ordered)),
        key=lambda i: (
            locus_markup_score(ordered[i]),
            -i,
        ),
    )
    indices.add(richest_index)

    # If duplicate first/middle/last/richest choices produce fewer than
    # requested loci, fill deterministically with evenly distributed loci.
    if len(indices) < min(limit, len(ordered)):
        if len(ordered) == 1:
            candidate_indices = [0]
        else:
            candidate_indices = [
                round(i * (len(ordered) - 1) / (limit - 1))
                for i in range(limit)
            ]
        for index in candidate_indices:
            indices.add(index)
            if len(indices) >= min(limit, len(ordered)):
                break

    chosen = sorted(indices)[:limit]
    return [ordered[i] for i in chosen]


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a frozen train-only sample for manual validation "
            "against Yale Beinecke MS 408 manuscript images."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        source_hash = require_hash(
            args.source,
            EXPECTED_ZL3B_SHA256,
            "ZL3b source",
        )
        train_hash = require_hash(
            args.train,
            EXPECTED_TRAIN_SHA256,
            "Frozen train split",
        )

        if not args.config.is_file():
            raise ValueError(f"Config not found: {args.config}")
        config_hash = sha256_file(args.config)
        config = load_config(args.config)

        train_leaves = parse_train_leaf_ids(
            args.train.read_text(encoding="utf-8")
        )
        if len(train_leaves) != 72:
            raise ValueError(
                f"Expected 72 frozen train leaves; found {len(train_leaves)}"
            )

        all_records = load_ivtff(
            args.source,
            transcription_id="ZL3b",
            strict=True,
        )
        train_records = select_train_records(
            all_records,
            train_leaves,
        )

        candidates, by_page = build_page_candidates(train_records)
        selected = select_representative_pages(
            candidates,
            sample_size=config["sample_size"],
            seed=config["seed"],
            max_pages_per_leaf_group=config["max_pages_per_leaf_group"],
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        page_rows = []
        locus_rows = []

        for sample_index, candidate in enumerate(selected, start=1):
            page_rows.append(
                {
                    "sample_index": sample_index,
                    "folio": candidate.folio,
                    "leaf_group": candidate.leaf_group,
                    "physical_leaf": candidate.physical_leaf,
                    "recto_verso": candidate.recto_verso,
                    "currier": "|".join(candidate.curriers),
                    "scribe": "|".join(candidate.scribes),
                    "section": "|".join(candidate.sections),
                    "locus_types": "|".join(candidate.locus_types),
                    "locus_count": candidate.locus_count,
                    "uncertain_spaces": candidate.uncertain_spaces,
                    "alternative_readings": candidate.alternative_readings,
                    "unknown_question_marks": candidate.unknown_question_marks,
                    "drawing_interruptions": candidate.drawing_interruptions,
                    "high_ascii_codes": candidate.high_ascii_codes,
                    "yale_source_url": YALE_SOURCE_URL,
                    "review_status": "",
                    "image_identifier_or_url": "",
                    "page_alignment": "",
                    "overall_transcription_agreement": "",
                    "reviewer_notes": "",
                }
            )

            for locus_index, record in enumerate(
                representative_loci(
                    by_page[candidate.folio],
                    limit=config["loci_per_page"],
                ),
                start=1,
            ):
                diag = markup_diagnostics(record.text_normalized)
                locus_rows.append(
                    {
                        "sample_index": sample_index,
                        "folio": candidate.folio,
                        "locus_check_index": locus_index,
                        "locus": record.locus,
                        "source_line_number": record.source_line_number,
                        "text_normalized": record.text_normalized,
                        "uncertain_spaces": diag["uncertain_spaces"],
                        "alternative_readings": diag["alternative_readings"],
                        "unknown_question_marks": diag["unknown_question_marks"],
                        "drawing_interruptions": diag["drawing_interruptions"],
                        "check_status": "",
                        "glyph_alignment": "",
                        "spacing_alignment": "",
                        "locus_alignment": "",
                        "notes": "",
                    }
                )

        pages_path = args.output_dir / "pages.csv"
        loci_path = args.output_dir / "loci.csv"

        write_csv(pages_path, page_rows)
        write_csv(loci_path, locus_rows)

        selected_features = sorted(
            set().union(*(page.feature_set() for page in selected))
        )

        manifest = {
            "schema_version": "1.0",
            "purpose": (
                "manual manuscript-image validation before additional "
                "model-development decisions"
            ),
            "scope": "TRAIN ONLY",
            "locked_test_accessed": False,
            "validation_split_accessed": False,
            "sample_size_pages": len(selected),
            "locus_checks": len(locus_rows),
            "seed": config["seed"],
            "selection": {
                "algorithm": (
                    "deterministic greedy feature coverage with seeded "
                    "tie-breaking"
                ),
                "max_pages_per_leaf_group": (
                    config["max_pages_per_leaf_group"]
                ),
                "coverage_dimensions": [
                    "Currier",
                    "scribe",
                    "section",
                    "locus type",
                    "uncertain spaces",
                    "alternative readings",
                    "unknown glyph marks",
                    "drawing interruptions",
                    "ordinary/no-markup pages",
                    "foldout/panel-like folio identifiers",
                ],
                "selected_features": selected_features,
            },
            "primary_image_source": {
                "repository": (
                    "Beinecke Rare Book and Manuscript Library, "
                    "Yale University"
                ),
                "shelfmark": "Beinecke MS 408",
                "url": YALE_SOURCE_URL,
            },
            "provenance": {
                "source_path": str(args.source),
                "source_sha256": source_hash,
                "train_split_path": str(args.train),
                "train_split_sha256": train_hash,
                "config_path": str(args.config),
                "config_sha256": config_hash,
            },
            "selected_pages": [
                {
                    "sample_index": i,
                    "folio": page.folio,
                    "leaf_group": page.leaf_group,
                }
                for i, page in enumerate(selected, start=1)
            ],
        }

        (args.output_dir / "sample_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        readme = f"""# Manual manuscript-image validation — v1

## Scope

This checkpoint validates a deterministic **TRAIN-ONLY** sample of ZL3b
transcriptions against high-resolution images of Yale Beinecke MS 408.

It does **not** inspect the frozen validation membership or `test_LOCKED.txt`.

Primary image source:
{YALE_SOURCE_URL}

## Review procedure

For every row in `pages.csv`:

1. Open the corresponding manuscript folio in Yale's MS 408 image viewer.
2. Confirm that the sampled transcription page corresponds to the correct
   manuscript page.
3. Review all associated rows in `loci.csv`.
4. Compare the parsed transcription with the visible manuscript text.
5. Distinguish ordinary transcription ambiguity from parser/alignment failure.

### `pages.csv` allowed `review_status`

- `PASS`
- `TRANSCRIPTION_DISAGREEMENT`
- `PARSER_OR_ALIGNMENT_ISSUE`
- `UNRESOLVED`

### `loci.csv` allowed `check_status`

- `PASS`
- `TRANSCRIPTION_DISAGREEMENT`
- `PARSER_OR_ALIGNMENT_ISSUE`
- `UNRESOLVED`

Suggested alignment fields use `YES`, `NO`, or `UNCERTAIN`.

## Interpretation

A `TRANSCRIPTION_DISAGREEMENT` does not automatically invalidate the corpus.
It means the image plausibly supports a different glyph/space reading and
should be documented as transcription uncertainty.

A `PARSER_OR_ALIGNMENT_ISSUE` is more serious: it indicates the analytical
record may have been assigned to the wrong locus/page or transformed
incorrectly. Resolve such issues before additional model-development choices.

Do not alter the sampled page list after looking at manuscript images.
"""
        (args.output_dir / "README.md").write_text(
            readme,
            encoding="utf-8",
        )

        print("=" * 78)
        print("MANUSCRIPT-IMAGE VALIDATION SAMPLE — TRAIN ONLY")
        print("=" * 78)
        print(f"Seed:           {config['seed']}")
        print(f"Train leaves:   {len(train_leaves)}")
        print(f"Candidate pages:{len(candidates):>5}")
        print(f"Sampled pages:  {len(selected):>5}")
        print(f"Locus checks:   {len(locus_rows):>5}")
        print("Validation:     NOT ACCESSED")
        print("Locked test:    NOT ACCESSED")
        print()
        for i, page in enumerate(selected, start=1):
            print(
                f"{i:2d}. {page.folio:<8} "
                f"{page.leaf_group:<8} "
                f"Currier={'/'.join(page.curriers):<8} "
                f"scribe={'/'.join(page.scribes):<8} "
                f"section={'/'.join(page.sections)}"
            )
        print()
        print(f"Saved: {pages_path}")
        print(f"Saved: {loci_path}")
        print(f"Saved: {args.output_dir / 'README.md'}")
        print(f"Saved: {args.output_dir / 'sample_manifest.json'}")
        print()
        print(
            "Freeze these files before manually viewing the sampled pages."
        )
        return 0

    except Exception as exc:
        print(f"IMAGE-VALIDATION SAMPLE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
