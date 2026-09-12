#!/usr/bin/env python3
"""Cross-transcription agreement report for the frozen manuscript-image QC loci.

Compares the 57 loci already selected in:
    results/qc/image_validation/v1/loci.csv

across:
    ZL3b  (primary)
    GC2a
    IT2a

The native transcription alphabets are converted independently through the
project's official STA1 bitrans rules before comparison. This avoids treating
Eva- and EvaT spelling conventions as manuscript disagreement.

Purpose
-------
This is a triage tool. It reduces the manual manuscript-image audit to a small
set of high-information cases rather than asking an inexperienced reviewer to
paleographically re-transcribe every sampled line.

It does NOT:
- decide which transcription is "correct";
- use RF1b as independent corroboration (RF is derived);
- read validation.txt or test_LOCKED.txt;
- modify the frozen loci.csv or pages.csv worksheets.

Outputs
-------
results/qc/image_validation/v1/cross_transcription/
    agreement_report.csv
    recommended_manual_review.csv
    summary.json
    report.md
    run_manifest.json

Comparison levels
-----------------
1. STA1 glyph stream, ignoring token boundaries.
2. STA1 tokenized stream, treating '.' and ',' as boundaries.
3. Pairwise normalized Levenshtein similarity.
4. Exact two-of-three and three-of-three agreement.
5. Special detection of GC2a + IT2a consensus against primary ZL3b.

Manual-review policy
--------------------
Always recommend:
- a sampled locus missing/unconvertible in any corpus;
- GC2a + IT2a exact consensus against ZL3b;
- very low three-way glyph agreement;
- all-three glyph agreement but boundary disagreement (important because
  the slot-grammar model is segmentation-sensitive).

Then fill the targeted-disagreement quota with the highest remaining
disagreements, while favoring different folios. Finally add a small number of
three-way-exact controls. Mandatory cases are never dropped to satisfy a cap.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ivtff import LocusRecord, load_ivtff
from src.data.sta1 import (
    STA1Error,
    convert_native_text_to_sta1,
    iter_sta1_symbols,
    load_bitrans_rules,
    rule_file_for_alphabet,
)


DEFAULT_SAMPLE = Path("results/qc/image_validation/v1/loci.csv")
DEFAULT_CONFIG = Path("configs/qc/cross_transcription_sample_v1.json")
DEFAULT_OUTPUT = Path(
    "results/qc/image_validation/v1/cross_transcription"
)
DEFAULT_RULES_DIR = Path("data/reference/sta1")

CORPUS_PATHS = {
    "ZL3b": Path("data/raw/zl/ZL3b-n.txt"),
    "GC2a": Path("data/raw/gc/GC2a-n.txt"),
    "IT2a": Path("data/raw/it/IT2a-n.txt"),
}
CORPUS_ORDER = ("ZL3b", "GC2a", "IT2a")

EXPECTED_CORPUS_SHA256 = {
    "ZL3b": (
        "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
    ),
    "GC2a": (
        "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f"
    ),
    "IT2a": (
        "7f27a8b0feed8f6de0a99900df6bf912dd1d295c38e5f830bac8b41c3f536fb5"
    ),
}

_ALT_RE = re.compile(r"\[[^:\[\]]*:[^\[\]]*\]")
_UNKNOWN_RE = re.compile(r"\?+")


@dataclass(frozen=True)
class ComparableText:
    glyphs: Tuple[str, ...]
    tokens: Tuple[Tuple[str, ...], ...]
    glyph_count: int
    token_count: int
    unknown_glyphs: int
    uncertain_spaces: int
    alternatives: int
    canonical_glyph_text: str
    canonical_token_text: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_corpus_hash(
    corpus: str,
    path: Path,
    *,
    allow_hash_mismatch: bool,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{corpus}: missing transcription {path}")

    observed = sha256_file(path)
    expected = EXPECTED_CORPUS_SHA256[corpus]

    if observed != expected and not allow_hash_mismatch:
        raise ValueError(
            f"{corpus}: frozen raw SHA-256 mismatch.\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}\n"
            "Use the frozen raw corpus, or explicitly pass "
            "--allow-hash-mismatch for diagnostic work only."
        )

    return observed


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "schema_version",
        "low_glyph_similarity_threshold",
        "high_glyph_similarity_threshold",
        "target_disagreement_reviews",
        "consensus_control_reviews",
        "max_initial_reviews_per_folio",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(
            "Config missing fields: " + ", ".join(sorted(missing))
        )

    if data["schema_version"] != "1.0":
        raise ValueError("Unsupported config schema")

    low = float(data["low_glyph_similarity_threshold"])
    high = float(data["high_glyph_similarity_threshold"])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Require 0 <= low < high <= 1")

    if int(data["target_disagreement_reviews"]) < 1:
        raise ValueError("target_disagreement_reviews must be >= 1")
    if int(data["consensus_control_reviews"]) < 0:
        raise ValueError("consensus_control_reviews must be >= 0")
    if int(data["max_initial_reviews_per_folio"]) < 1:
        raise ValueError("max_initial_reviews_per_folio must be >= 1")

    return data


def read_sample(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen sampled loci not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No sampled loci in {path}")

    required = {"sample_index", "folio", "locus", "text_normalized"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"{path} missing columns: {', '.join(sorted(missing))}"
        )

    loci = [row["locus"] for row in rows]
    if len(loci) != len(set(loci)):
        raise ValueError("Frozen loci.csv contains duplicate locus IDs")

    return rows


def index_records(
    records: Sequence[LocusRecord],
    *,
    corpus: str,
) -> Dict[str, LocusRecord]:
    index: Dict[str, LocusRecord] = {}
    for record in records:
        if record.locus in index:
            raise ValueError(
                f"{corpus}: duplicate locus ID {record.locus!r}"
            )
        index[record.locus] = record
    return index


def load_corpus(
    corpus: str,
    path: Path,
    *,
    rules_dir: Path,
) -> Tuple[Dict[str, LocusRecord], object, dict]:
    records = load_ivtff(
        path,
        transcription_id=corpus,
        strict=True,
    )
    if not records:
        raise ValueError(f"{corpus}: parsed zero loci")

    alphabets = {record.ivtff_alphabet for record in records}
    if len(alphabets) != 1:
        raise ValueError(
            f"{corpus}: expected one IVTFF alphabet; found {alphabets}"
        )
    alphabet = next(iter(alphabets))
    if alphabet is None:
        raise ValueError(f"{corpus}: IVTFF alphabet is missing")

    rule_path = rule_file_for_alphabet(alphabet, rules_dir)
    rules = load_bitrans_rules(
        rule_path,
        expected_native_alphabet=alphabet,
    )

    return (
        index_records(records, corpus=corpus),
        rules,
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(records),
            "ivtff_alphabet": alphabet,
            "sta1_rule_path": str(rule_path),
            "sta1_rule_sha256": sha256_file(rule_path),
        },
    )


def comparable_text(
    record: LocusRecord,
    rules,
) -> ComparableText:
    """Convert one native locus to a comparable STA1 representation."""
    native = record.text_normalized

    converted = convert_native_text_to_sta1(
        native,
        rules,
        strict=True,
    )

    # Drawing interruptions are structural boundaries rather than glyphs.
    # iter_sta1_symbols normally removes angle spans completely, so convert
    # these two dedicated interruption markers into explicit boundaries first.
    converted = converted.replace("<->", ".").replace("<~>", ".")

    stream = list(
        iter_sta1_symbols(
            converted,
            first_alternative=True,
        )
    )

    glyphs: List[str] = []
    tokens: List[Tuple[str, ...]] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            tokens.append(tuple(current))
            current = []

    for symbol in stream:
        if symbol is None:
            flush()
            continue
        glyphs.append(symbol)
        current.append(symbol)
    flush()

    return ComparableText(
        glyphs=tuple(glyphs),
        tokens=tuple(tokens),
        glyph_count=len(glyphs),
        token_count=len(tokens),
        unknown_glyphs=sum(glyph == "Z1" for glyph in glyphs),
        uncertain_spaces=native.count(","),
        alternatives=len(_ALT_RE.findall(native)),
        canonical_glyph_text=" ".join(glyphs),
        canonical_token_text=".".join(
            " ".join(token)
            for token in tokens
        ),
    )


def levenshtein_distance(
    left: Sequence[object],
    right: Sequence[object],
) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    # Keep the inner row on the shorter sequence.
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))

    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (
                0 if left_item == right_item else 1
            )
            current.append(
                min(insertion, deletion, substitution)
            )
        previous = current

    return previous[-1]


def normalized_similarity(
    left: Sequence[object],
    right: Sequence[object],
) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return 1.0 - (
        levenshtein_distance(left, right) / denominator
    )


def pair_key(a: str, b: str) -> str:
    return f"{a}_{b}"


def disagreement_features(
    comparable: Mapping[str, ComparableText],
) -> dict:
    glyph_exact = {}
    token_exact = {}
    glyph_similarity = {}
    token_similarity = {}

    for i, a in enumerate(CORPUS_ORDER):
        for b in CORPUS_ORDER[i + 1 :]:
            key = pair_key(a, b)
            glyph_exact[key] = comparable[a].glyphs == comparable[b].glyphs
            token_exact[key] = comparable[a].tokens == comparable[b].tokens
            glyph_similarity[key] = normalized_similarity(
                comparable[a].glyphs,
                comparable[b].glyphs,
            )
            token_similarity[key] = normalized_similarity(
                comparable[a].tokens,
                comparable[b].tokens,
            )

    all_glyph_exact = all(glyph_exact.values())
    all_token_exact = all(token_exact.values())

    alternate_glyph_consensus_against_primary = (
        comparable["GC2a"].glyphs == comparable["IT2a"].glyphs
        and comparable["ZL3b"].glyphs != comparable["GC2a"].glyphs
    )
    alternate_token_consensus_against_primary = (
        comparable["GC2a"].tokens == comparable["IT2a"].tokens
        and comparable["ZL3b"].tokens != comparable["GC2a"].tokens
    )

    primary_exact_supporters = [
        other
        for other in ("GC2a", "IT2a")
        if comparable["ZL3b"].tokens == comparable[other].tokens
    ]

    min_glyph_similarity = min(glyph_similarity.values())
    mean_glyph_similarity = sum(glyph_similarity.values()) / 3.0

    return {
        "glyph_exact": glyph_exact,
        "token_exact": token_exact,
        "glyph_similarity": glyph_similarity,
        "token_similarity": token_similarity,
        "all_glyph_exact": all_glyph_exact,
        "all_token_exact": all_token_exact,
        "alternate_glyph_consensus_against_primary": (
            alternate_glyph_consensus_against_primary
        ),
        "alternate_token_consensus_against_primary": (
            alternate_token_consensus_against_primary
        ),
        "primary_exact_supporters": primary_exact_supporters,
        "min_glyph_similarity": min_glyph_similarity,
        "mean_glyph_similarity": mean_glyph_similarity,
    }


def classify_comparison(
    features: Mapping[str, object],
    comparable: Mapping[str, ComparableText],
    *,
    low_threshold: float,
    high_threshold: float,
) -> Tuple[str, bool, float, str]:
    """Return class, mandatory_review, priority_score, human-readable reason."""
    all_glyph_exact = bool(features["all_glyph_exact"])
    all_token_exact = bool(features["all_token_exact"])
    min_sim = float(features["min_glyph_similarity"])

    uncertainty = sum(
        item.unknown_glyphs
        + item.uncertain_spaces
        + item.alternatives
        for item in comparable.values()
    )

    if features["alternate_token_consensus_against_primary"]:
        return (
            "GC_IT_CONSENSUS_AGAINST_ZL",
            True,
            1000.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                "GC2a and IT2a agree with each other at the tokenized STA1 "
                "level but differ from primary ZL3b."
            ),
        )

    if features["alternate_glyph_consensus_against_primary"]:
        return (
            "GC_IT_GLYPH_CONSENSUS_AGAINST_ZL",
            True,
            950.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                "GC2a and IT2a agree on the STA1 glyph stream but differ "
                "from ZL3b; boundary placement may also differ."
            ),
        )

    if min_sim < low_threshold:
        return (
            "LOW_GLYPH_AGREEMENT",
            False,
            900.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                f"At least one pair has glyph similarity below "
                f"{low_threshold:.2f}."
            ),
        )

    if all_glyph_exact and not all_token_exact:
        return (
            "BOUNDARY_ONLY_DISAGREEMENT",
            True,
            850.0 + uncertainty,
            (
                "All three agree on glyphs but disagree on token boundaries; "
                "this is directly relevant to segmentation-sensitive models."
            ),
        )

    if all_token_exact:
        if uncertainty:
            return (
                "ALL_THREE_EXACT_WITH_UNCERTAINTY_MARKUP",
                False,
                40.0 + uncertainty,
                (
                    "All three canonical token streams agree; at least one "
                    "source still carries uncertainty markup."
                ),
            )
        return (
            "ALL_THREE_EXACT",
            False,
            0.0,
            "All three canonical STA1 token streams agree exactly.",
        )

    if min_sim < high_threshold:
        return (
            "MODERATE_GLYPH_AGREEMENT",
            False,
            650.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                f"Three-way glyph agreement is below the high-agreement "
                f"threshold {high_threshold:.2f}."
            ),
        )

    exact_support = len(features["primary_exact_supporters"])
    if exact_support == 1:
        return (
            "PRIMARY_SUPPORTED_BY_ONE_TRANSCRIPTION",
            False,
            350.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                "ZL3b exactly matches one comparison transcription while "
                "the third differs."
            ),
        )

    return (
        "HIGH_BUT_NONEXACT_GLYPH_AGREEMENT",
        False,
        450.0 + 100.0 * (1.0 - min_sim) + uncertainty,
        (
            "Glyph streams are highly similar but no exact three-way "
            "tokenized agreement exists."
        ),
    )


def build_report_rows(
    sample_rows: Sequence[dict],
    indexes: Mapping[str, Mapping[str, LocusRecord]],
    rules: Mapping[str, object],
    *,
    config: Mapping[str, object],
) -> List[dict]:
    output: List[dict] = []

    for sample_order, sample in enumerate(sample_rows, start=1):
        locus = sample["locus"]
        folio = sample["folio"]

        missing = [
            corpus
            for corpus in CORPUS_ORDER
            if locus not in indexes[corpus]
        ]

        base = {
            "sample_order": sample_order,
            "sample_index": sample.get("sample_index", ""),
            "locus_check_index": sample.get("locus_check_index", ""),
            "folio": folio,
            "locus": locus,
        }

        if missing:
            # Special coverage difference: ZL3b includes certain marginal
            # @Lx loci that GC2a and IT2a do not encode under a corresponding
            # locus. This is a corpus-scope difference, not evidence that the
            # ZL3b reading is wrong.
            coverage_difference = (
                locus in indexes["ZL3b"]
                and set(missing) == {"GC2a", "IT2a"}
                and ",@Lx" in locus
            )

            output.append(
                {
                    **base,
                    "comparison_class": (
                        "COVERAGE_DIFFERENCE_SPECIAL_MARGINALIA"
                        if coverage_difference
                        else "MISSING_LOCUS"
                    ),
                    "mandatory_manual_review": (
                        False if coverage_difference else True
                    ),
                    "exclude_from_manual_review": coverage_difference,
                    "priority_score": (
                        -1.0 if coverage_difference else 1200.0
                    ),
                    "review_reason": (
                        (
                            "ZL3b-only @Lx marginal locus absent from both "
                            "GC2a and IT2a; treated as a transcription-scope "
                            "coverage difference, not a disagreement."
                        )
                        if coverage_difference
                        else (
                            "Exact locus ID missing from: "
                            + ", ".join(missing)
                        )
                    ),
                    "primary_support": "UNDETERMINED",
                    "min_pairwise_glyph_similarity": "",
                    "mean_pairwise_glyph_similarity": "",
                    "all_three_glyph_exact": False,
                    "all_three_token_exact": False,
                    "gc_it_consensus_against_zl": False,
                    "zl_gc_glyph_similarity": "",
                    "zl_it_glyph_similarity": "",
                    "gc_it_glyph_similarity": "",
                    "zl_gc_token_similarity": "",
                    "zl_it_token_similarity": "",
                    "gc_it_token_similarity": "",
                    "ZL3b_native": (
                        indexes["ZL3b"][locus].text_normalized
                        if locus in indexes["ZL3b"]
                        else ""
                    ),
                    "GC2a_native": (
                        indexes["GC2a"][locus].text_normalized
                        if locus in indexes["GC2a"]
                        else ""
                    ),
                    "IT2a_native": (
                        indexes["IT2a"][locus].text_normalized
                        if locus in indexes["IT2a"]
                        else ""
                    ),
                    "ZL3b_STA1_tokens": "",
                    "GC2a_STA1_tokens": "",
                    "IT2a_STA1_tokens": "",
                }
            )
            continue

        comparable: Dict[str, ComparableText] = {}

        conversion_error = None
        for corpus in CORPUS_ORDER:
            try:
                comparable[corpus] = comparable_text(
                    indexes[corpus][locus],
                    rules[corpus],
                )
            except (STA1Error, ValueError) as exc:
                conversion_error = f"{corpus}: {exc}"
                break

        if conversion_error is not None:
            output.append(
                {
                    **base,
                    "comparison_class": "UNCOMPARABLE",
                    "mandatory_manual_review": True,
                    "priority_score": 1150.0,
                    "review_reason": conversion_error,
                    "primary_support": "UNDETERMINED",
                    "min_pairwise_glyph_similarity": "",
                    "mean_pairwise_glyph_similarity": "",
                    "all_three_glyph_exact": False,
                    "all_three_token_exact": False,
                    "gc_it_consensus_against_zl": False,
                    "zl_gc_glyph_similarity": "",
                    "zl_it_glyph_similarity": "",
                    "gc_it_glyph_similarity": "",
                    "zl_gc_token_similarity": "",
                    "zl_it_token_similarity": "",
                    "gc_it_token_similarity": "",
                    "ZL3b_native": indexes["ZL3b"][locus].text_normalized,
                    "GC2a_native": indexes["GC2a"][locus].text_normalized,
                    "IT2a_native": indexes["IT2a"][locus].text_normalized,
                    "ZL3b_STA1_tokens": "",
                    "GC2a_STA1_tokens": "",
                    "IT2a_STA1_tokens": "",
                }
            )
            continue

        features = disagreement_features(comparable)

        (
            comparison_class,
            mandatory,
            priority_score,
            review_reason,
        ) = classify_comparison(
            features,
            comparable,
            low_threshold=float(
                config["low_glyph_similarity_threshold"]
            ),
            high_threshold=float(
                config["high_glyph_similarity_threshold"]
            ),
        )

        exact_supporters = features["primary_exact_supporters"]
        if len(exact_supporters) == 2:
            primary_support = "BOTH_GC_AND_IT_EXACT"
        elif len(exact_supporters) == 1:
            primary_support = f"EXACT_{exact_supporters[0]}"
        else:
            primary_support = "NO_EXACT_TOKEN_SUPPORT"

        gs = features["glyph_similarity"]
        ts = features["token_similarity"]

        output.append(
            {
                **base,
                "comparison_class": comparison_class,
                "mandatory_manual_review": mandatory,
                "priority_score": round(priority_score, 6),
                "review_reason": review_reason,
                "primary_support": primary_support,
                "min_pairwise_glyph_similarity": round(
                    float(features["min_glyph_similarity"]),
                    6,
                ),
                "mean_pairwise_glyph_similarity": round(
                    float(features["mean_glyph_similarity"]),
                    6,
                ),
                "all_three_glyph_exact": features["all_glyph_exact"],
                "all_three_token_exact": features["all_token_exact"],
                "gc_it_consensus_against_zl": (
                    features[
                        "alternate_token_consensus_against_primary"
                    ]
                    or features[
                        "alternate_glyph_consensus_against_primary"
                    ]
                ),
                "zl_gc_glyph_similarity": round(
                    gs["ZL3b_GC2a"], 6
                ),
                "zl_it_glyph_similarity": round(
                    gs["ZL3b_IT2a"], 6
                ),
                "gc_it_glyph_similarity": round(
                    gs["GC2a_IT2a"], 6
                ),
                "zl_gc_token_similarity": round(
                    ts["ZL3b_GC2a"], 6
                ),
                "zl_it_token_similarity": round(
                    ts["ZL3b_IT2a"], 6
                ),
                "gc_it_token_similarity": round(
                    ts["GC2a_IT2a"], 6
                ),
                "ZL3b_glyph_count": comparable["ZL3b"].glyph_count,
                "GC2a_glyph_count": comparable["GC2a"].glyph_count,
                "IT2a_glyph_count": comparable["IT2a"].glyph_count,
                "ZL3b_token_count": comparable["ZL3b"].token_count,
                "GC2a_token_count": comparable["GC2a"].token_count,
                "IT2a_token_count": comparable["IT2a"].token_count,
                "ZL3b_unknown_STA1": comparable["ZL3b"].unknown_glyphs,
                "GC2a_unknown_STA1": comparable["GC2a"].unknown_glyphs,
                "IT2a_unknown_STA1": comparable["IT2a"].unknown_glyphs,
                "ZL3b_uncertain_spaces": comparable["ZL3b"].uncertain_spaces,
                "GC2a_uncertain_spaces": comparable["GC2a"].uncertain_spaces,
                "IT2a_uncertain_spaces": comparable["IT2a"].uncertain_spaces,
                "ZL3b_native": indexes["ZL3b"][locus].text_normalized,
                "GC2a_native": indexes["GC2a"][locus].text_normalized,
                "IT2a_native": indexes["IT2a"][locus].text_normalized,
                "ZL3b_STA1_tokens": comparable["ZL3b"].canonical_token_text,
                "GC2a_STA1_tokens": comparable["GC2a"].canonical_token_text,
                "IT2a_STA1_tokens": comparable["IT2a"].canonical_token_text,
            }
        )

    return output


def row_priority_key(row: Mapping[str, object]) -> Tuple[float, int]:
    return (
        -float(row["priority_score"]),
        int(row["sample_order"]),
    )


def select_recommended_review(
    rows: Sequence[dict],
    *,
    config: Mapping[str, object],
) -> List[dict]:
    target_disagreements = int(
        config["target_disagreement_reviews"]
    )
    controls_requested = int(
        config["consensus_control_reviews"]
    )
    max_per_folio = int(
        config["max_initial_reviews_per_folio"]
    )

    mandatory = sorted(
        [
            row for row in rows
            if bool(row["mandatory_manual_review"])
        ],
        key=row_priority_key,
    )

    selected: List[Tuple[dict, str]] = [
        (row, "MANDATORY_DISAGREEMENT")
        for row in mandatory
    ]
    selected_loci = {row["locus"] for row in mandatory}

    # Add highest remaining disagreements until the target is met.
    candidates = sorted(
        [
            row
            for row in rows
            if row["locus"] not in selected_loci
            and not bool(row["all_three_token_exact"])
            and not bool(row.get("exclude_from_manual_review", False))
        ],
        key=row_priority_key,
    )

    folio_counts: Dict[str, int] = {}
    for row, _ in selected:
        folio = str(row["folio"])
        folio_counts[folio] = folio_counts.get(folio, 0) + 1

    desired_total = max(target_disagreements, len(selected))

    # First pass favors different folios.
    for row in candidates:
        if len(selected) >= desired_total:
            break
        folio = str(row["folio"])
        if folio_counts.get(folio, 0) >= max_per_folio:
            continue
        selected.append((row, "TARGETED_DISAGREEMENT"))
        selected_loci.add(row["locus"])
        folio_counts[folio] = folio_counts.get(folio, 0) + 1

    # Second pass fills quota if diversity restriction prevented it.
    for row in candidates:
        if len(selected) >= desired_total:
            break
        if row["locus"] in selected_loci:
            continue
        selected.append((row, "TARGETED_DISAGREEMENT"))
        selected_loci.add(row["locus"])

    # Add exact consensus controls from folios not yet represented when possible.
    controls = sorted(
        [
            row
            for row in rows
            if bool(row["all_three_token_exact"])
            and row["locus"] not in selected_loci
        ],
        key=lambda row: (
            int(row["sample_order"]),
            str(row["locus"]),
        ),
    )

    control_added = 0
    used_folios = {str(row["folio"]) for row, _ in selected}

    for row in controls:
        if control_added >= controls_requested:
            break
        if str(row["folio"]) in used_folios:
            continue
        selected.append((row, "CONSENSUS_CONTROL"))
        selected_loci.add(row["locus"])
        used_folios.add(str(row["folio"]))
        control_added += 1

    for row in controls:
        if control_added >= controls_requested:
            break
        if row["locus"] in selected_loci:
            continue
        selected.append((row, "CONSENSUS_CONTROL"))
        selected_loci.add(row["locus"])
        control_added += 1

    result: List[dict] = []
    for rank, (row, role) in enumerate(selected, start=1):
        result.append(
            {
                "review_rank": rank,
                "review_role": role,
                **row,
            }
        )
    return result


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    rows: Sequence[dict],
    recommended: Sequence[dict],
    class_counts: Mapping[str, int],
) -> None:
    exact_three = sum(
        bool(row["all_three_token_exact"])
        for row in rows
    )
    alternate_consensus = sum(
        bool(row["gc_it_consensus_against_zl"])
        for row in rows
    )

    lines = [
        "# Cross-Transcription Agreement — Frozen Image-QC Sample",
        "",
        "This report compares the sampled loci across **ZL3b, GC2a, and IT2a** "
        "after converting each native transcription alphabet to STA1.",
        "",
        "It is a triage report, not an adjudication of which transcription is "
        "paleographically correct.",
        "",
        "## Summary",
        "",
        f"- Sampled loci: **{len(rows)}**",
        f"- Exact three-way token agreement: **{exact_three}/{len(rows)}**",
        (
            "- GC2a + IT2a consensus against primary ZL3b: "
            f"**{alternate_consensus}**"
        ),
        f"- Recommended manuscript-image checks: **{len(recommended)}**",
        "",
        "### Agreement classes",
        "",
    ]

    for key, count in sorted(
        class_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"- `{key}`: {count}")

    lines.extend(
        [
            "",
            "## Recommended Manual Checks",
            "",
            "For these cases, you are **not** being asked to re-transcribe the "
            "whole line. Locate the correct manuscript region and focus only "
            "on the disagreement described in `review_reason`. If the glyph "
            "distinction is not obvious, record it as unresolved rather than "
            "guessing.",
            "",
        ]
    )

    if not recommended:
        lines.append(
            "No disagreement cases were selected; only the automated "
            "cross-transcription report is available."
        )

    for row in recommended:
        lines.extend(
            [
                (
                    f"### {row['review_rank']:02d}. `{row['locus']}` "
                    f"— {row['review_role']}"
                ),
                "",
                f"- Folio: `{row['folio']}`",
                f"- Class: `{row['comparison_class']}`",
                f"- Reason: {row['review_reason']}",
                f"- Primary support: `{row['primary_support']}`",
                (
                    "- Minimum pairwise glyph similarity: "
                    f"`{row['min_pairwise_glyph_similarity']}`"
                ),
                "",
                "**ZL3b**",
                "",
                "```text",
                str(row["ZL3b_native"]),
                "```",
                "",
                "**GC2a**",
                "",
                "```text",
                str(row["GC2a_native"]),
                "```",
                "",
                "**IT2a**",
                "",
                "```text",
                str(row["IT2a_native"]),
                "```",
                "",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen image-validation loci across ZL3b, GC2a, "
            "and IT2a and generate a targeted image-review shortlist."
        )
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    parser.add_argument("--zl3b", type=Path, default=CORPUS_PATHS["ZL3b"])
    parser.add_argument("--gc2a", type=Path, default=CORPUS_PATHS["GC2a"])
    parser.add_argument("--it2a", type=Path, default=CORPUS_PATHS["IT2a"])
    parser.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="Diagnostic only: permit non-frozen raw transcription hashes.",
    )
    args = parser.parse_args()

    paths = {
        "ZL3b": args.zl3b,
        "GC2a": args.gc2a,
        "IT2a": args.it2a,
    }

    try:
        config = load_config(args.config)
        sample_rows = read_sample(args.sample)

        corpus_hashes = {
            corpus: require_corpus_hash(
                corpus,
                paths[corpus],
                allow_hash_mismatch=args.allow_hash_mismatch,
            )
            for corpus in CORPUS_ORDER
        }

        indexes = {}
        rules = {}
        provenance = {}

        for corpus in CORPUS_ORDER:
            index, rule_set, info = load_corpus(
                corpus,
                paths[corpus],
                rules_dir=args.rules_dir,
            )
            indexes[corpus] = index
            rules[corpus] = rule_set
            provenance[corpus] = info

        rows = build_report_rows(
            sample_rows,
            indexes,
            rules,
            config=config,
        )
        recommended = select_recommended_review(
            rows,
            config=config,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            args.output_dir / "agreement_report.csv",
            rows,
        )
        write_csv(
            args.output_dir / "recommended_manual_review.csv",
            recommended,
        )

        class_counts: Dict[str, int] = {}
        for row in rows:
            key = str(row["comparison_class"])
            class_counts[key] = class_counts.get(key, 0) + 1

        summary = {
            "schema_version": "1.0",
            "analysis": (
                "STA1-harmonized ZL3b vs GC2a vs IT2a agreement "
                "on frozen train-only manuscript-image QC loci"
            ),
            "sample_path": str(args.sample),
            "sample_sha256": sha256_file(args.sample),
            "sampled_loci": len(rows),
            "all_three_token_exact": sum(
                bool(row["all_three_token_exact"])
                for row in rows
            ),
            "all_three_glyph_exact": sum(
                bool(row["all_three_glyph_exact"])
                for row in rows
            ),
            "gc_it_consensus_against_zl": sum(
                bool(row["gc_it_consensus_against_zl"])
                for row in rows
            ),
            "mandatory_manual_review": sum(
                bool(row["mandatory_manual_review"])
                for row in rows
            ),
            "recommended_manual_review": len(recommended),
            "recommended_roles": {
                role: sum(
                    row["review_role"] == role
                    for row in recommended
                )
                for role in (
                    "MANDATORY_DISAGREEMENT",
                    "TARGETED_DISAGREEMENT",
                    "CONSENSUS_CONTROL",
                )
            },
            "class_counts": dict(sorted(class_counts.items())),
            "thresholds": config,
            "provenance": provenance,
            "raw_corpus_hashes": corpus_hashes,
            "RF1b_used_as_independent_evidence": False,
            "validation_split_accessed": False,
            "locked_test_accessed": False,
            "interpretation": (
                "Cross-transcription agreement is corroboration, not proof "
                "that any one reading is paleographically correct."
            ),
        }

        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        write_markdown(
            args.output_dir / "report.md",
            rows=rows,
            recommended=recommended,
            class_counts=class_counts,
        )

        manifest = {
            "schema_version": "1.0",
            "sample": {
                "path": str(args.sample),
                "sha256": sha256_file(args.sample),
            },
            "config": {
                "path": str(args.config),
                "sha256": sha256_file(args.config),
                "values": config,
            },
            "provenance": provenance,
            "outputs": [
                "agreement_report.csv",
                "recommended_manual_review.csv",
                "summary.json",
                "report.md",
            ],
            "validation_split_accessed": False,
            "locked_test_accessed": False,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print("=" * 82)
        print("CROSS-TRANSCRIPTION AGREEMENT — FROZEN IMAGE-QC LOCI")
        print("=" * 82)
        print(f"Sampled loci:                 {len(rows):>4}")
        print(
            f"All-three exact glyph stream: "
            f"{summary['all_three_glyph_exact']:>4}"
        )
        print(
            f"All-three exact token stream: "
            f"{summary['all_three_token_exact']:>4}"
        )
        print(
            f"GC+IT consensus against ZL:   "
            f"{summary['gc_it_consensus_against_zl']:>4}"
        )
        print(
            f"Mandatory image checks:       "
            f"{summary['mandatory_manual_review']:>4}"
        )
        print(
            f"Recommended total checks:     "
            f"{summary['recommended_manual_review']:>4}"
        )
        print("Validation split:              NOT ACCESSED")
        print("Locked test:                   NOT ACCESSED")
        print()
        print("Agreement classes:")
        for key, count in sorted(
            class_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {key:<42} {count:>3}")
        print()
        print("Recommended manual review:")
        if recommended:
            for row in recommended:
                print(
                    f"  {row['review_rank']:>2}. "
                    f"{row['locus']:<24} "
                    f"{row['review_role']:<24} "
                    f"{row['comparison_class']}"
                )
        else:
            print("  none")
        print()
        print(
            f"Saved: {args.output_dir / 'agreement_report.csv'}"
        )
        print(
            f"Saved: {args.output_dir / 'recommended_manual_review.csv'}"
        )
        print(f"Saved: {args.output_dir / 'summary.json'}")
        print(f"Saved: {args.output_dir / 'report.md'}")
        return 0

    except Exception as exc:
        print(
            f"CROSS-TRANSCRIPTION ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
