#!/usr/bin/env python3
"""Build fresh frozen historical-language controls for VOYAGER Phase 5A.

This is a DATA-PREPARATION utility only.

It:
* accepts one new UTF-8 source text each for French, German, Italian, and Latin;
* normalizes text into deterministic whitespace-tokenized source lines;
* freezes exactly the first N usable normalized tokens per language (default 20,000);
* makes one contiguous line-preserving TRAIN/locked-TEST split (default 70/30 tokens);
* writes train.txt and test_LOCKED.txt;
* computes SHA-256, line counts, and token counts;
* writes data/controls/languages_phase5a_v1/manifest.json;
* writes a build manifest and SHA256SUMS.

It DOES NOT:
* import or run the recurrence-aware generator;
* fit an HMM or n-gram;
* compute boundary entropy;
* compute any Phase-5A outcome;
* access any Voynich file.

Prospective interpretation
--------------------------
The builder necessarily constructs and hashes the locked TEST bytes. "Untouched"
for Phase 5A means untouched by the Phase-5A outcome/scoring pipeline before the
protocol/data freeze, not that no software process has ever read those bytes.

Freeze/commit the builder outputs before running:
    python scripts/run_phase_5a_boundary_prediction.py
and certainly before:
    python scripts/run_phase_5a_boundary_prediction.py --execute-prospective-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "1.0"
FREEZE_ID = "phase5a-fresh-historical-controls-v1"

LANGUAGE_SPECS = (
    ("French", "french"),
    ("German", "german"),
    ("Italian", "italian"),
    ("Latin", "latin"),
)

FORBIDDEN_PRIOR_SOURCE_IDS = {
    "french_profiterole_ud218",
    "german_rem_v21",
    "italian_old_ud218",
    "latin_udante_ud218",
}

# Unicode-aware because Python's \w includes Unicode letters/digits. We then
# reject tokens containing digits or underscore and require every retained
# codepoint to be alphabetic, combining-mark, apostrophe, or hyphen.
ROUGH_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class SourceMetadata:
    language: str
    slug: str
    raw_path: Path
    source_id: str
    corpus: str
    version: str
    license: str
    provenance: str


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def canonicalize_token(token: str) -> str | None:
    """Return one deterministic lexical token, or None if unusable."""
    value = unicodedata.normalize("NFC", token).strip().lower()
    value = value.replace("’", "'")

    # Trim punctuation-like connectors at token edges.
    value = value.strip("'-")
    if not value:
        return None

    # No digits/underscore. Internal apostrophe/hyphen are allowed.
    if any(ch.isdigit() or ch == "_" for ch in value):
        return None

    for ch in value:
        category = unicodedata.category(ch)
        if ch in "'-":
            continue
        if ch.isalpha():
            continue
        # Retain Unicode combining marks attached to letters.
        if category.startswith("M"):
            continue
        return None

    # Require at least one alphabetic codepoint.
    if not any(ch.isalpha() for ch in value):
        return None
    return value


def tokenize_line(raw_line: str) -> Tuple[str, ...]:
    normalized = unicodedata.normalize("NFC", raw_line)
    output: List[str] = []
    for match in ROUGH_TOKEN_RE.finditer(normalized):
        token = canonicalize_token(match.group(0))
        if token is not None:
            output.append(token)
    return tuple(output)


def normalized_lines(path: Path) -> List[Tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    lines: List[Tuple[str, ...]] = []

    for raw_line in text.splitlines():
        tokens = tokenize_line(raw_line)
        if tokens:
            lines.append(tokens)

    if not lines:
        raise ValueError(f"No usable tokenized lines found in {path}")
    return lines


def token_count(lines: Sequence[Sequence[str]]) -> int:
    return sum(len(line) for line in lines)


def freeze_prefix_without_splitting_lines(
    lines: Sequence[Tuple[str, ...]],
    *,
    target_tokens: int,
) -> List[Tuple[str, ...]]:
    """Take the shortest line prefix reaching at least target_tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    selected: List[Tuple[str, ...]] = []
    total = 0
    for line in lines:
        selected.append(line)
        total += len(line)
        if total >= target_tokens:
            break

    if total < target_tokens:
        raise ValueError(
            f"Source has only {total} usable tokens; "
            f"needs at least {target_tokens}"
        )
    return selected


def contiguous_line_preserving_split(
    lines: Sequence[Tuple[str, ...]],
    *,
    train_fraction: float,
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """Choose the line boundary closest to the requested TRAIN token fraction."""
    if not (0.5 < train_fraction < 0.9):
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    if len(lines) < 30:
        raise ValueError(
            "Need at least 30 normalized lines so both TRAIN and TEST "
            "can support the Phase-5A 15-unit TEST partition."
        )

    total = token_count(lines)
    target_train = total * train_fraction

    cumulative = 0
    candidates = []
    for i, line in enumerate(lines[:-1], start=1):
        cumulative += len(line)
        train_lines = i
        test_lines = len(lines) - i

        # Phase 5A TEST requires >=15 nonempty lines. Keep TRAIN nontrivial too.
        if train_lines < 15 or test_lines < 15:
            continue

        candidates.append(
            (
                abs(cumulative - target_train),
                i,
                cumulative,
            )
        )

    if not candidates:
        raise ValueError(
            "Could not form a line-preserving split with >=15 lines "
            "in both TRAIN and TEST"
        )

    _, split_index, _ = min(candidates, key=lambda item: (item[0], item[1]))
    train = list(lines[:split_index])
    test = list(lines[split_index:])

    if not train or not test:
        raise RuntimeError("Split unexpectedly produced an empty partition")
    if len(test) < 15:
        raise RuntimeError("Locked TEST has fewer than 15 lines")
    return train, test


def serialize_lines(lines: Sequence[Sequence[str]]) -> bytes:
    """Exact format consumed by Phase-3/Phase-5 plain source readers."""
    text = "".join(" ".join(line) + "\n" for line in lines)
    return text.encode("utf-8")


def write_exact(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing frozen file without --overwrite: {path}"
        )
    path.write_bytes(data)


def load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata JSON must be an object")
    return data


def source_metadata(
    metadata: Mapping[str, object],
    *,
    language: str,
    slug: str,
    raw_path: Path,
) -> SourceMetadata:
    record = metadata.get(language)
    if not isinstance(record, dict):
        raise ValueError(
            f"Metadata JSON requires an object keyed by {language!r}"
        )

    required = ("source_id", "corpus", "version", "license", "provenance")
    missing = [
        field
        for field in required
        if not str(record.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(
            f"{language}: metadata fields missing/blank: {missing}"
        )

    source_id = str(record["source_id"]).strip()
    if source_id in FORBIDDEN_PRIOR_SOURCE_IDS:
        raise ValueError(
            f"{language}: source_id {source_id!r} is a prior Phase-3 source "
            "and is forbidden for prospective Phase 5A"
        )

    return SourceMetadata(
        language=language,
        slug=slug,
        raw_path=raw_path,
        source_id=source_id,
        corpus=str(record["corpus"]).strip(),
        version=str(record["version"]).strip(),
        license=str(record["license"]).strip(),
        provenance=str(record["provenance"]).strip(),
    )


def validate_unique_source_ids(sources: Sequence[SourceMetadata]) -> None:
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("All four fresh source_id values must be distinct")


def build_one(
    source: SourceMetadata,
    *,
    output_root: Path,
    target_tokens: int,
    train_fraction: float,
    overwrite: bool,
) -> Tuple[dict, dict]:
    raw_sha = sha256_file(source.raw_path)
    all_lines = normalized_lines(source.raw_path)

    frozen_lines = freeze_prefix_without_splitting_lines(
        all_lines,
        target_tokens=target_tokens,
    )
    train_lines, test_lines = contiguous_line_preserving_split(
        frozen_lines,
        train_fraction=train_fraction,
    )

    language_dir = output_root / source.slug
    train_path = language_dir / "train.txt"
    test_path = language_dir / "test_LOCKED.txt"

    train_bytes = serialize_lines(train_lines)
    test_bytes = serialize_lines(test_lines)

    # Compute hashes from exact bytes BEFORE write, then verify after write.
    train_sha = sha256_bytes(train_bytes)
    test_sha = sha256_bytes(test_bytes)

    write_exact(train_path, train_bytes, overwrite=overwrite)
    write_exact(test_path, test_bytes, overwrite=overwrite)

    if sha256_file(train_path) != train_sha:
        raise RuntimeError(f"{source.language}: TRAIN write/hash mismatch")
    if sha256_file(test_path) != test_sha:
        raise RuntimeError(f"{source.language}: TEST write/hash mismatch")

    train_tokens = token_count(train_lines)
    test_tokens = token_count(test_lines)

    manifest_entry = {
        "source_id": source.source_id,
        "language": source.language,
        "corpus": source.corpus,
        "version": source.version,
        "license": source.license,
        "provenance": source.provenance,
        "train_path": train_path.as_posix(),
        "train_sha256": train_sha,
        "train_lines": len(train_lines),
        "train_tokens": train_tokens,
        "test_locked_path": test_path.as_posix(),
        "test_locked_sha256": test_sha,
        "test_locked_lines": len(test_lines),
        "test_locked_tokens": test_tokens,
        "prospective_status": "UNTOUCHED_BEFORE_PHASE5A_EXECUTION",
        "selection_frozen_before_outcomes": True,
    }

    build_entry = {
        "source_id": source.source_id,
        "language": source.language,
        "raw_path": source.raw_path.as_posix(),
        "raw_sha256": raw_sha,
        "raw_normalized_usable_lines": len(all_lines),
        "raw_normalized_usable_tokens": token_count(all_lines),
        "prefix_target_tokens": target_tokens,
        "frozen_prefix_lines": len(frozen_lines),
        "frozen_prefix_tokens": token_count(frozen_lines),
        "requested_train_fraction": train_fraction,
        "realized_train_fraction_tokens": (
            train_tokens / (train_tokens + test_tokens)
        ),
        "train_lines": len(train_lines),
        "train_tokens": train_tokens,
        "train_sha256": train_sha,
        "test_locked_lines": len(test_lines),
        "test_locked_tokens": test_tokens,
        "test_locked_sha256": test_sha,
        "normalization": {
            "unicode": "NFC",
            "case": "lower",
            "token_rule": (
                "Unicode alphabetic tokens; internal apostrophe/hyphen retained; "
                "digits/underscore/nonletter symbols excluded"
            ),
            "blank_or_zero_token_lines": "discarded",
            "source_order": "preserved",
            "line_splitting": False,
        },
    }

    return manifest_entry, build_entry


def verify_output_contract(manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Manifest schema mismatch")
    if manifest.get("freeze_id") != FREEZE_ID:
        raise RuntimeError("Manifest freeze_id mismatch")
    if manifest.get("selection_frozen_before_outcomes") is not True:
        raise RuntimeError("Manifest is not marked frozen before outcomes")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise RuntimeError("Manifest must contain exactly four sources")

    languages = [row["language"] for row in sources]
    expected = [language for language, _ in LANGUAGE_SPECS]
    if sorted(languages) != sorted(expected):
        raise RuntimeError(
            f"Manifest languages differ: observed={languages}, expected={expected}"
        )

    for row in sources:
        for field in (
            "train_sha256",
            "test_locked_sha256",
        ):
            value = str(row[field])
            if len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value.lower()
            ):
                raise RuntimeError(
                    f"{row['language']}: invalid generated {field}"
                )

        if int(row["train_lines"]) <= 0 or int(row["train_tokens"]) <= 0:
            raise RuntimeError(f"{row['language']}: invalid TRAIN counts")
        if int(row["test_locked_lines"]) < 15:
            raise RuntimeError(
                f"{row['language']}: TEST must contain at least 15 lines"
            )
        if int(row["test_locked_tokens"]) <= 0:
            raise RuntimeError(f"{row['language']}: invalid TEST token count")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and freeze fresh historical-language data for "
            "VOYAGER Phase 5A. No model/outcome analysis is performed."
        )
    )
    parser.add_argument(
        "--french",
        type=Path,
        default=Path("data/raw/phase5a/french_new.txt"),
    )
    parser.add_argument(
        "--german",
        type=Path,
        default=Path("data/raw/phase5a/german_new.txt"),
    )
    parser.add_argument(
        "--italian",
        type=Path,
        default=Path("data/raw/phase5a/italian_new.txt"),
    )
    parser.add_argument(
        "--latin",
        type=Path,
        default=Path("data/raw/phase5a/latin_new.txt"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("configs/phase5a_fresh_sources_metadata.json"),
        help=(
            "JSON metadata keyed by French/German/Italian/Latin with "
            "source_id, corpus, version, license, provenance."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/controls/languages_phase5a_v1"),
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=20_000,
        help=(
            "Minimum token target for the frozen contiguous prefix of each "
            "source. A complete line may cause a small overshoot."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing build. Do not use after Phase-5A TEST "
            "execution or after outcome inspection."
        ),
    )
    args = parser.parse_args()

    try:
        if args.target_tokens < 5_000:
            raise ValueError(
                "--target-tokens must be >=5000 for a meaningful fresh control"
            )

        required_inputs = [
            args.french,
            args.german,
            args.italian,
            args.latin,
            args.metadata,
        ]
        missing_inputs = [path for path in required_inputs if not path.is_file()]
        if missing_inputs:
            expected = "\n".join(f"  {path}" for path in missing_inputs)
            raise FileNotFoundError(
                "Required Phase-5A fresh input files are missing:\n"
                f"{expected}\n\n"
                "Place four genuinely new UTF-8 corpora at the default paths "
                "and fill configs/phase5a_fresh_sources_metadata.json, or "
                "override paths with --french/--german/--italian/--latin/--metadata."
            )

        metadata = load_metadata(args.metadata)
        raw_paths = {
            "French": args.french,
            "German": args.german,
            "Italian": args.italian,
            "Latin": args.latin,
        }

        sources = [
            source_metadata(
                metadata,
                language=language,
                slug=slug,
                raw_path=raw_paths[language],
            )
            for language, slug in LANGUAGE_SPECS
        ]
        validate_unique_source_ids(sources)

        manifest_path = args.output_root / "manifest.json"
        build_manifest_path = args.output_root / "BUILD_MANIFEST.json"
        sums_path = args.output_root / "SHA256SUMS"

        if not args.overwrite:
            for protected in (
                manifest_path,
                build_manifest_path,
                sums_path,
            ):
                if protected.exists():
                    raise FileExistsError(
                        f"Existing Phase-5A build detected: {protected}. "
                        "Refusing to mutate it without --overwrite."
                    )

        manifest_entries: List[dict] = []
        build_entries: List[dict] = []

        for source in sources:
            manifest_entry, build_entry = build_one(
                source,
                output_root=args.output_root,
                target_tokens=int(args.target_tokens),
                train_fraction=float(args.train_fraction),
                overwrite=bool(args.overwrite),
            )
            manifest_entries.append(manifest_entry)
            build_entries.append(build_entry)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "freeze_id": FREEZE_ID,
            "selection_frozen_before_outcomes": True,
            "sources": manifest_entries,
        }
        verify_output_contract(manifest)

        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        build_manifest = {
            "schema_version": SCHEMA_VERSION,
            "builder": "scripts/build_phase5a_fresh_controls.py",
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "purpose": (
                "data preparation only; no generator/model/outcome analysis"
            ),
            "target_tokens": int(args.target_tokens),
            "train_fraction": float(args.train_fraction),
            "locked_test_constructed_before_phase5a_outcome_execution": True,
            "locked_test_outcome_analyzed_by_builder": False,
            "voynich_files_accessed": False,
            "sources": build_entries,
            "phase5a_manifest_sha256": sha256_file(manifest_path),
        }
        build_manifest_path.write_text(
            json.dumps(
                build_manifest,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        checksum_targets = sorted(
            path
            for path in args.output_root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        sums_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.relative_to(args.output_root)}"
                for path in checksum_targets
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 96)
        print("PHASE 5A — FRESH CONTROL BUILD")
        print("=" * 96)
        print("Outcome/model analysis: NONE")
        print("Voynich access:         NONE")
        print(f"Output root:            {args.output_root}")
        print(f"Manifest SHA-256:       {sha256_file(manifest_path)}")
        print()

        for row in manifest_entries:
            total = int(row["train_tokens"]) + int(row["test_locked_tokens"])
            print(
                f"{row['language']:<10s} "
                f"source={row['source_id']} "
                f"total={total} "
                f"TRAIN={row['train_tokens']} tokens/{row['train_lines']} lines "
                f"TEST={row['test_locked_tokens']} tokens/{row['test_locked_lines']} lines"
            )
            print(f"  TRAIN SHA {row['train_sha256']}")
            print(f"  TEST  SHA {row['test_locked_sha256']}")

        print()
        print("BUILD PASS")
        print()
        print("Next:")
        print(
            "  python scripts/run_phase_5a_boundary_prediction.py"
        )
        print(
            "This should perform PRE-FLIGHT ONLY and leave the frozen TEST "
            "files untouched by the outcome pipeline."
        )
        print()
        print(
            "Commit the builder, raw-source provenance/metadata, manifest, "
            "TRAIN, and TEST files before --execute-prospective-test."
        )
        return 0

    except Exception as exc:
        print(
            f"PHASE 5A FRESH-CONTROL BUILD ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
