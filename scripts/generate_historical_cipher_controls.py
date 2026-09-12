#!/usr/bin/env python3
"""Generate all frozen historical cipher-control families from plaintext."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ciphers.historical_controls import (
    FAMILIES,
    ciphertext_to_text,
    generate_family,
    normalize_plaintext,
    result_summary,
    verify_round_trip,
)


DEFAULT_CONFIG = Path(
    "configs/cipher/historical_controls_v1.json"
)
DEFAULT_OUTPUT = Path(
    "data/controls/ciphers/historical_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Config not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))

    if config.get("schema_version") != "1.0":
        raise ValueError("Expected schema_version=1.0")

    if tuple(config.get("families", ())) != FAMILIES:
        raise ValueError(
            "Cipher-family order differs from frozen protocol"
        )

    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise ValueError("Frozen protocol requires exactly five seeds")

    if len(set(int(seed) for seed in seeds)) != 5:
        raise ValueError("Cipher-control seeds must be unique")

    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Config missing parameters object")

    for family in FAMILIES:
        if family not in parameters:
            raise ValueError(
                f"Config missing family parameters: {family}"
            )

    # Freeze mechanism-defining values in code as well as in the committed
    # config. Descriptive strings may change without altering the experiment.
    expected_parameters = {
        "homophonic": {
            "homophones_per_symbol": 3,
            "selection": "uniform_per_occurrence",
        },
        "verbose": {
            "codeword_length": 2,
        },
        "null_insertion": {
            "insertion_probability": 0.20,
            "null_symbol_count": 3,
        },
        "nomenclator": {
            "top_word_count": 64,
            "codeword_length": 2,
        },
        "transposition": {
            "width": 5,
        },
        "syllabic_hybrid": {
            "top_bigram_count": 64,
            "minimum_bigram_count": 2,
        },
    }

    for family, frozen in expected_parameters.items():
        observed = parameters[family]
        for key, expected_value in frozen.items():
            if observed.get(key) != expected_value:
                raise ValueError(
                    f"Frozen parameter mismatch for {family}.{key}: "
                    f"{observed.get(key)!r}; expected {expected_value!r}"
                )

    expected_normalization = {
        "unicode_normalization": "NFKC",
        "case": "lower",
        "token_definition": "contiguous_unicode_alphabetic_characters",
        "punctuation_digits": "separators",
        "preserve_nonempty_line_boundaries": True,
        "language_specific_modernization": False,
    }

    if config.get("plaintext_normalization") != expected_normalization:
        raise ValueError(
            "Plaintext normalization differs from frozen protocol"
        )

    return config


def corpus_json_lines(result) -> str:
    rows = []
    for line_index, line in enumerate(result.ciphertext, start=1):
        rows.append(
            json.dumps(
                {
                    "line_index": line_index,
                    "tokens": [list(token) for token in line],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return "\n".join(rows) + "\n"


def write_result(
    output_dir: Path,
    source_id: str,
    result,
    source_corpus,
    *,
    config_hash: str,
    source_hash: str,
) -> dict:
    run_dir = (
        output_dir
        / source_id
        / result.family
        / f"seed_{result.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    verify_round_trip(source_corpus, result)
    summary = result_summary(source_corpus, result)

    (run_dir / "corpus.jsonl").write_text(
        corpus_json_lines(result),
        encoding="utf-8",
    )
    (run_dir / "corpus.txt").write_text(
        ciphertext_to_text(result.ciphertext),
        encoding="utf-8",
    )
    (run_dir / "key.json").write_text(
        json.dumps(
            result.key,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_summary = {
        "schema_version": "1.0",
        "source_id": source_id,
        "source_sha256": source_hash,
        "config_sha256": config_hash,
        **summary,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(
            run_summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checksums = []
    for name in (
        "corpus.jsonl",
        "corpus.txt",
        "key.json",
        "summary.json",
    ):
        path = run_dir / name
        checksums.append(
            f"{sha256_file(path)}  {name}"
        )
    (run_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )

    return {
        "family": result.family,
        "seed": result.seed,
        "path": str(run_dir),
        "ciphertext_glyphs": summary["ciphertext"]["glyphs"],
        "ciphertext_tokens": summary["ciphertext"]["tokens"],
        "ciphertext_alphabet_size": (
            summary["ciphertext"]["alphabet_size"]
        ),
        "round_trip_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate monoalphabetic, homophonic, verbose, null-insertion, "
            "nomenclator-like, transposition, and syllabic/hybrid control "
            "corpora from one frozen plaintext source."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "UTF-8 plaintext source. Historical spelling should already be "
            "frozen; this script does not modernize it."
        ),
    )
    parser.add_argument(
        "--source-id",
        required=True,
        help="Stable short source label, e.g. latin_text_01",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    try:
        if not args.source.is_file():
            raise ValueError(
                f"Plaintext source not found: {args.source}"
            )

        config = load_config(args.config)
        config_hash = sha256_file(args.config)
        source_hash = sha256_file(args.source)

        text = args.source.read_text(encoding="utf-8")
        corpus = normalize_plaintext(text)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        runs = []

        print("=" * 92)
        print("HISTORICAL CIPHER CONTROL GENERATION — FROZEN V1")
        print("=" * 92)
        print(f"Source:       {args.source}")
        print(f"Source ID:    {args.source_id}")
        print(f"Source SHA:   {source_hash}")
        print(f"Config:       {args.config}")
        print(f"Config SHA:   {config_hash}")
        print(f"Seeds:        {', '.join(str(x) for x in config['seeds'])}")
        print("Voynich test: NOT ACCESSED")
        print()

        for family in config["families"]:
            parameters = config["parameters"][family]
            print(f"[{family}]")

            for seed in config["seeds"]:
                result = generate_family(
                    corpus,
                    family=family,
                    seed=int(seed),
                    parameters=parameters,
                )
                run = write_result(
                    args.output_dir,
                    args.source_id,
                    result,
                    corpus,
                    config_hash=config_hash,
                    source_hash=source_hash,
                )
                runs.append(run)

                print(
                    f"  seed={seed}  "
                    f"glyphs={run['ciphertext_glyphs']:,}  "
                    f"tokens={run['ciphertext_tokens']:,}  "
                    f"alphabet={run['ciphertext_alphabet_size']:,}  "
                    "round-trip=PASS"
                )

            print()

        manifest = {
            "schema_version": "1.0",
            "analysis": "historical cipher control corpus generation",
            "source_id": args.source_id,
            "source_path": str(args.source),
            "source_sha256": source_hash,
            "config_path": str(args.config),
            "config_sha256": config_hash,
            "families": config["families"],
            "seeds": config["seeds"],
            "plaintext_normalization": config[
                "plaintext_normalization"
            ],
            "voynich_validation_accessed": False,
            "voynich_locked_test_accessed": False,
            "runs": runs,
        }

        manifest_path = (
            args.output_dir
            / args.source_id
            / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        print(f"Saved manifest: {manifest_path}")
        print(
            "Leakage guard: no Voynich validation or locked-test file "
            "was read or required."
        )
        return 0

    except Exception as exc:
        print(
            f"HISTORICAL-CIPHER CONTROL ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
