#!/usr/bin/env python3
"""Fit unigram->5-gram baselines on train and evaluate on validation only."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import (
    make_analytical_loci,
    sha256_file,
)
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.ngram import (
    VALID_VIEWS,
    WittenBellNGram,
    build_baseline_sequences,
    evaluate_model,
    parse_leaf_ids,
    select_records_by_leaves,
)


EXPECTED_ZL3B_SHA256 = (
    "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
)
EXPECTED_TRAIN_SHA256 = (
    "5bb5232d5e211a4bb90800f259294c554c1c9e2005548055ebaf94e782e3cc00"
)
EXPECTED_VALIDATION_SHA256 = (
    "bdae856ac8c52849a0abd0fd6361f42fd57579e76f7f844309a7ddc9e0fc02f1"
)
EXPECTED_RULES_SHA256 = (
    "7f37853510144fb3e2dc3ee9458d634f41e6d95bc1fbf1c4b8f479a53a021f81"
)

DEFAULT_SOURCE = Path("data/raw/zl/ZL3b-n.txt")
DEFAULT_TRAIN = Path("data/splits/v1/train.txt")
DEFAULT_VALIDATION = Path("data/splits/v1/validation.txt")
DEFAULT_RULES = Path("data/reference/sta1/STA-Eva_def.bit")
DEFAULT_CONFIG = Path("configs/baselines/ngram_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/ngram/v1")


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch.\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}\n"
            "Refusing to run against a non-frozen input."
        )
    return observed


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported n-gram config schema")
    if data.get("smoothing") != "interpolated_witten_bell":
        raise ValueError(
            "Baseline smoothing is frozen to interpolated_witten_bell"
        )

    orders = data.get("orders")
    if orders != [1, 2, 3, 4, 5]:
        raise ValueError("Baseline orders must be exactly [1,2,3,4,5]")

    views = data.get("views")
    if views != ["space_free", "token_aware"]:
        raise ValueError(
            "Baseline views must be exactly space_free and token_aware"
        )

    if data.get("unknown_policy") != "reset_and_do_not_score":
        raise ValueError("Unexpected unknown policy")

    return data


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    *,
    aggregate_rows: Sequence[dict],
    leaf_rows: Sequence[dict],
    locus_rows: Sequence[dict],
    model_rows: Sequence[dict],
    provenance: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "validation_results.csv", aggregate_rows)
    write_csv(output_dir / "per_leaf_validation.csv", leaf_rows)
    write_csv(output_dir / "per_locus_validation.csv", locus_rows)
    write_csv(output_dir / "model_stats.csv", model_rows)

    best_by_view = {}
    for view in config["views"]:
        candidates = [
            row for row in aggregate_rows if row["view"] == view
        ]
        best = min(
            candidates,
            key=lambda row: (row["bits_per_event"], row["order"]),
        )
        best_by_view[view] = {
            "selected_order_by_validation_bits_per_event": best["order"],
            "bits_per_event": best["bits_per_event"],
            "perplexity_per_event": best["perplexity_per_event"],
            "glyph_bits_per_glyph": best["glyph_bits_per_glyph"],
            "note": (
                "Selection is within-view only. Event alphabets differ "
                "between token-aware and space-free views."
            ),
        }

    summary = {
        "schema_version": "1.0",
        "analysis_tier": "Tier 1 simple n-gram baselines",
        "fit_scope": "TRAIN ONLY",
        "evaluation_scope": "VALIDATION ONLY",
        "locked_test_accessed": False,
        "orders": list(config["orders"]),
        "views": list(config["views"]),
        "smoothing": config["smoothing"],
        "unknown_policy": config["unknown_policy"],
        "best_order_by_view": best_by_view,
        "validation_results": list(aggregate_rows),
        "provenance": dict(provenance),
        "interpretation_guardrails": [
            "Do not compare token-aware and space-free bits/event as if they share an event alphabet.",
            "Validation may select n-gram order; locked test remains untouched.",
            "No smoothing hyperparameter was tuned on validation.",
            "Per-leaf rows preserve the physical-leaf unit for later paired uncertainty estimates.",
        ],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.0",
        "fit_scope": "TRAIN ONLY",
        "evaluation_scope": "VALIDATION ONLY",
        "locked_test_accessed": False,
        "provenance": dict(provenance),
        "config": dict(config),
        "artifacts": [
            "validation_results.csv",
            "per_leaf_validation.csv",
            "per_locus_validation.csv",
            "model_stats.csv",
            "summary.json",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in artifacts
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit frozen unigram->5-gram baselines on train and evaluate "
            "validation only. test_LOCKED.txt is not read."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
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
        validation_hash = require_hash(
            args.validation,
            EXPECTED_VALIDATION_SHA256,
            "Frozen validation split",
        )
        rules_hash = require_hash(
            args.rules,
            EXPECTED_RULES_SHA256,
            "STA-Eva rules",
        )

        if not args.config.is_file():
            raise ValueError(f"Config not found: {args.config}")
        config_hash = sha256_file(args.config)
        config = load_config(args.config)

        train_leaves = parse_leaf_ids(
            args.train.read_text(encoding="utf-8"),
            label="train",
        )
        validation_leaves = parse_leaf_ids(
            args.validation.read_text(encoding="utf-8"),
            label="validation",
        )

        if len(train_leaves) != 72:
            raise ValueError(
                f"Expected 72 train leaves; found {len(train_leaves)}"
            )
        if len(validation_leaves) != 15:
            raise ValueError(
                "Expected 15 validation leaves; "
                f"found {len(validation_leaves)}"
            )
        overlap = set(train_leaves) & set(validation_leaves)
        if overlap:
            raise ValueError(
                "Train/validation leaf overlap: "
                + ", ".join(f"f{x}" for x in sorted(overlap))
            )

        records = load_ivtff(
            args.source,
            transcription_id="ZL3b",
            strict=True,
        )
        train_records = select_records_by_leaves(
            records,
            train_leaves,
            split_name="train",
        )
        validation_records = select_records_by_leaves(
            records,
            validation_leaves,
            split_name="validation",
        )

        rules = load_bitrans_rules(
            args.rules,
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != rules_hash:
            raise ValueError("Loaded rules hash differs from verified hash")

        train_loci = make_analytical_loci(train_records, rules)
        validation_loci = make_analytical_loci(
            validation_records,
            rules,
        )

        aggregate_rows = []
        leaf_rows = []
        locus_rows = []
        model_rows = []

        max_order = max(config["orders"])

        for view in config["views"]:
            train_sequences = build_baseline_sequences(
                train_loci,
                view=view,
            )
            validation_sequences = build_baseline_sequences(
                validation_loci,
                view=view,
            )

            model = WittenBellNGram(max_order=max_order)
            model.fit(seq.symbols for seq in train_sequences)

            aggregates, leaves, loci = evaluate_model(
                model,
                validation_sequences,
                orders=config["orders"],
                view=view,
            )
            aggregate_rows.extend(aggregates)
            leaf_rows.extend(leaves)
            locus_rows.extend(loci)

            for row in model.context_counts():
                model_rows.append(
                    {
                        "view": view,
                        "training_sequences": len(train_sequences),
                        "training_events": model.training_events,
                        "seen_vocabulary_size": len(
                            model.seen_vocabulary
                        ),
                        **row,
                    }
                )

        provenance = {
            "primary_transcription": "ZL3b",
            "source_path": str(args.source),
            "source_sha256": source_hash,
            "train_split_path": str(args.train),
            "train_split_sha256": train_hash,
            "validation_split_path": str(args.validation),
            "validation_split_sha256": validation_hash,
            "sta1_rules_path": str(args.rules),
            "sta1_rules_sha256": rules_hash,
            "config_path": str(args.config),
            "config_sha256": config_hash,
            "train_physical_leaves": len(train_leaves),
            "validation_physical_leaves": len(validation_leaves),
            "locked_test_accessed": False,
        }

        write_outputs(
            args.output_dir,
            aggregate_rows=aggregate_rows,
            leaf_rows=leaf_rows,
            locus_rows=locus_rows,
            model_rows=model_rows,
            provenance=provenance,
            config=config,
        )

        print("=" * 78)
        print("TIER 1 N-GRAM BASELINES — TRAIN FIT / VALIDATION EVALUATION")
        print("=" * 78)
        print("Smoothing: interpolated Witten-Bell (no tuned alpha)")
        print("Orders:    1, 2, 3, 4, 5")
        print("Test:      NOT ACCESSED")
        print()

        for view in config["views"]:
            print(view)
            print("-" * len(view))
            rows = [
                row for row in aggregate_rows
                if row["view"] == view
            ]
            for row in rows:
                print(
                    f"{row['order']}-gram  "
                    f"bits/event={row['bits_per_event']:.6f}  "
                    f"ppl={row['perplexity_per_event']:.3f}  "
                    f"glyph bits/glyph={row['glyph_bits_per_glyph']:.6f}  "
                    f"OOV={row['oov_events']}"
                )
            best = min(
                rows,
                key=lambda row: (row["bits_per_event"], row["order"]),
            )
            print(
                f"best validation order: {best['order']} "
                f"({best['bits_per_event']:.6f} bits/event)"
            )
            print()

        print(f"Saved baseline artifacts: {args.output_dir}")
        print(
            "Leakage guard: test_LOCKED.txt was not read or required."
        )
        return 0

    except Exception as exc:
        print(f"N-GRAM BASELINE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
