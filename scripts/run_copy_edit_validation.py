#!/usr/bin/env python3
"""Fit local copy-edit models on TRAIN; select source window on VALIDATION."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import make_analytical_loci, sha256_file
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.copy_edit import (
    LocalCopyEditModel,
    build_matched_trigram_examples,
    evaluate_copy_edit,
    extract_copy_sequences,
)
from src.models.ngram import (
    WittenBellNGram,
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
DEFAULT_CONFIG = Path("configs/baselines/copy_edit_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/copy_edit/v1")


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

    expected = {
        "schema_version": "1.0",
        "source_windows": [1, 4, 16],
        "max_token_length": 64,
        "pseudocount": 0.5,
        "rho_max_iterations": 100,
        "rho_tolerance": 1e-10,
        "validation_selection": "lowest_bits_per_event",
        "validation_tie_breaker": "lower_source_window",
        "unknown_policy": "exclude_token_if_contains_Z1_and_reset_context",
        "matched_trigram_order": 3,
        "source_scope": "previous_readable_tokens_within_same_locus_segment",
        "training_source_inference": (
            "minimum_normalized_levenshtein_then_raw_distance_then_recency"
        ),
    }

    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(
                f"Unexpected frozen config value for {key}: "
                f"{data.get(key)!r}; expected {value!r}"
            )

    return data


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
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

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_model(
    path: Path,
    model: LocalCopyEditModel,
) -> None:
    params = model.parameters()
    np.savez_compressed(path, **params)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit frozen local copy-edit/self-copy candidates on TRAIN and "
            "select source-window size on VALIDATION only. "
            "The locked test is not read or required."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
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
            raise ValueError(
                "Loaded rules hash differs from verified hash"
            )

        train_loci = make_analytical_loci(train_records, rules)
        validation_loci = make_analytical_loci(
            validation_records,
            rules,
        )

        train_sequences, train_diag = extract_copy_sequences(
            train_loci
        )
        validation_sequences, validation_diag = (
            extract_copy_sequences(validation_loci)
        )

        max_train_length = max(
            len(example.glyphs)
            for sequence in train_sequences
            for example in sequence
        )
        max_validation_length = max(
            len(example.glyphs)
            for sequence in validation_sequences
            for example in sequence
        )

        if max_train_length > config["max_token_length"]:
            raise ValueError(
                "Training token length exceeds frozen support: "
                f"{max_train_length}"
            )
        if max_validation_length > config["max_token_length"]:
            raise ValueError(
                "Validation token length exceeds frozen support: "
                f"{max_validation_length}"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = args.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        fit_rows = []
        validation_rows = []
        leaf_rows = []
        locus_rows = []

        train_tokens_only = [
            [example.glyphs for example in sequence]
            for sequence in train_sequences
        ]

        for source_window in config["source_windows"]:
            print(
                f"[copy-edit] source_window={source_window}"
            )

            model = LocalCopyEditModel(
                source_window,
                max_token_length=config["max_token_length"],
                pseudocount=config["pseudocount"],
                rho_max_iterations=config[
                    "rho_max_iterations"
                ],
                rho_tolerance=config["rho_tolerance"],
            )

            fit = model.fit(train_tokens_only)

            fit_rows.append(
                {
                    "source_window": source_window,
                    "training_tokens": fit.training_tokens,
                    "training_glyphs": fit.training_glyphs,
                    "training_events": fit.training_events,
                    "inferred_pairs": fit.inferred_pairs,
                    "insertion_count": fit.insertion_count,
                    "deletion_count": fit.deletion_count,
                    "copy_count": fit.copy_count,
                    "substitution_count": fit.substitution_count,
                    "p_insert": fit.p_insert,
                    "p_delete": fit.p_delete,
                    "p_copy": fit.p_copy,
                    "rho_copy": fit.rho_copy,
                    "rho_iterations": fit.rho_iterations,
                    "rho_converged": fit.rho_converged,
                }
            )

            aggregate, leaves, loci = evaluate_copy_edit(
                model,
                validation_sequences,
            )

            aggregate.update(
                {
                    "p_insert": fit.p_insert,
                    "p_delete": fit.p_delete,
                    "p_copy": fit.p_copy,
                    "rho_copy": fit.rho_copy,
                    "rho_iterations": fit.rho_iterations,
                    "rho_converged": fit.rho_converged,
                }
            )

            validation_rows.append(aggregate)
            leaf_rows.extend(leaves)
            locus_rows.extend(loci)

            save_model(
                model_dir
                / f"copy_edit_W{source_window}_train_fit.npz",
                model,
            )

            print(
                f"  p_insert={fit.p_insert:.6f} "
                f"p_delete={fit.p_delete:.6f} "
                f"p_copy={fit.p_copy:.6f} "
                f"rho={fit.rho_copy:.6f}"
            )
            print(
                f"  validation={aggregate['bits_per_event']:.6f} "
                f"bits/event  "
                f"copy-posterior="
                f"{aggregate['mean_copy_posterior_with_context']:.4f}"
            )

        selected = min(
            validation_rows,
            key=lambda row: (
                row["bits_per_event"],
                row["source_window"],
            ),
        )

        # Matched token-reset 3-gram + EOT on exactly the same readable tokens.
        train_trigram_examples = build_matched_trigram_examples(
            train_sequences
        )
        validation_trigram_examples = (
            build_matched_trigram_examples(validation_sequences)
        )

        trigram = WittenBellNGram(
            max_order=config["matched_trigram_order"]
        )
        trigram.fit(
            example.symbols
            for example in train_trigram_examples
        )

        (
            matched_aggregate_rows,
            matched_leaf_rows,
            matched_locus_rows,
        ) = evaluate_model(
            trigram,
            validation_trigram_examples,
            orders=(config["matched_trigram_order"],),
            view="token_reset_eot",
        )
        matched = matched_aggregate_rows[0]

        if int(matched["total_events"]) != int(selected["events"]):
            raise ValueError(
                "Matched trigram and copy-edit event counts differ: "
                f"trigram={matched['total_events']}, "
                f"copy-edit={selected['events']}"
            )

        write_csv(
            args.output_dir / "training_fit.csv",
            fit_rows,
        )
        write_csv(
            args.output_dir / "validation_results.csv",
            validation_rows,
        )
        write_csv(
            args.output_dir / "per_leaf_validation.csv",
            leaf_rows,
        )
        write_csv(
            args.output_dir / "per_locus_validation.csv",
            locus_rows,
        )
        write_csv(
            args.output_dir / "matched_trigram_validation.csv",
            matched_aggregate_rows,
        )
        write_csv(
            args.output_dir / "matched_trigram_per_leaf.csv",
            matched_leaf_rows,
        )
        write_csv(
            args.output_dir / "matched_trigram_per_locus.csv",
            matched_locus_rows,
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
            "numpy_version": np.__version__,
            "train_physical_leaves": len(train_leaves),
            "validation_physical_leaves": len(
                validation_leaves
            ),
            "locked_test_accessed": False,
        }

        summary = {
            "schema_version": "1.0",
            "analysis_tier": (
                "explicit local copy-edit/self-copy generative competitor"
            ),
            "fit_scope": "TRAIN ONLY",
            "evaluation_scope": "VALIDATION ONLY",
            "locked_test_accessed": False,
            "model": {
                "family": "local copy-edit / self-copy",
                "source_windows": config["source_windows"],
                "source_scope": config["source_scope"],
                "training_source_inference": config[
                    "training_source_inference"
                ],
                "edit_channel": (
                    "geometric gap insertions + source deletion + "
                    "exact-copy-or-unigram emission"
                ),
                "innovation_model": (
                    "smoothed token-length categorical times "
                    "smoothed independent glyph categorical"
                ),
                "mixture": (
                    "train-estimated rho between innovation and "
                    "marginalized local copy-edit probability"
                ),
            },
            "unknown_policy": config["unknown_policy"],
            "train_token_diagnostics": train_diag,
            "validation_token_diagnostics": validation_diag,
            "max_train_token_length": max_train_length,
            "max_validation_token_length": max_validation_length,
            "selected_source_window": selected[
                "source_window"
            ],
            "selected_validation_bits_per_event": selected[
                "bits_per_event"
            ],
            "selected_validation_bits_per_token": selected[
                "bits_per_token"
            ],
            "selected_mean_copy_posterior_with_context": selected[
                "mean_copy_posterior_with_context"
            ],
            "matched_trigram": {
                "representation": (
                    "independently reset token + explicit EOT"
                ),
                "order": config["matched_trigram_order"],
                "bits_per_event": matched["bits_per_event"],
                "perplexity_per_event": matched[
                    "perplexity_per_event"
                ],
                "events": matched["total_events"],
                "note": (
                    "Matched comparator uses exactly the same readable "
                    "tokens and event denominator."
                ),
            },
            "training_fit": fit_rows,
            "validation_results": validation_rows,
            "provenance": provenance,
            "guardrails": [
                (
                    "Training source alignment is inferred from TRAIN "
                    "only; validation targets never choose their source."
                ),
                (
                    "Validation source probability marginalizes over "
                    "previous observed readable tokens only."
                ),
                (
                    "Validation selects source window only from the "
                    "frozen grid [1, 4, 16]."
                ),
                (
                    "Tokens containing Z1 are excluded and reset copy "
                    "context in both train and validation."
                ),
                (
                    "Matched token-reset trigram uses exactly the same "
                    "token set and glyph+token event denominator."
                ),
                "Locked test remains untouched.",
            ],
        }

        (args.output_dir / "summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "1.0",
            "config": config,
            "provenance": provenance,
            "artifacts": [
                "training_fit.csv",
                "validation_results.csv",
                "per_leaf_validation.csv",
                "per_locus_validation.csv",
                "matched_trigram_validation.csv",
                "matched_trigram_per_leaf.csv",
                "matched_trigram_per_locus.csv",
                "summary.json",
                "models/*.npz",
            ],
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        artifacts = sorted(
            path
            for path in args.output_dir.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (args.output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  "
                f"{path.relative_to(args.output_dir)}"
                for path in artifacts
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 88)
        print(
            "LOCAL COPY-EDIT / SELF-COPY — "
            "TRAIN FIT / VALIDATION SELECTION"
        )
        print("=" * 88)
        print(
            "Source-window grid: "
            + ", ".join(
                str(x) for x in config["source_windows"]
            )
        )
        print(
            "Source scope:       previous readable tokens "
            "within same locus segment"
        )
        print("Test:               NOT ACCESSED")
        print()
        print(
            f"Train tokens:        "
            f"{train_diag['included_tokens']:,} "
            f"(excluded Z1={train_diag['excluded_tokens_with_Z1']:,})"
        )
        print(
            f"Validation tokens:   "
            f"{validation_diag['included_tokens']:,} "
            f"(excluded Z1={validation_diag['excluded_tokens_with_Z1']:,})"
        )
        print(
            f"Max token length:    train={max_train_length}, "
            f"validation={max_validation_length}"
        )
        print()

        for row in sorted(
            validation_rows,
            key=lambda item: item["source_window"],
        ):
            print(
                f"W={row['source_window']:2d}  "
                f"bits/event={row['bits_per_event']:.6f}  "
                f"ppl={row['perplexity_per_event']:.3f}  "
                f"rho={row['rho_copy']:.4f}  "
                f"copy-post="
                f"{row['mean_copy_posterior_with_context']:.4f}  "
                f"OOV={row['oov_glyphs']}"
            )

        print()
        print(
            f"best validation W: {selected['source_window']} "
            f"({selected['bits_per_event']:.6f} bits/event)"
        )
        if (
            selected["source_window"]
            == max(config["source_windows"])
        ):
            print(
                "NOTE: selected W is the upper frozen grid edge; "
                "do not expand the grid after seeing validation."
            )

        print()
        print("Matched token-reset 3-gram + EOT")
        print("--------------------------------")
        print(
            f"bits/event={matched['bits_per_event']:.6f}  "
            f"ppl={matched['perplexity_per_event']:.3f}  "
            f"events={matched['total_events']}"
        )

        delta = (
            selected["bits_per_event"]
            - matched["bits_per_event"]
        )
        print()
        if delta < 0:
            print(
                "Point estimate: copy-edit better than matched "
                f"3-gram by {-delta:.6f} bits/event."
            )
        else:
            print(
                "Point estimate: matched 3-gram better than copy-edit "
                f"by {delta:.6f} bits/event."
            )

        print(
            "Use a paired physical-leaf bootstrap before making a "
            "model-superiority claim."
        )
        print()
        print(f"Saved copy-edit artifacts: {args.output_dir}")
        print(
            "Leakage guard: test_LOCKED.txt was not read or required."
        )
        return 0

    except Exception as exc:
        print(
            f"COPY-EDIT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
