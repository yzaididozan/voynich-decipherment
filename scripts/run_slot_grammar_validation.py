#!/usr/bin/env python3
"""Fit finite-template slot grammar on train; select K on validation."""

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
from src.models.ngram import (
    WittenBellNGram,
    evaluate_model,
    parse_leaf_ids,
    select_records_by_leaves,
)
from src.models.slot_grammar import (
    FiniteTemplateSlotGrammar,
    build_matched_trigram_examples,
    evaluate_slot_grammar,
    extract_slot_tokens,
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
DEFAULT_CONFIG = Path("configs/baselines/slot_grammar_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/slot_grammar/v1")


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
        raise ValueError("Unsupported slot-grammar config schema")
    if data.get("templates") != [1, 2, 4, 8, 16]:
        raise ValueError(
            "Template grid must be exactly [1, 2, 4, 8, 16]"
        )
    if data.get("slot_count") != 8:
        raise ValueError("slot_count must be exactly 8")
    if data.get("max_token_length") != 64:
        raise ValueError("max_token_length must be exactly 64")
    if data.get("restarts") != 5:
        raise ValueError("restarts must be exactly 5")
    if float(data.get("pseudocount", -1)) != 0.5:
        raise ValueError("pseudocount must be exactly 0.5")
    if (
        data.get("restart_selection")
        != "highest_training_log_likelihood"
    ):
        raise ValueError(
            "Restarts must be selected by training likelihood only"
        )
    if (
        data.get("validation_selection")
        != "lowest_bits_per_event"
    ):
        raise ValueError(
            "Validation must select template count by bits/event"
        )
    if data.get("matched_trigram_order") != 3:
        raise ValueError("Matched comparator must be trigram order 3")
    if data.get("unknown_policy") != "exclude_token_if_contains_Z1":
        raise ValueError("Unexpected transcription-unknown policy")

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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_model(
    path: Path,
    model: FiniteTemplateSlotGrammar,
    *,
    templates: int,
    restart: int,
    seed: int,
) -> None:
    params = model.parameters()
    np.savez_compressed(
        path,
        vocabulary=params["vocabulary"],
        length_probabilities=params["length_probabilities"],
        mixture_weights=params["mixture_weights"],
        emission=params["emission"],
        templates=np.asarray(templates),
        restart=np.asarray(restart),
        seed=np.asarray(seed),
        slot_count=np.asarray(model.slot_count),
        max_token_length=np.asarray(model.max_token_length),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the finite-template slot grammar on frozen train leaves "
            "and select template count on frozen validation leaves only. "
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

        train_tokens, train_diag = extract_slot_tokens(train_loci)
        validation_tokens, validation_diag = extract_slot_tokens(
            validation_loci
        )

        max_train_length = max(
            len(example.glyphs)
            for example in train_tokens
        )
        max_validation_length = max(
            len(example.glyphs)
            for example in validation_tokens
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

        restart_rows = []
        validation_rows = []
        leaf_rows = []
        locus_rows = []

        train_token_sequences = [
            example.glyphs
            for example in train_tokens
        ]

        for templates in config["templates"]:
            fitted = []

            for restart in range(config["restarts"]):
                seed = (
                    int(config["seed"])
                    + templates * 10_000
                    + restart
                )

                print(
                    f"[slot] K={templates:2d} "
                    f"restart={restart + 1}/{config['restarts']} "
                    f"seed={seed}"
                )

                model = FiniteTemplateSlotGrammar(
                    templates,
                    slot_count=config["slot_count"],
                    max_token_length=config["max_token_length"],
                    pseudocount=config["pseudocount"],
                    max_iterations=config["max_iterations"],
                    min_iterations=config["min_iterations"],
                    tolerance_bits_per_token=(
                        config["tolerance_bits_per_token"]
                    ),
                )

                fit = model.fit(
                    train_token_sequences,
                    seed=seed,
                )

                restart_rows.append(
                    {
                        "templates": templates,
                        "restart": restart,
                        "seed": seed,
                        "iterations": fit.iterations,
                        "converged": fit.converged,
                        "training_log_likelihood_nats": (
                            fit.training_log_likelihood_nats
                        ),
                        "training_tokens": fit.training_tokens,
                        "training_glyphs": fit.training_glyphs,
                        "training_bits_per_token": (
                            fit.training_bits_per_token
                        ),
                        "training_bits_per_event": (
                            fit.training_bits_per_event
                        ),
                    }
                )
                fitted.append((fit, restart, seed, model))

            # Restart selection is intentionally TRAIN ONLY.
            best_fit, best_restart, best_seed, best_model = max(
                fitted,
                key=lambda item: (
                    item[0].training_log_likelihood_nats,
                    -item[1],
                ),
            )

            aggregate, leaves, loci = evaluate_slot_grammar(
                best_model,
                validation_tokens,
                templates=templates,
                restart=best_restart,
                seed=best_seed,
            )

            aggregate["training_bits_per_event"] = (
                best_fit.training_bits_per_event
            )
            aggregate["training_bits_per_token"] = (
                best_fit.training_bits_per_token
            )
            aggregate["training_iterations"] = best_fit.iterations
            aggregate["training_converged"] = best_fit.converged
            aggregate["train_selected_restart"] = True

            validation_rows.append(aggregate)
            leaf_rows.extend(leaves)
            locus_rows.extend(loci)

            save_model(
                model_dir / f"slot_K{templates}_train_best.npz",
                best_model,
                templates=templates,
                restart=best_restart,
                seed=best_seed,
            )

            print(
                f"  train-best restart={best_restart + 1}; "
                f"train={best_fit.training_bits_per_event:.6f} "
                f"bits/event; "
                f"validation={aggregate['bits_per_event']:.6f} "
                f"bits/event"
            )

        selected = min(
            validation_rows,
            key=lambda row: (
                row["bits_per_event"],
                row["templates"],
            ),
        )

        # Matched token-reset trigram. This is a NEW matched comparator and
        # does not replace the previously frozen primary n-gram baseline.
        train_trigram_examples = build_matched_trigram_examples(
            train_tokens
        )
        validation_trigram_examples = (
            build_matched_trigram_examples(validation_tokens)
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

        # Sanity guard: the matched denominator must exactly equal the slot
        # grammar denominator: one terminal event per included token.
        if int(matched["total_events"]) != int(selected["events"]):
            raise ValueError(
                "Matched trigram and slot grammar event counts differ: "
                f"trigram={matched['total_events']}, "
                f"slot={selected['events']}"
            )

        write_csv(
            args.output_dir / "training_restarts.csv",
            restart_rows,
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
            "validation_physical_leaves": len(validation_leaves),
            "locked_test_accessed": False,
        }

        summary = {
            "schema_version": "1.0",
            "analysis_tier": (
                "finite-template slot-grammar generative competitor"
            ),
            "fit_scope": "TRAIN ONLY",
            "evaluation_scope": "VALIDATION ONLY",
            "locked_test_accessed": False,
            "model": {
                "family": "latent finite-template slot grammar",
                "slot_count": config["slot_count"],
                "relative_slot_mapping": (
                    "equally spaced token-relative positions; "
                    "initial->slot0, final->slot7"
                ),
                "length_model": (
                    "Dirichlet-smoothed categorical over lengths 1..64"
                ),
                "glyph_emissions": (
                    "template- and relative-slot-specific categorical"
                ),
                "template_grid": config["templates"],
            },
            "unknown_policy": config["unknown_policy"],
            "train_token_diagnostics": train_diag,
            "validation_token_diagnostics": validation_diag,
            "max_train_token_length": max_train_length,
            "max_validation_token_length": max_validation_length,
            "selected_template_count": selected["templates"],
            "selected_validation_bits_per_event": (
                selected["bits_per_event"]
            ),
            "selected_validation_bits_per_token": (
                selected["bits_per_token"]
            ),
            "matched_trigram": {
                "representation": (
                    "independently reset token + explicit EOT"
                ),
                "order": config["matched_trigram_order"],
                "bits_per_event": matched["bits_per_event"],
                "perplexity_per_event": matched["perplexity_per_event"],
                "events": matched["total_events"],
                "note": (
                    "This matched comparator is for direct slot-model "
                    "comparison and does not replace the frozen primary "
                    "locus-level 3-gram baseline."
                ),
            },
            "validation_results": validation_rows,
            "provenance": provenance,
            "guardrails": [
                "Random restarts selected by training likelihood only.",
                "Validation selects template count only from frozen grid.",
                "Tokens containing transcription-unknown Z1 are excluded from both slot and matched-trigram evaluation.",
                "Raw slot-model bits/event must not be compared directly with the earlier locus-level token-aware n-gram; use the matched token-reset trigram.",
                "Locked test remains untouched.",
            ],
        }

        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "1.0",
            "config": config,
            "provenance": provenance,
            "artifacts": [
                "training_restarts.csv",
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
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
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
        print("=" * 82)
        print("FINITE-TEMPLATE SLOT GRAMMAR — TRAIN FIT / VALIDATION SELECTION")
        print("=" * 82)
        print(f"Relative slots:  {config['slot_count']}")
        print(
            "Template grid:   "
            + ", ".join(str(x) for x in config["templates"])
        )
        print(f"Restarts/K:      {config['restarts']}")
        print(
            "Restart rule:    highest TRAINING likelihood only"
        )
        print("Test:            NOT ACCESSED")
        print()
        print(
            f"Train tokens:     {train_diag['included_tokens']:,} "
            f"(excluded Z1 tokens={train_diag['excluded_tokens_with_Z1']:,})"
        )
        print(
            f"Validation tokens:{validation_diag['included_tokens']:>7,} "
            f"(excluded Z1 tokens={validation_diag['excluded_tokens_with_Z1']:,})"
        )
        print(
            f"Max token length: train={max_train_length}, "
            f"validation={max_validation_length}"
        )
        print()

        for row in sorted(
            validation_rows,
            key=lambda item: item["templates"],
        ):
            print(
                f"K={row['templates']:2d}  "
                f"bits/event={row['bits_per_event']:.6f}  "
                f"ppl={row['perplexity_per_event']:.3f}  "
                f"bits/token={row['bits_per_token']:.3f}  "
                f"restart={row['restart'] + 1}  "
                f"OOV glyphs={row['oov_glyphs']}"
            )

        print()
        print(
            f"best validation K: {selected['templates']} "
            f"({selected['bits_per_event']:.6f} bits/event)"
        )
        print()
        print("Matched token-reset 3-gram + EOT")
        print("--------------------------------")
        print(
            f"bits/event={matched['bits_per_event']:.6f}  "
            f"ppl={matched['perplexity_per_event']:.3f}  "
            f"events={matched['total_events']}"
        )
        print()
        delta = (
            selected["bits_per_event"]
            - matched["bits_per_event"]
        )
        if delta < 0:
            print(
                "Point estimate: slot grammar better than matched "
                f"3-gram by {-delta:.6f} bits/event."
            )
        else:
            print(
                "Point estimate: matched 3-gram better than slot "
                f"grammar by {delta:.6f} bits/event."
            )
        print(
            "Use paired physical-leaf bootstrap before making a "
            "model-superiority claim."
        )
        print()
        print(f"Saved slot-grammar artifacts: {args.output_dir}")
        print(
            "Leakage guard: test_LOCKED.txt was not read or required."
        )
        return 0

    except Exception as exc:
        print(f"SLOT-GRAMMAR ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
