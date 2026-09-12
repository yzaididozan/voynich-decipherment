#!/usr/bin/env python3
"""Fit token-level Markov model on TRAIN; select context order on VALIDATION."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import make_analytical_loci, sha256_file
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.ngram import (
    evaluate_model,
    parse_leaf_ids,
    select_records_by_leaves,
)
from src.models.token_markov import (
    TokenMarkovModel,
    build_matched_trigram_examples,
    evaluate_token_markov,
    extract_token_sequences,
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
DEFAULT_CONFIG = Path("configs/baselines/token_markov_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/token_markov/v1")


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
        "context_orders": [1, 2],
        "max_context": 2,
        "char_base_order": 3,
        "smoothing": "interpolated_witten_bell_to_char_trigram",
        "sequence_boundary": "locus_or_Z1_reset",
        "bos_policy": "left_pad_with_TOKEN_BOS",
        "unknown_policy": "exclude_token_if_contains_Z1_and_reset_context",
        "validation_selection": "lowest_bits_per_event",
        "validation_tie_breaker": "lower_context_order",
        "matched_comparator": "token_reset_char_trigram_with_explicit_EOT",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit frozen token-level Markov candidates on TRAIN and select "
            "token context order on VALIDATION only. The locked test is not "
            "read or required."
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

        train_sequences, train_diag = extract_token_sequences(
            train_loci
        )
        validation_sequences, validation_diag = (
            extract_token_sequences(validation_loci)
        )

        train_token_sequences = [
            [example.glyphs for example in sequence]
            for sequence in train_sequences
        ]

        model = TokenMarkovModel(
            max_context=config["max_context"],
            char_base_order=config["char_base_order"],
        )
        model.fit(train_token_sequences)

        validation_rows = []
        leaf_rows = []
        locus_rows = []

        for context_order in config["context_orders"]:
            aggregate, leaves, loci = evaluate_token_markov(
                model,
                validation_sequences,
                context_order=context_order,
            )

            validation_rows.append(aggregate)
            leaf_rows.extend(leaves)
            locus_rows.extend(loci)

            print(
                f"[token-markov] context={context_order} "
                f"(token {context_order + 1}-gram) "
                f"validation={aggregate['bits_per_event']:.6f} "
                f"bits/event  "
                f"OOV tokens={aggregate['oov_tokens']}"
            )

        selected = min(
            validation_rows,
            key=lambda row: (
                row["bits_per_event"],
                row["context_order"],
            ),
        )

        train_trigram_examples = build_matched_trigram_examples(
            train_sequences
        )
        validation_trigram_examples = (
            build_matched_trigram_examples(validation_sequences)
        )

        # The internal open-vocabulary base was fit on exactly these train
        # token-reset glyph+EOT sequences. Use that same fitted object for the
        # external matched comparator.
        (
            matched_aggregate_rows,
            matched_leaf_rows,
            matched_locus_rows,
        ) = evaluate_model(
            model.char_base_model,
            validation_trigram_examples,
            orders=(config["char_base_order"],),
            view="token_reset_eot",
        )
        matched = matched_aggregate_rows[0]

        if int(matched["total_events"]) != int(selected["events"]):
            raise ValueError(
                "Matched trigram and token-Markov event counts differ: "
                f"trigram={matched['total_events']}, "
                f"token-Markov={selected['events']}"
            )

        # Strong matched-comparator guard: the character base used inside
        # the token Markov recursion is exactly the same fitted token-reset
        # glyph+EOT model evaluated above. Recompute its validation NLL token
        # by token and require exact agreement up to floating-point tolerance.
        direct_base_bits = 0.0
        for sequence in validation_sequences:
            for example in sequence:
                probability = model.character_token_probability(
                    example.glyphs
                )
                if probability <= 0.0:
                    raise RuntimeError(
                        "Non-positive direct character-base probability"
                    )
                direct_base_bits += -__import__("math").log2(probability)

        if abs(direct_base_bits - float(matched["total_bits"])) > 1e-8:
            raise ValueError(
                "Internal character base and matched comparator disagree: "
                f"direct={direct_base_bits:.12f}, "
                f"matched={float(matched['total_bits']):.12f}"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            args.output_dir / "training_context_summary.csv",
            model.context_summary(),
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
            "train_physical_leaves": len(train_leaves),
            "validation_physical_leaves": len(
                validation_leaves
            ),
            "locked_test_accessed": False,
        }

        summary = {
            "schema_version": "1.0",
            "analysis_tier": (
                "token-level Markov generative competitor"
            ),
            "fit_scope": "TRAIN ONLY",
            "evaluation_scope": "VALIDATION ONLY",
            "locked_test_accessed": False,
            "model": {
                "family": (
                    "interpolated token Markov with "
                    "character-trigram open-vocabulary base"
                ),
                "context_orders": config["context_orders"],
                "max_context": config["max_context"],
                "char_base_order": config["char_base_order"],
                "smoothing": config["smoothing"],
                "sequence_boundary": config[
                    "sequence_boundary"
                ],
                "bos_policy": config["bos_policy"],
                "token_base_note": (
                    "Token recursion backs off to the exact same fitted "
                    "token-reset glyph+EOT 3-gram used as the external "
                    "matched comparator; no renormalization is applied."
                ),
            },
            "unknown_policy": config["unknown_policy"],
            "train_token_diagnostics": train_diag,
            "validation_token_diagnostics": validation_diag,
            "training_tokens": model.training_tokens,
            "training_glyphs": model.training_glyphs,
            "training_events": model.training_events,
            "selected_context_order": selected[
                "context_order"
            ],
            "selected_token_ngram_order": selected[
                "token_ngram_order"
            ],
            "selected_validation_bits_per_event": selected[
                "bits_per_event"
            ],
            "selected_validation_bits_per_token": selected[
                "bits_per_token"
            ],
            "matched_trigram": {
                "representation": (
                    "independently reset token + explicit EOT"
                ),
                "order": config["char_base_order"],
                "bits_per_event": matched["bits_per_event"],
                "perplexity_per_event": matched[
                    "perplexity_per_event"
                ],
                "events": matched["total_events"],
            },
            "training_context_summary": (
                model.context_summary()
            ),
            "validation_results": validation_rows,
            "provenance": provenance,
            "guardrails": [
                (
                    "Token context uses previous observed validation tokens "
                    "only; validation targets never update counts."
                ),
                (
                    "Validation selects context order only from frozen "
                    "orders [1, 2]."
                ),
                (
                    "Tokens containing Z1 are excluded and reset token "
                    "context in both train and validation."
                ),
                (
                    "All evaluated token models and matched comparator use "
                    "the same readable token set and glyph+EOT event count."
                ),
                (
                    "No context order above 2 may be added after viewing "
                    "validation results."
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
                "training_context_summary.csv",
                "validation_results.csv",
                "per_leaf_validation.csv",
                "per_locus_validation.csv",
                "matched_trigram_validation.csv",
                "matched_trigram_per_leaf.csv",
                "matched_trigram_per_locus.csv",
                "summary.json",
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
            "TOKEN-LEVEL MARKOV — TRAIN FIT / VALIDATION SELECTION"
        )
        print("=" * 88)
        print(
            "Context grid:       "
            + ", ".join(
                str(x) for x in config["context_orders"]
            )
            + " previous token(s)"
        )
        print(
            f"Character base:     "
            f"{config['char_base_order']}-gram + EOT"
        )
        print(
            "Sequence boundary:   locus boundary or Z1 reset"
        )
        print("Test:                NOT ACCESSED")
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
        print()

        for row in sorted(
            validation_rows,
            key=lambda item: item["context_order"],
        ):
            print(
                f"context={row['context_order']}  "
                f"token-{row['token_ngram_order']}gram  "
                f"bits/event={row['bits_per_event']:.6f}  "
                f"ppl={row['perplexity_per_event']:.3f}  "
                f"OOV tokens={row['oov_tokens']}  "
                f"OOV glyphs={row['oov_glyphs']}"
            )

        print()
        print(
            f"best validation context: "
            f"{selected['context_order']} previous token(s) "
            f"({selected['bits_per_event']:.6f} bits/event)"
        )

        if (
            selected["context_order"]
            == max(config["context_orders"])
        ):
            print(
                "NOTE: selected context is the upper frozen grid edge; "
                "do not expand the grid after seeing validation."
            )

        print()
        print("Matched token-reset character 3-gram + EOT")
        print("------------------------------------------")
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
                "Point estimate: token Markov better than matched "
                f"3-gram by {-delta:.6f} bits/event."
            )
        else:
            print(
                "Point estimate: matched 3-gram better than token "
                f"Markov by {delta:.6f} bits/event."
            )

        print(
            "Use the pre-frozen paired physical-leaf bootstrap script "
            "before making a model-superiority claim."
        )
        print()
        print(
            f"Saved token-Markov artifacts: {args.output_dir}"
        )
        print(
            "Leakage guard: test_LOCKED.txt was not read or required."
        )

        return 0

    except Exception as exc:
        print(
            f"TOKEN-MARKOV ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
