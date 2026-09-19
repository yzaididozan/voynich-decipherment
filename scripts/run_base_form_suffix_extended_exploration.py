#!/usr/bin/env python3
"""Run Phase 3A.4 extended base-form surface multiplicity exploration.

Intended repository destination:
    scripts/run_base_form_suffix_extended_exploration.py

Grid:
    4 languages x K={4,8,16} x 5 seeds = 60 controls
    60 controls x 5 frozen predictive relations = 300 relation rows

Critical bridge:
    For every source/seed, K=4 corpus.jsonl must be byte-for-byte identical to
    the frozen Phase-3A.3 K=4 corpus.jsonl before aggregate results are written.

The runner reuses the frozen Level-2 evaluator on CONTROL TRAIN/VALIDATION
only. Reserved control-test lines are not encoded, parsed, fitted, or scored.
Voynich test_LOCKED.txt is not read or required.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.ciphers import base_form_suffix_extended
from src.ciphers.base_form_suffix_extended import (
    generate_control,
    read_source_lines,
    sha256_file,
    token_count,
    write_generated_control,
)
from src.evaluation.level2_predictive_controls import (
    evaluate_one_control,
    read_control_units,
    write_csv,
)


DEFAULT_CONFIG = Path(
    "configs/mechanisms/base_form_suffix_extended_v1.json"
)


def config_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true", "1", "yes",
    }


def normalize_relation_row(row: Mapping[str, object]) -> dict:
    out = dict(row)

    int_fields = (
        "control_seed",
        "variant_count",
        "selected_hyperparameter",
        "bootstrap_units",
        "bootstrap_replicates",
        "bootstrap_seed",
    )
    float_fields = (
        "competitor_bits_per_event",
        "trigram_bits_per_event",
        "delta_competitor_minus_trigram_bits_per_event",
        "relative_nll_reduction_trigram_vs_competitor",
        "ci_95_lower",
        "ci_95_upper",
        "bootstrap_probability_trigram_better",
        "bootstrap_probability_competitor_better",
    )
    bool_fields = (
        "direction_match_voynich",
        "strong_direction_match_voynich",
    )

    for key in int_fields:
        if key in out and out[key] not in ("", None):
            out[key] = int(out[key])

    for key in float_fields:
        if key in out and out[key] not in ("", None):
            out[key] = float(out[key])

    for key in bool_fields:
        if key in out:
            out[key] = as_bool(out[key])

    return out


def verify_parent_freeze(
    config: Mapping[str, object],
) -> dict:
    parent = config["parent_phase3a3"]

    parent_config_path = Path(
        parent["config_path"]
    )
    parent_generator_path = Path(
        parent["generator_path"]
    )

    if not parent_config_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.3 config is missing: {parent_config_path}"
        )
    if not parent_generator_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.3 generator is missing: {parent_generator_path}"
        )

    observed_config_sha = sha256_file(
        parent_config_path
    )
    observed_generator_sha = sha256_file(
        parent_generator_path
    )

    if (
        observed_config_sha
        != parent["expected_config_sha256"]
    ):
        raise ValueError(
            "Frozen Phase-3A.3 config SHA differs from the Phase-3A.4 "
            f"bridge declaration: {observed_config_sha}"
        )

    if (
        observed_generator_sha
        != parent["expected_generator_sha256"]
    ):
        raise ValueError(
            "Frozen Phase-3A.3 generator SHA differs from the Phase-3A.4 "
            f"bridge declaration: {observed_generator_sha}"
        )

    return {
        "parent_phase3a3_config_sha256": observed_config_sha,
        "parent_phase3a3_generator_sha256": observed_generator_sha,
    }


def verify_k4_bridge(
    *,
    config: Mapping[str, object],
    source_id: str,
    seed: int,
    new_corpus_path: Path,
    new_key_path: Path,
) -> dict:
    """Require exact corpus and key-prefix continuity with Phase 3A.3 K=4."""
    parent_root = Path(
        config["parent_phase3a3"]["control_root"]
    )
    parent_dir = (
        parent_root
        / source_id
        / "K4"
        / f"seed_{seed}"
    )
    parent_corpus_path = (
        parent_dir / "corpus.jsonl"
    )
    parent_key_path = (
        parent_dir / "key.json"
    )

    if not parent_corpus_path.is_file():
        raise FileNotFoundError(
            "Phase-3A.3 bridge corpus is missing: "
            f"{parent_corpus_path}"
        )
    if not parent_key_path.is_file():
        raise FileNotFoundError(
            "Phase-3A.3 bridge key is missing: "
            f"{parent_key_path}"
        )

    parent_corpus_sha = sha256_file(
        parent_corpus_path
    )
    new_corpus_sha = sha256_file(
        new_corpus_path
    )

    if new_corpus_sha != parent_corpus_sha:
        raise ValueError(
            f"K=4 BRIDGE FAILURE for {source_id}/seed={seed}: "
            f"Phase-3A.4 corpus {new_corpus_sha} != "
            f"Phase-3A.3 corpus {parent_corpus_sha}"
        )

    parent_key = json.loads(
        parent_key_path.read_text(
            encoding="utf-8"
        )
    )
    new_key = json.loads(
        new_key_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        new_key["parent_phase3a3_master_key_sha256"]
        != parent_key["master_key_sha256"]
    ):
        raise ValueError(
            f"K=4 parent master-key fingerprint mismatch: "
            f"{source_id}/seed={seed}"
        )

    if (
        new_key["mono_forward"]
        != parent_key["mono_forward"]
    ):
        raise ValueError(
            f"K=4 base mapping differs from Phase 3A.3: "
            f"{source_id}/seed={seed}"
        )

    if (
        new_key["legacy_phase3a3_suffixes"]
        != parent_key["master_suffixes"]
    ):
        raise ValueError(
            f"K=4 legacy suffix order differs from Phase 3A.3: "
            f"{source_id}/seed={seed}"
        )

    return {
        "source_id": source_id,
        "seed": int(seed),
        "parent_corpus_sha256": parent_corpus_sha,
        "phase3a4_k4_corpus_sha256": new_corpus_sha,
        "corpus_byte_identical": True,
        "base_mapping_identical": True,
        "first_four_suffixes_identical": True,
        "parent_master_key_sha256": parent_key[
            "master_key_sha256"
        ],
        "phase3a4_parent_master_key_sha256": new_key[
            "parent_phase3a3_master_key_sha256"
        ],
    }


def flatten_mechanism_diagnostics(
    *,
    source_id: str,
    language: str,
    variant_count: int,
    seed: int,
    generated_summary: Mapping[str, object],
    master_key_sha256: str,
    parent_master_key_sha256: str,
) -> dict:
    train = generated_summary["train_diagnostics"]
    validation = generated_summary["validation_diagnostics"]

    row = {
        "source_id": source_id,
        "language": language,
        "variant_count": int(variant_count),
        "seed": int(seed),
        "master_key_sha256": master_key_sha256,
        "parent_phase3a3_master_key_sha256": (
            parent_master_key_sha256
        ),
        "reserved_test_encoded": bool(
            generated_summary["reserved_test_encoded"]
        ),
    }

    for prefix, diag in (
        ("train", train),
        ("validation", validation),
    ):
        for key, value in diag.items():
            row[f"{prefix}_{key}"] = value

    return row


def verify_master_key_invariance(
    diagnostic_rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
) -> None:
    groups = {}

    for row in diagnostic_rows:
        key = (
            row["source_id"],
            int(row["seed"]),
        )
        groups.setdefault(key, []).append(row)

    for (source_id, seed), rows in groups.items():
        observed_k = sorted(
            int(row["variant_count"])
            for row in rows
        )
        if observed_k != sorted(
            int(x) for x in variant_counts
        ):
            raise ValueError(
                f"{source_id}/seed={seed}: expected K grid "
                f"{list(variant_counts)}, found {observed_k}"
            )

        extended_fingerprints = {
            row["master_key_sha256"]
            for row in rows
        }
        if len(extended_fingerprints) != 1:
            raise ValueError(
                f"{source_id}/seed={seed}: extended master key changed across K"
            )

        parent_fingerprints = {
            row["parent_phase3a3_master_key_sha256"]
            for row in rows
        }
        if len(parent_fingerprints) != 1:
            raise ValueError(
                f"{source_id}/seed={seed}: parent Phase-3A.3 key changed across K"
            )


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
    languages: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], List[dict]]:
    language_rows: List[dict] = []
    global_rows: List[dict] = []

    for k in variant_counts:
        for relation in relations:
            global_group = [
                row
                for row in rows
                if int(row["variant_count"]) == int(k)
                and row["relation"] == relation
            ]

            if len(global_group) != 20:
                raise ValueError(
                    f"K={k}/{relation}: expected 20 runs; "
                    f"found {len(global_group)}"
                )

            deltas = [
                float(
                    row[
                        "delta_competitor_minus_trigram_bits_per_event"
                    ]
                )
                for row in global_group
            ]

            global_rows.append(
                {
                    "variant_count": int(k),
                    "relation": relation,
                    "runs": len(global_group),
                    "mean_delta_bits_per_event": mean(deltas),
                    "median_delta_bits_per_event": median(deltas),
                    "sd_delta_bits_per_event": pstdev(deltas),
                    "min_delta_bits_per_event": min(deltas),
                    "max_delta_bits_per_event": max(deltas),
                    "positive_direction_runs": sum(
                        bool(row["direction_match_voynich"])
                        for row in global_group
                    ),
                    "strong_direction_runs": sum(
                        bool(
                            row[
                                "strong_direction_match_voynich"
                            ]
                        )
                        for row in global_group
                    ),
                    "note": (
                        "Exploratory descriptive summary only; "
                        "no automatic winner threshold."
                    ),
                }
            )

            for language in languages:
                group = [
                    row
                    for row in global_group
                    if row["language"] == language
                ]
                if len(group) != 5:
                    raise ValueError(
                        f"K={k}/{relation}/{language}: "
                        f"expected 5 seeds; found {len(group)}"
                    )

                language_deltas = [
                    float(
                        row[
                            "delta_competitor_minus_trigram_bits_per_event"
                        ]
                    )
                    for row in group
                ]

                language_rows.append(
                    {
                        "variant_count": int(k),
                        "relation": relation,
                        "language": language,
                        "seeds": 5,
                        "mean_delta_bits_per_event": mean(
                            language_deltas
                        ),
                        "sd_delta_bits_per_event": pstdev(
                            language_deltas
                        ),
                        "positive_direction_seeds": sum(
                            bool(
                                row[
                                    "direction_match_voynich"
                                ]
                            )
                            for row in group
                        ),
                        "strong_direction_seeds": sum(
                            bool(
                                row[
                                    "strong_direction_match_voynich"
                                ]
                            )
                            for row in group
                        ),
                    }
                )

    return language_rows, global_rows


def summarize_suffix_diagnostics(
    rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
    languages: Sequence[str],
) -> List[dict]:
    output = []

    fields = [
        "validation_token_coverage_fraction",
        "validation_base_form_recovery_accuracy",
        "validation_base_form_preservation_fraction",
        "validation_token_length_plus_one_accuracy",
        "validation_mean_realized_surface_forms_per_plaintext_word_present",
        "validation_same_surface_probability_given_same_plaintext_word",
        "validation_exact_cipher_token_recurrence_fraction",
        "validation_suffix_entropy_bits",
        "validation_suffix_entropy_fraction_of_max",
    ]

    for k in variant_counts:
        for language in languages:
            group = [
                row
                for row in rows
                if int(row["variant_count"]) == int(k)
                and row["language"] == language
            ]

            if len(group) != 5:
                raise ValueError(
                    f"K={k}/{language}: expected 5 diagnostics; "
                    f"found {len(group)}"
                )

            output_row = {
                "variant_count": int(k),
                "language": language,
                "seeds": 5,
            }

            for field in fields:
                values = [
                    float(item[field])
                    for item in group
                    if item[field] not in ("", None)
                ]
                output_row[f"mean_{field}"] = (
                    mean(values)
                    if values else None
                )

            output.append(output_row)

    return output


def trajectory_summary(
    global_rows: Sequence[Mapping[str, object]],
    suffix_rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
) -> dict:
    token_rows = {
        int(row["variant_count"]): row
        for row in global_rows
        if row["relation"] == "token_markov"
    }

    token_means = [
        float(
            token_rows[int(k)][
                "mean_delta_bits_per_event"
            ]
        )
        for k in variant_counts
    ]

    non_token_relations = {
        "hmm_space_free",
        "hmm_token_aware",
        "slot_grammar",
        "copy_edit",
    }

    preserved = {}
    all_other_four_positive = {}

    for k in variant_counts:
        relevant = [
            row
            for row in global_rows
            if int(row["variant_count"]) == int(k)
            and row["relation"] in non_token_relations
        ]

        preserved[str(k)] = {
            row["relation"]: {
                "mean_delta_bits_per_event": float(
                    row["mean_delta_bits_per_event"]
                ),
                "positive_direction_runs_20": int(
                    row["positive_direction_runs"]
                ),
                "strong_direction_runs_20": int(
                    row["strong_direction_runs"]
                ),
            }
            for row in relevant
        }

        all_other_four_positive[str(k)] = all(
            float(row["mean_delta_bits_per_event"]) > 0
            for row in relevant
        )

    same_surface = {}
    for k in variant_counts:
        relevant = [
            row
            for row in suffix_rows
            if int(row["variant_count"]) == int(k)
        ]
        values = [
            float(
                row[
                    "mean_validation_same_surface_probability_given_same_plaintext_word"
                ]
            )
            for row in relevant
            if row[
                "mean_validation_same_surface_probability_given_same_plaintext_word"
            ] is not None
        ]
        same_surface[str(k)] = (
            mean(values)
            if values else None
        )

    return {
        "token_markov_mean_delta_by_k": {
            str(k): token_means[i]
            for i, k in enumerate(variant_counts)
        },
        "token_markov_mean_delta_monotone_non_decreasing": all(
            right >= left
            for left, right in zip(
                token_means,
                token_means[1:],
            )
        ),
        "token_markov_mean_delta_positive_by_k": {
            str(k): token_means[i] > 0
            for i, k in enumerate(variant_counts)
        },
        "mean_same_surface_probability_by_k": same_surface,
        "other_four_relations_by_k": preserved,
        "all_other_four_mean_deltas_positive_by_k": (
            all_other_four_positive
        ),
        "interpretation": (
            "Exploratory trajectory only. K is capped at 16 in this experiment. "
            "Do not select a confirmatory K or access reserved-test data without "
            "a separately frozen Phase-3B protocol."
        ),
    }


def require_level2_compatibility(
    phase3a4: Mapping[str, object],
    level2: Mapping[str, object],
) -> None:
    if level2.get("freeze_id") != "level2-predictive-controls-v1":
        raise ValueError(
            "Unexpected Level-2 evaluator config freeze_id"
        )

    if (
        list(level2["operational_relations"])
        != list(phase3a4["operational_relations"])
    ):
        raise ValueError(
            "Phase-3A.4 operational relations differ from Level 2"
        )

    for source_id, source in phase3a4["sources"].items():
        level2_source = level2["source_units"].get(source_id)
        if level2_source is None:
            raise ValueError(
                f"Source absent from Level-2 config: {source_id}"
            )
        if (
            level2_source["source_sha256"]
            != source["source_sha256"]
        ):
            raise ValueError(
                f"Source hash differs from Level 2: {source_id}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate Phase-3A.4 extended base-form suffix "
            "multiplicity controls on frozen Level-2 TRAIN/VALIDATION."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        config_hash = config_sha256(
            args.config
        )
        config = json.loads(
            args.config.read_text(
                encoding="utf-8"
            )
        )

        if (
            config.get("experiment_id")
            != "base-form-suffix-extended-exploration-v1"
        ):
            raise ValueError(
                "Unexpected Phase-3A.4 experiment_id"
            )
        if config.get("status") != "exploratory":
            raise ValueError(
                "Phase 3A.4 must remain explicitly exploratory"
            )

        parent_freeze = verify_parent_freeze(
            config
        )

        mechanism = config["mechanism"]
        variant_counts = [
            int(x)
            for x in mechanism["variant_counts"]
        ]
        if variant_counts != [4, 8, 16]:
            raise ValueError(
                "Extended suffix K grid must be exactly [4,8,16]"
            )
        if int(mechanism["master_suffix_count"]) != 16:
            raise ValueError(
                "master_suffix_count must be exactly 16"
            )
        if (
            mechanism["coverage"]
            != "all_train_and_validation_tokens"
        ):
            raise ValueError(
                "Phase 3A.4 coverage must be all TRAIN/VALIDATION tokens"
            )

        seeds = [
            int(x)
            for x in config["seeds"]
        ]
        if len(seeds) != 5:
            raise ValueError(
                "Phase 3A.4 requires exactly five frozen seeds"
            )

        expected_runs = int(
            config["expected_runs"]
        )
        calculated_runs = (
            len(config["sources"])
            * len(variant_counts)
            * len(seeds)
        )
        if expected_runs != calculated_runs:
            raise ValueError(
                "expected_runs does not match source/K/seed cardinality"
            )

        generator_hash = sha256_file(
            Path(
                base_form_suffix_extended.__file__
            )
        )

        level2_config_path = Path(
            config["level2_config_path"]
        )
        level2_hash = config_sha256(
            level2_config_path
        )
        level2 = json.loads(
            level2_config_path.read_text(
                encoding="utf-8"
            )
        )
        require_level2_compatibility(
            config,
            level2,
        )

        source_root = Path(
            config["source_root"]
        )
        split_root = Path(
            config["level2_split_root"]
        )
        control_root = Path(
            config["control_output_root"]
        )
        result_root = Path(
            config["result_output_root"]
        )
        result_run_root = (
            result_root / "runs"
        )
        result_run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_relations: List[dict] = []
        mechanism_rows: List[dict] = []
        bridge_rows: List[dict] = []
        completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A.4 — EXTENDED BASE-FORM SURFACE MULTIPLICITY"
        )
        print("=" * 96)
        print(f"Experiment:       {config['experiment_id']}")
        print(f"Status:           {config['status'].upper()}")
        print(f"Config SHA:       {config_hash}")
        print(f"Generator SHA:    {generator_hash}")
        print(
            "Parent 3A.3 cfg:  "
            f"{parent_freeze['parent_phase3a3_config_sha256']}"
        )
        print(
            "Parent 3A.3 gen:  "
            f"{parent_freeze['parent_phase3a3_generator_sha256']}"
        )
        print(f"Level-2 config:   {level2_hash}")
        print("Languages:        4")
        print("Suffix K grid:    4, 8, 16")
        print("Coverage:         100% TRAIN/VALIDATION tokens")
        print("Token length:     base + exactly 1 suffix (all K)")
        print("K=4 bridge:       BYTE-IDENTICAL Phase 3A.3 REQUIRED")
        print("Seeds/K/source:   5")
        print(f"Expected runs:    {expected_runs}")
        print("Control test:     NOT ENCODED / NOT SCORED")
        print("Voynich test:     NOT ACCESSED")
        print()

        for source_id, source_cfg in config[
            "sources"
        ].items():
            language = source_cfg["language"]
            source_path = (
                source_root
                / source_id
                / "source.txt"
            )
            source_hash = sha256_file(
                source_path
            )

            if (
                source_hash
                != source_cfg["source_sha256"]
            ):
                raise ValueError(
                    f"{source_id}: frozen source SHA mismatch"
                )

            lines = read_source_lines(
                source_path
            )
            if len(lines) != int(
                source_cfg["expected_lines"]
            ):
                raise ValueError(
                    f"{source_id}: expected "
                    f"{source_cfg['expected_lines']} lines, "
                    f"found {len(lines)}"
                )
            if token_count(lines) != int(
                source_cfg["expected_tokens"]
            ):
                raise ValueError(
                    f"{source_id}: expected "
                    f"{source_cfg['expected_tokens']} tokens, "
                    f"found {token_count(lines)}"
                )

            split_path = (
                split_root / f"{source_id}.json"
            )
            split_hash = sha256_file(
                split_path
            )
            split = json.loads(
                split_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                split.get("source_sha256")
                != source_hash
            ):
                raise ValueError(
                    f"{source_id}: Level-2 split source hash mismatch"
                )

            train_indices = [
                int(x) for x in split["train"]
            ]
            validation_indices = [
                int(x)
                for x in split["validation"]
            ]
            reserved_indices = [
                int(x)
                for x in split["reserved_test"]
            ]

            print(
                f"[{language}] {source_id}: "
                f"train={len(train_indices)} "
                f"validation={len(validation_indices)} "
                f"reserved={len(reserved_indices)}"
            )

            for k in variant_counts:
                for seed in seeds:
                    control_dir = (
                        control_root
                        / source_id
                        / f"K{k}"
                        / f"seed_{seed}"
                    )
                    corpus_path = (
                        control_dir / "corpus.jsonl"
                    )
                    control_summary_path = (
                        control_dir / "summary.json"
                    )
                    key_path = (
                        control_dir / "key.json"
                    )

                    if (
                        corpus_path.is_file()
                        and control_summary_path.is_file()
                        and key_path.is_file()
                    ):
                        existing = json.loads(
                            control_summary_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        provenance = existing.get(
                            "provenance",
                            {},
                        )

                        required = {
                            "phase3a4_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                            "parent_phase3a3_generator_sha256": (
                                parent_freeze[
                                    "parent_phase3a3_generator_sha256"
                                ]
                            ),
                        }
                        for key_name, value in required.items():
                            if provenance.get(key_name) != value:
                                raise ValueError(
                                    f"Existing generated control has "
                                    f"different {key_name}: {control_dir}"
                                )

                        if int(
                            existing["variant_count"]
                        ) != k or int(
                            existing["seed"]
                        ) != seed:
                            raise ValueError(
                                "Existing generated control identity mismatch"
                            )

                        generated_summary = existing
                        generated_key = json.loads(
                            key_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        generation_status = "REUSED"
                    else:
                        generated = generate_control(
                            lines,
                            train_indices=train_indices,
                            validation_indices=validation_indices,
                            reserved_indices=reserved_indices,
                            seed=seed,
                            variant_count=k,
                            master_suffix_count=16,
                        )

                        provenance = {
                            "phase3a4_config_path": str(
                                args.config
                            ),
                            "phase3a4_config_sha256": config_hash,
                            "generator_path": str(
                                Path(
                                    base_form_suffix_extended.__file__
                                )
                            ),
                            "generator_sha256": generator_hash,
                            "parent_phase3a3_config_sha256": (
                                parent_freeze[
                                    "parent_phase3a3_config_sha256"
                                ]
                            ),
                            "parent_phase3a3_generator_sha256": (
                                parent_freeze[
                                    "parent_phase3a3_generator_sha256"
                                ]
                            ),
                            "source_path": str(
                                source_path
                            ),
                            "source_sha256": source_hash,
                            "split_path": str(
                                split_path
                            ),
                            "split_sha256": split_hash,
                            "level2_config_path": str(
                                level2_config_path
                            ),
                            "level2_config_sha256": level2_hash,
                        }

                        write_generated_control(
                            control_dir,
                            generated,
                            provenance=provenance,
                        )
                        generated_summary = json.loads(
                            control_summary_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        generated_key = json.loads(
                            key_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        generation_status = "GENERATED"

                    if generated_summary.get(
                        "reserved_test_encoded"
                    ) is not False:
                        raise ValueError(
                            "Reserved test must not be encoded"
                        )
                    if generated_summary.get(
                        "base_form_preservation_verified_train_validation"
                    ) is not True:
                        raise ValueError(
                            "Base-form preservation was not verified"
                        )
                    if generated_summary.get(
                        "token_length_control_verified_train_validation"
                    ) is not True:
                        raise ValueError(
                            "Token-length control was not verified"
                        )

                    bridge_status = "N/A"
                    if k == 4:
                        bridge = verify_k4_bridge(
                            config=config,
                            source_id=source_id,
                            seed=seed,
                            new_corpus_path=corpus_path,
                            new_key_path=key_path,
                        )
                        bridge["language"] = language
                        bridge_rows.append(
                            bridge
                        )
                        bridge_status = "BRIDGE-PASS"

                    mechanism_rows.append(
                        flatten_mechanism_diagnostics(
                            source_id=source_id,
                            language=language,
                            variant_count=k,
                            seed=seed,
                            generated_summary=generated_summary,
                            master_key_sha256=generated_key[
                                "master_key_sha256"
                            ],
                            parent_master_key_sha256=generated_key[
                                "parent_phase3a3_master_key_sha256"
                            ],
                        )
                    )

                    corpus_hash = sha256_file(
                        corpus_path
                    )
                    run_dir = (
                        result_run_root
                        / source_id
                        / f"K{k}"
                        / f"seed_{seed}"
                    )
                    relations_path = (
                        run_dir / "relations.csv"
                    )
                    result_summary_path = (
                        run_dir / "summary.json"
                    )

                    if (
                        relations_path.is_file()
                        and result_summary_path.is_file()
                    ):
                        saved = json.loads(
                            result_summary_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        required = {
                            "phase3a4_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "control_corpus_sha256": corpus_hash,
                            "level2_config_sha256": level2_hash,
                        }
                        for key_name, value in required.items():
                            if saved.get(key_name) != value:
                                raise ValueError(
                                    f"Existing Phase-3A.4 result has "
                                    f"different {key_name}: {run_dir}"
                                )

                        relation_rows = [
                            normalize_relation_row(row)
                            for row in read_csv(
                                relations_path
                            )
                        ]
                        evaluation_status = "REUSED"
                    else:
                        train_units, validation_units = (
                            read_control_units(
                                corpus_path,
                                train_indices=train_indices,
                                validation_indices=validation_indices,
                                expected_line_count=len(
                                    lines
                                ),
                            )
                        )

                        relation_rows, detail = (
                            evaluate_one_control(
                                source_id=source_id,
                                language=language,
                                family=(
                                    f"base_form_suffix_extended_K{k}"
                                ),
                                control_seed=seed,
                                train_units=train_units,
                                validation_units=validation_units,
                                config=level2,
                            )
                        )

                        for row in relation_rows:
                            row["variant_count"] = k
                            row["phase3a4_mechanism"] = (
                                "base_form_suffix_extended"
                            )

                        run_dir.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        write_csv(
                            relations_path,
                            relation_rows,
                        )

                        result_summary = {
                            "schema_version": "1.0",
                            "experiment_id": config[
                                "experiment_id"
                            ],
                            "status": "exploratory",
                            "source_id": source_id,
                            "language": language,
                            "variant_count": k,
                            "seed": seed,
                            "phase3a4_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                            "level2_config_sha256": level2_hash,
                            "control_corpus_sha256": corpus_hash,
                            "master_key_sha256": generated_key[
                                "master_key_sha256"
                            ],
                            "parent_phase3a3_master_key_sha256": (
                                generated_key[
                                    "parent_phase3a3_master_key_sha256"
                                ]
                            ),
                            "k4_bridge_verified": (
                                True if k == 4 else None
                            ),
                            "relations": relation_rows,
                            "model_selection_detail": detail,
                            "reserved_test_encoded": False,
                            "reserved_test_parsed_or_scored": False,
                            "voynich_locked_test_accessed": False,
                            "interpretation": (
                                "Exploratory extension of Phase 3A.3. "
                                "Do not promote a K to confirmatory status "
                                "without a separately frozen Phase-3B protocol."
                            ),
                        }
                        result_summary_path.write_text(
                            json.dumps(
                                result_summary,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )

                        (run_dir / "SHA256SUMS").write_text(
                            "\n".join(
                                [
                                    f"{sha256_file(relations_path)}  relations.csv",
                                    f"{sha256_file(result_summary_path)}  summary.json",
                                ]
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        evaluation_status = "EVALUATED"

                    expected_relations = set(
                        config[
                            "operational_relations"
                        ]
                    )
                    if {
                        row["relation"]
                        for row in relation_rows
                    } != expected_relations:
                        raise ValueError(
                            f"Relation set mismatch: {run_dir}"
                        )

                    for row in relation_rows:
                        row["variant_count"] = k
                        row["phase3a4_mechanism"] = (
                            "base_form_suffix_extended"
                        )

                    all_relations.extend(
                        relation_rows
                    )
                    completed += 1

                    token_row = next(
                        row
                        for row in relation_rows
                        if row["relation"]
                        == "token_markov"
                    )
                    same_surface = generated_summary[
                        "validation_diagnostics"
                    ][
                        "same_surface_probability_given_same_plaintext_word"
                    ]

                    print(
                        f"  [{completed:2d}/{expected_runs}] "
                        f"K={k:<2d} seed={seed} "
                        f"{generation_status}/{evaluation_status} "
                        f"{bridge_status:<11} "
                        f"same_surface={float(same_surface):.3f} "
                        f"tokenΔ="
                        f"{float(token_row['delta_competitor_minus_trigram_bits_per_event']):+.4f}"
                    )

            print()

        if completed != expected_runs:
            raise RuntimeError(
                f"Expected {expected_runs} runs; completed {completed}"
            )

        expected_relation_rows = int(
            config["expected_relation_rows"]
        )
        if len(all_relations) != expected_relation_rows:
            raise RuntimeError(
                f"Expected {expected_relation_rows} relation rows; "
                f"found {len(all_relations)}"
            )

        if len(bridge_rows) != (
            len(config["sources"])
            * len(seeds)
        ):
            raise RuntimeError(
                "Expected exactly one K=4 bridge verification per source/seed"
            )

        if not all(
            bool(row["corpus_byte_identical"])
            for row in bridge_rows
        ):
            raise RuntimeError(
                "At least one K=4 bridge failed"
            )

        verify_master_key_invariance(
            mechanism_rows,
            variant_counts=variant_counts,
        )

        languages = [
            source["language"]
            for source in config["sources"].values()
        ]

        language_summary, global_summary = (
            summarize_relations(
                all_relations,
                variant_counts=variant_counts,
                languages=languages,
                relations=config[
                    "operational_relations"
                ],
            )
        )
        suffix_summary = summarize_suffix_diagnostics(
            mechanism_rows,
            variant_counts=variant_counts,
            languages=languages,
        )
        trajectory = trajectory_summary(
            global_summary,
            suffix_summary,
            variant_counts=variant_counts,
        )

        result_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        relation_matrix_path = (
            result_root
            / "run_relation_matrix.csv"
        )
        mechanism_path = (
            result_root
            / "mechanism_diagnostics.csv"
        )
        bridge_path = (
            result_root
            / "k4_bridge_verification.csv"
        )
        language_path = (
            result_root
            / "k_language_relation_summary.csv"
        )
        global_path = (
            result_root
            / "k_global_relation_summary.csv"
        )
        suffix_path = (
            result_root
            / "suffix_diagnostics_summary.csv"
        )
        summary_path = (
            result_root / "summary.json"
        )

        write_csv(
            relation_matrix_path,
            all_relations,
        )
        write_csv(
            mechanism_path,
            mechanism_rows,
        )
        write_csv(
            bridge_path,
            bridge_rows,
        )
        write_csv(
            language_path,
            language_summary,
        )
        write_csv(
            global_path,
            global_summary,
        )
        write_csv(
            suffix_path,
            suffix_summary,
        )

        top_summary = {
            "schema_version": "1.0",
            "experiment_id": config[
                "experiment_id"
            ],
            "status": "exploratory",
            "phase3a4_config_sha256": config_hash,
            "generator_sha256": generator_hash,
            **parent_freeze,
            "level2_config_sha256": level2_hash,
            "runs": completed,
            "relation_rows": len(
                all_relations
            ),
            "variant_counts": variant_counts,
            "coverage_fraction": 1.0,
            "base_form_preservation_verified": True,
            "token_length_base_plus_one_verified": True,
            "master_key_invariance_across_k": True,
            "suffix_choice_excludes_plaintext_token_identity": True,
            "k4_bridge_verifications": len(
                bridge_rows
            ),
            "k4_phase3a3_corpus_byte_identity_all": True,
            "trajectory": trajectory,
            "primary_prediction": config[
                "primary_prediction"
            ],
            "stop_rule": config[
                "stop_rule"
            ],
            "reserved_control_test_encoded": False,
            "reserved_control_test_parsed_or_scored": False,
            "voynich_locked_test_accessed": False,
            "no_single_similarity_score": True,
            "automatic_winner_selected": False,
            "next_step": (
                "Interpret the K=4/8/16 trajectory jointly across all five "
                "relations. If K=8 or K=16 enters a stable complete Voynich-"
                "direction hierarchy, stop exploratory extension and write a "
                "separate Phase-3B confirmatory protocol before any reserved-"
                "test use. If it plateaus or disrupts another relation, retain "
                "the negative result and do not extend K beyond 16 post hoc."
            ),
        }

        summary_path.write_text(
            json.dumps(
                top_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        primary_paths = [
            relation_matrix_path,
            mechanism_path,
            bridge_path,
            language_path,
            global_path,
            suffix_path,
            summary_path,
        ]
        (result_root / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in primary_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print("PHASE 3A.4 EXPLORATION COMPLETE")
        print("=" * 96)
        print(f"Runs:              {completed}")
        print(f"Relations:         {len(all_relations)}")
        print(f"Run matrix:        {relation_matrix_path}")
        print(f"Mechanism diag:    {mechanism_path}")
        print(f"K=4 bridge:        {bridge_path}")
        print(f"Suffix summary:    {suffix_path}")
        print(f"K summary:         {global_path}")
        print(f"Trajectory:        {summary_path}")
        print(
            "K=4 bridge:        BYTE-IDENTICAL Phase 3A.3 "
            f"({len(bridge_rows)}/{len(bridge_rows)} PASS)"
        )
        print("Base form:         PRESERVED (verified)")
        print("Master keys:       INVARIANT ACROSS K (verified)")
        print("Control test:      NOT ENCODED / NOT SCORED")
        print("Voynich test:      NOT ACCESSED")
        print()
        print("Token-Markov mean delta by K:")
        for k, value in (
            trajectory[
                "token_markov_mean_delta_by_k"
            ].items()
        ):
            print(
                f"  K={k}: {float(value):+.6f} bits/event"
            )
        print()
        print(
            "Do not extend K beyond 16 or select a confirmatory K from "
            "this runner. Phase 3A.4 is exploratory."
        )
        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A.4 INTERRUPTED. Existing hash-matched controls/results "
            "are reusable; rerun the same command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A.4 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
