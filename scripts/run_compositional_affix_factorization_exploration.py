#!/usr/bin/env python3
"""Run Phase 3A.5 compositional affix factorization exploration.

Intended repository destination:
    scripts/run_compositional_affix_factorization_exploration.py

Grid:
    4 languages x {1x16,2x8,4x4} x 5 seeds = 60 controls
    60 controls x 5 frozen predictive relations = 300 relation rows

Lineage constraints:
    * exact Phase-3A.4 TRAIN-fitted base mapping
    * exact Phase-3A.4 K=16 occurrence-level variant-index schedule
    * same abstract token-identity partition across all three factorizations

The runner uses CONTROL TRAIN/VALIDATION only. Reserved control-test lines are
not encoded, parsed, fitted, or scored. Voynich test_LOCKED.txt is not read.
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


from src.ciphers import compositional_affix_factorization
from src.ciphers.compositional_affix_factorization import (
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
    "configs/mechanisms/compositional_affix_factorization_v1.json"
)


def config_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> List[dict]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def normalize_relation_row(
    row: Mapping[str, object],
) -> dict:
    out = dict(row)

    int_fields = (
        "control_seed",
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
        if (
            key in out
            and out[key] not in ("", None)
        ):
            out[key] = int(out[key])

    for key in float_fields:
        if (
            key in out
            and out[key] not in ("", None)
        ):
            out[key] = float(out[key])

    for key in bool_fields:
        if key in out:
            out[key] = as_bool(
                out[key]
            )

    return out


def verify_parent_freeze(
    config: Mapping[str, object],
) -> dict:
    parent = config["parent_phase3a4"]

    parent_config_path = Path(
        parent["config_path"]
    )
    parent_generator_path = Path(
        parent["generator_path"]
    )

    if not parent_config_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.4 config missing: {parent_config_path}"
        )
    if not parent_generator_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.4 generator missing: {parent_generator_path}"
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
            "Frozen Phase-3A.4 config SHA differs from the Phase-3A.5 "
            f"lineage declaration: {observed_config_sha}"
        )

    if (
        observed_generator_sha
        != parent["expected_generator_sha256"]
    ):
        raise ValueError(
            "Frozen Phase-3A.4 generator SHA differs from the Phase-3A.5 "
            f"lineage declaration: {observed_generator_sha}"
        )

    return {
        "parent_phase3a4_config_sha256": observed_config_sha,
        "parent_phase3a4_generator_sha256": observed_generator_sha,
    }


def parent_abstract_identity_sha256(
    *,
    corpus_path: Path,
    key_path: Path,
    line_indices: Sequence[int],
) -> str:
    """Hash Phase-3A.4 K=16 base + suffix index without parsing reserved rows."""
    selected = set(
        int(x) for x in line_indices
    )

    key = json.loads(
        key_path.read_text(
            encoding="utf-8"
        )
    )
    suffixes = list(
        key["master_suffixes"]
    )
    if len(suffixes) != 16:
        raise ValueError(
            f"Parent K=16 key does not contain 16 suffixes: {key_path}"
        )

    suffix_to_index = {
        suffix: index
        for index, suffix in enumerate(
            suffixes
        )
    }

    digest = sha256()
    found = set()

    with corpus_path.open(
        encoding="utf-8",
    ) as handle:
        for line_index, raw in enumerate(
            handle,
            start=1,
        ):
            if line_index not in selected:
                continue

            payload = json.loads(raw)
            if "tokens" not in payload:
                raise ValueError(
                    "Selected parent TRAIN/VALIDATION row is opaque"
                )

            found.add(
                line_index
            )

            for token_index, token in enumerate(
                payload["tokens"]
            ):
                token = tuple(token)
                if len(token) < 2:
                    raise ValueError(
                        "Parent token is too short"
                    )

                suffix = token[-1]
                try:
                    variant_index = suffix_to_index[
                        suffix
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown parent K=16 suffix {suffix!r}"
                    ) from exc

                base = token[:-1]
                record = [
                    int(line_index),
                    int(token_index),
                    list(base),
                    int(variant_index),
                ]
                digest.update(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                digest.update(b"\n")

    if found != selected:
        missing = sorted(
            selected - found
        )
        raise ValueError(
            f"Parent corpus missing selected lines: {missing[:10]}"
        )

    return digest.hexdigest()


def verify_parent_lineage(
    *,
    config: Mapping[str, object],
    source_id: str,
    language: str,
    seed: int,
    factorization: str,
    new_key: Mapping[str, object],
    new_summary: Mapping[str, object],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> dict:
    parent_root = Path(
        config[
            "parent_phase3a4"
        ]["control_root"]
    )
    parent_dir = (
        parent_root
        / source_id
        / "K16"
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
            f"Parent Phase-3A.4 K=16 corpus missing: {parent_corpus_path}"
        )
    if not parent_key_path.is_file():
        raise FileNotFoundError(
            f"Parent Phase-3A.4 K=16 key missing: {parent_key_path}"
        )

    parent_key = json.loads(
        parent_key_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        new_key["mono_forward"]
        != parent_key["mono_forward"]
    ):
        raise ValueError(
            f"Base-map lineage failure: {source_id}/seed={seed}/{factorization}"
        )

    if (
        new_key[
            "parent_phase3a4_master_key_sha256"
        ]
        != parent_key["master_key_sha256"]
    ):
        raise ValueError(
            f"Parent master-key lineage failure: "
            f"{source_id}/seed={seed}/{factorization}"
        )

    parent_train_fingerprint = (
        parent_abstract_identity_sha256(
            corpus_path=parent_corpus_path,
            key_path=parent_key_path,
            line_indices=train_indices,
        )
    )
    parent_validation_fingerprint = (
        parent_abstract_identity_sha256(
            corpus_path=parent_corpus_path,
            key_path=parent_key_path,
            line_indices=validation_indices,
        )
    )

    new_train_fingerprint = (
        new_summary[
            "train_diagnostics"
        ][
            "abstract_token_identity_sha256"
        ]
    )
    new_validation_fingerprint = (
        new_summary[
            "validation_diagnostics"
        ][
            "abstract_token_identity_sha256"
        ]
    )

    if (
        new_train_fingerprint
        != parent_train_fingerprint
    ):
        raise ValueError(
            f"TRAIN variant-schedule lineage failure: "
            f"{source_id}/seed={seed}/{factorization}"
        )

    if (
        new_validation_fingerprint
        != parent_validation_fingerprint
    ):
        raise ValueError(
            f"VALIDATION variant-schedule lineage failure: "
            f"{source_id}/seed={seed}/{factorization}"
        )

    return {
        "source_id": source_id,
        "language": language,
        "seed": int(seed),
        "factorization": factorization,
        "base_mapping_identical_to_phase3a4": True,
        "parent_master_key_identical": True,
        "train_abstract_identity_sha256": new_train_fingerprint,
        "validation_abstract_identity_sha256": new_validation_fingerprint,
        "train_variant_schedule_identical_to_phase3a4_k16": True,
        "validation_variant_schedule_identical_to_phase3a4_k16": True,
        "parent_k16_corpus_sha256": sha256_file(
            parent_corpus_path
        ),
    }


def flatten_mechanism_diagnostics(
    *,
    source_id: str,
    language: str,
    factorization: str,
    seed: int,
    generated_summary: Mapping[str, object],
    master_key_sha256: str,
    parent_master_key_sha256: str,
) -> dict:
    train = generated_summary[
        "train_diagnostics"
    ]
    validation = generated_summary[
        "validation_diagnostics"
    ]

    row = {
        "source_id": source_id,
        "language": language,
        "factorization": factorization,
        "seed": int(seed),
        "master_key_sha256": master_key_sha256,
        "parent_phase3a4_master_key_sha256": (
            parent_master_key_sha256
        ),
        "reserved_test_encoded": bool(
            generated_summary[
                "reserved_test_encoded"
            ]
        ),
    }

    for prefix, diag in (
        ("train", train),
        ("validation", validation),
    ):
        for key, value in diag.items():
            row[
                f"{prefix}_{key}"
            ] = value

    return row


def verify_factorization_invariants(
    diagnostic_rows: Sequence[
        Mapping[str, object]
    ],
    *,
    factorization_labels: Sequence[str],
) -> None:
    """Require one parent key and one abstract token partition across conditions."""
    groups = {}

    for row in diagnostic_rows:
        group_key = (
            row["source_id"],
            int(row["seed"]),
        )
        groups.setdefault(
            group_key,
            [],
        ).append(row)

    for (
        source_id,
        seed,
    ), rows in groups.items():
        labels = sorted(
            row["factorization"]
            for row in rows
        )
        if labels != sorted(
            factorization_labels
        ):
            raise ValueError(
                f"{source_id}/seed={seed}: expected factorizations "
                f"{list(factorization_labels)}, found {labels}"
            )

        master_keys = {
            row["master_key_sha256"]
            for row in rows
        }
        if len(master_keys) != 1:
            raise ValueError(
                f"{source_id}/seed={seed}: Phase-3A.5 master key "
                "changed across factorizations"
            )

        parent_keys = {
            row[
                "parent_phase3a4_master_key_sha256"
            ]
            for row in rows
        }
        if len(parent_keys) != 1:
            raise ValueError(
                f"{source_id}/seed={seed}: parent Phase-3A.4 key "
                "changed across factorizations"
            )

        for split_name in (
            "train",
            "validation",
        ):
            fingerprints = {
                row[
                    f"{split_name}_abstract_token_identity_sha256"
                ]
                for row in rows
            }
            if len(fingerprints) != 1:
                raise ValueError(
                    f"{source_id}/seed={seed}: abstract token identity "
                    f"changed across factorizations on {split_name}"
                )

            recurrence = {
                row[
                    f"{split_name}_exact_cipher_token_recurrence_fraction"
                ]
                for row in rows
            }
            if len(recurrence) != 1:
                raise ValueError(
                    f"{source_id}/seed={seed}: exact-token recurrence "
                    f"changed across factorizations on {split_name}"
                )

            unique_tokens = {
                row[
                    f"{split_name}_unique_cipher_tokens"
                ]
                for row in rows
            }
            if len(unique_tokens) != 1:
                raise ValueError(
                    f"{source_id}/seed={seed}: unique exact-token count "
                    f"changed across factorizations on {split_name}"
                )


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    factorization_labels: Sequence[str],
    languages: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], List[dict]]:
    language_rows: List[dict] = []
    global_rows: List[dict] = []

    for factorization in factorization_labels:
        for relation in relations:
            global_group = [
                row
                for row in rows
                if row["factorization"]
                == factorization
                and row["relation"]
                == relation
            ]

            if len(global_group) != 20:
                raise ValueError(
                    f"{factorization}/{relation}: expected 20 runs; "
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
                    "factorization": factorization,
                    "relation": relation,
                    "runs": len(global_group),
                    "mean_delta_bits_per_event": mean(
                        deltas
                    ),
                    "median_delta_bits_per_event": median(
                        deltas
                    ),
                    "sd_delta_bits_per_event": pstdev(
                        deltas
                    ),
                    "min_delta_bits_per_event": min(
                        deltas
                    ),
                    "max_delta_bits_per_event": max(
                        deltas
                    ),
                    "positive_direction_runs": sum(
                        bool(
                            row[
                                "direction_match_voynich"
                            ]
                        )
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
                        "no aggregate similarity score."
                    ),
                }
            )

            for language in languages:
                group = [
                    row
                    for row in global_group
                    if row["language"]
                    == language
                ]
                if len(group) != 5:
                    raise ValueError(
                        f"{factorization}/{relation}/{language}: "
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

                strong_seeds = sum(
                    bool(
                        row[
                            "strong_direction_match_voynich"
                        ]
                    )
                    for row in group
                )

                language_rows.append(
                    {
                        "factorization": factorization,
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
                        "strong_direction_seeds": strong_seeds,
                        "language_relation_replicated": (
                            strong_seeds >= 4
                        ),
                    }
                )

    return language_rows, global_rows


def summarize_replication(
    language_rows: Sequence[
        Mapping[str, object]
    ],
    *,
    factorization_labels: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], dict]:
    rows = []
    full_hierarchy = {}

    for factorization in factorization_labels:
        relation_results = {}

        for relation in relations:
            group = [
                row
                for row in language_rows
                if row["factorization"]
                == factorization
                and row["relation"]
                == relation
            ]

            if len(group) != 4:
                raise ValueError(
                    f"{factorization}/{relation}: expected four language rows"
                )

            replicated_languages = [
                row["language"]
                for row in group
                if bool(
                    row[
                        "language_relation_replicated"
                    ]
                )
            ]
            robust = (
                len(
                    replicated_languages
                )
                >= 3
            )

            rows.append(
                {
                    "factorization": factorization,
                    "relation": relation,
                    "languages_replicated": len(
                        replicated_languages
                    ),
                    "replicated_languages": ";".join(
                        replicated_languages
                    ),
                    "relation_robust_3_of_4": robust,
                }
            )
            relation_results[
                relation
            ] = robust

        full_hierarchy[
            factorization
        ] = all(
            relation_results.get(
                relation,
                False,
            )
            for relation in relations
        )

    return rows, full_hierarchy


def summarize_mechanism(
    rows: Sequence[Mapping[str, object]],
    *,
    factorization_labels: Sequence[str],
    languages: Sequence[str],
) -> List[dict]:
    output = []

    fields = [
        "validation_token_coverage_fraction",
        "validation_base_form_recovery_accuracy",
        "validation_base_form_preservation_fraction",
        "validation_token_length_plus_two_accuracy",
        "validation_phase3a4_k16_variant_schedule_accuracy",
        "validation_mean_realized_surface_forms_per_plaintext_word_present",
        "validation_same_surface_probability_given_same_plaintext_word",
        "validation_exact_cipher_token_recurrence_fraction",
        "validation_slot_a_entropy_bits",
        "validation_slot_b_entropy_bits",
        "validation_joint_affix_pair_entropy_bits",
        "validation_abstract_variant_index_entropy_bits",
    ]

    for factorization in factorization_labels:
        for language in languages:
            group = [
                row
                for row in rows
                if row[
                    "factorization"
                ] == factorization
                and row[
                    "language"
                ] == language
            ]

            if len(group) != 5:
                raise ValueError(
                    f"{factorization}/{language}: expected 5 diagnostic rows"
                )

            result = {
                "factorization": factorization,
                "language": language,
                "seeds": 5,
            }

            for field in fields:
                values = [
                    float(
                        item[field]
                    )
                    for item in group
                    if item[field]
                    not in ("", None)
                ]
                result[
                    f"mean_{field}"
                ] = (
                    mean(values)
                    if values
                    else None
                )

            output.append(
                result
            )

    return output


def trajectory_summary(
    global_rows: Sequence[Mapping[str, object]],
    *,
    factorization_labels: Sequence[str],
    full_hierarchy: Mapping[str, bool],
) -> dict:
    token_rows = {
        row["factorization"]: row
        for row in global_rows
        if row["relation"]
        == "token_markov"
    }

    token_means = [
        float(
            token_rows[
                factorization
            ][
                "mean_delta_bits_per_event"
            ]
        )
        for factorization in factorization_labels
    ]

    return {
        "token_markov_mean_delta_by_factorization": {
            factorization: token_means[
                index
            ]
            for index, factorization in enumerate(
                factorization_labels
            )
        },
        "token_markov_prediction_order_non_decreasing": all(
            right >= left
            for left, right in zip(
                token_means,
                token_means[1:],
            )
        ),
        "full_inherited_level2_hierarchy_by_factorization": dict(
            full_hierarchy
        ),
        "interpretation": (
            "Exploratory development result only. Even a factorization that "
            "reaches the inherited Level-2 hierarchy must be frozen in a "
            "separate Phase-3B protocol before reserved-test use."
        ),
    }


def require_level2_compatibility(
    phase3a5: Mapping[str, object],
    level2: Mapping[str, object],
) -> None:
    if (
        level2.get("freeze_id")
        != "level2-predictive-controls-v1"
    ):
        raise ValueError(
            "Unexpected Level-2 evaluator config freeze_id"
        )

    if (
        list(
            level2[
                "operational_relations"
            ]
        )
        != list(
            phase3a5[
                "operational_relations"
            ]
        )
    ):
        raise ValueError(
            "Phase-3A.5 operational relations differ from Level 2"
        )

    for source_id, source in phase3a5[
        "sources"
    ].items():
        level2_source = level2[
            "source_units"
        ].get(source_id)

        if level2_source is None:
            raise ValueError(
                f"Source absent from Level-2 config: {source_id}"
            )

        if (
            level2_source[
                "source_sha256"
            ]
            != source[
                "source_sha256"
            ]
        ):
            raise ValueError(
                f"Source hash differs from Level 2: {source_id}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate Phase-3A.5 compositional affix "
            "factorization controls on frozen Level-2 TRAIN/VALIDATION."
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
            config.get(
                "experiment_id"
            )
            != "compositional-affix-factorization-exploration-v1"
        ):
            raise ValueError(
                "Unexpected Phase-3A.5 experiment_id"
            )
        if (
            config.get(
                "status"
            )
            != "exploratory"
        ):
            raise ValueError(
                "Phase 3A.5 must remain explicitly exploratory"
            )

        parent_freeze = verify_parent_freeze(
            config
        )

        specs = config[
            "mechanism"
        ][
            "factorizations"
        ]
        factorization_labels = [
            item["label"]
            for item in specs
        ]
        if (
            factorization_labels
            != [
                "1x16",
                "2x8",
                "4x4",
            ]
        ):
            raise ValueError(
                "Factorization grid must be exactly [1x16,2x8,4x4]"
            )

        for spec in specs:
            if (
                int(
                    spec[
                        "slot_a_states"
                    ]
                )
                * int(
                    spec[
                        "slot_b_states"
                    ]
                )
                != 16
            ):
                raise ValueError(
                    f"Factorization does not contain 16 variants: {spec}"
                )

        mechanism = config[
            "mechanism"
        ]
        if (
            int(
                mechanism[
                    "whole_token_variant_count"
                ]
            )
            != 16
        ):
            raise ValueError(
                "whole_token_variant_count must be exactly 16"
            )
        if (
            int(
                mechanism[
                    "suffix_length_glyphs"
                ]
            )
            != 2
        ):
            raise ValueError(
                "suffix_length_glyphs must be exactly 2"
            )

        seeds = [
            int(x)
            for x in config[
                "seeds"
            ]
        ]
        if len(seeds) != 5:
            raise ValueError(
                "Phase 3A.5 requires exactly five frozen seeds"
            )

        expected_runs = int(
            config[
                "expected_runs"
            ]
        )
        calculated_runs = (
            len(
                config[
                    "sources"
                ]
            )
            * len(
                factorization_labels
            )
            * len(seeds)
        )
        if (
            expected_runs
            != calculated_runs
        ):
            raise ValueError(
                "expected_runs does not match source/factorization/seed cardinality"
            )

        generator_hash = sha256_file(
            Path(
                compositional_affix_factorization.__file__
            )
        )

        level2_config_path = Path(
            config[
                "level2_config_path"
            ]
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
            config[
                "source_root"
            ]
        )
        split_root = Path(
            config[
                "level2_split_root"
            ]
        )
        control_root = Path(
            config[
                "control_output_root"
            ]
        )
        result_root = Path(
            config[
                "result_output_root"
            ]
        )
        result_run_root = (
            result_root
            / "runs"
        )
        result_run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_relations: List[dict] = []
        mechanism_rows: List[dict] = []
        lineage_rows: List[dict] = []
        completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A.5 — COMPOSITIONAL AFFIX FACTORIZATION"
        )
        print("=" * 96)
        print(
            f"Experiment:       {config['experiment_id']}"
        )
        print(
            f"Status:           {config['status'].upper()}"
        )
        print(
            f"Config SHA:       {config_hash}"
        )
        print(
            f"Generator SHA:    {generator_hash}"
        )
        print(
            "Parent 3A.4 cfg:  "
            f"{parent_freeze['parent_phase3a4_config_sha256']}"
        )
        print(
            "Parent 3A.4 gen:  "
            f"{parent_freeze['parent_phase3a4_generator_sha256']}"
        )
        print(
            f"Level-2 config:   {level2_hash}"
        )
        print(
            "Factorizations:   1x16, 2x8, 4x4"
        )
        print(
            "Variants:         16 in every condition"
        )
        print(
            "Suffix length:    2 glyphs in every condition"
        )
        print(
            "Coverage:         100% TRAIN/VALIDATION tokens"
        )
        print(
            "Variant schedule: exact Phase-3A.4 K=16 schedule"
        )
        print(
            "Expected runs:    "
            f"{expected_runs}"
        )
        print(
            "Control test:     NOT ENCODED / NOT SCORED"
        )
        print(
            "Voynich test:     NOT ACCESSED"
        )
        print()

        for source_id, source_cfg in config[
            "sources"
        ].items():
            language = source_cfg[
                "language"
            ]
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
                != source_cfg[
                    "source_sha256"
                ]
            ):
                raise ValueError(
                    f"{source_id}: frozen source SHA mismatch"
                )

            lines = read_source_lines(
                source_path
            )

            if (
                len(lines)
                != int(
                    source_cfg[
                        "expected_lines"
                    ]
                )
            ):
                raise ValueError(
                    f"{source_id}: expected "
                    f"{source_cfg['expected_lines']} lines, "
                    f"found {len(lines)}"
                )

            if (
                token_count(lines)
                != int(
                    source_cfg[
                        "expected_tokens"
                    ]
                )
            ):
                raise ValueError(
                    f"{source_id}: expected "
                    f"{source_cfg['expected_tokens']} tokens, "
                    f"found {token_count(lines)}"
                )

            split_path = (
                split_root
                / f"{source_id}.json"
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
                split.get(
                    "source_sha256"
                )
                != source_hash
            ):
                raise ValueError(
                    f"{source_id}: Level-2 split source hash mismatch"
                )

            train_indices = [
                int(x)
                for x in split[
                    "train"
                ]
            ]
            validation_indices = [
                int(x)
                for x in split[
                    "validation"
                ]
            ]
            reserved_indices = [
                int(x)
                for x in split[
                    "reserved_test"
                ]
            ]

            print(
                f"[{language}] {source_id}: "
                f"train={len(train_indices)} "
                f"validation={len(validation_indices)} "
                f"reserved={len(reserved_indices)}"
            )

            for factorization in factorization_labels:
                for seed in seeds:
                    safe_factor = (
                        "F"
                        + factorization
                    )
                    control_dir = (
                        control_root
                        / source_id
                        / safe_factor
                        / f"seed_{seed}"
                    )
                    corpus_path = (
                        control_dir
                        / "corpus.jsonl"
                    )
                    control_summary_path = (
                        control_dir
                        / "summary.json"
                    )
                    key_path = (
                        control_dir
                        / "key.json"
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
                            "phase3a5_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                            "parent_phase3a4_generator_sha256": (
                                parent_freeze[
                                    "parent_phase3a4_generator_sha256"
                                ]
                            ),
                        }
                        for (
                            key_name,
                            value,
                        ) in required.items():
                            if (
                                provenance.get(
                                    key_name
                                )
                                != value
                            ):
                                raise ValueError(
                                    f"Existing generated control has "
                                    f"different {key_name}: {control_dir}"
                                )

                        if (
                            existing[
                                "factorization"
                            ]
                            != factorization
                            or int(
                                existing[
                                    "seed"
                                ]
                            )
                            != seed
                        ):
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
                            factorization=factorization,
                        )

                        provenance = {
                            "phase3a5_config_path": str(
                                args.config
                            ),
                            "phase3a5_config_sha256": config_hash,
                            "generator_path": str(
                                Path(
                                    compositional_affix_factorization.__file__
                                )
                            ),
                            "generator_sha256": generator_hash,
                            "parent_phase3a4_config_sha256": (
                                parent_freeze[
                                    "parent_phase3a4_config_sha256"
                                ]
                            ),
                            "parent_phase3a4_generator_sha256": (
                                parent_freeze[
                                    "parent_phase3a4_generator_sha256"
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

                    for required_flag in (
                        "round_trip_verified_train_validation",
                        "base_form_preservation_verified_train_validation",
                        "token_length_control_verified_train_validation",
                        "phase3a4_k16_variant_schedule_verified_train_validation",
                    ):
                        if (
                            generated_summary.get(
                                required_flag
                            )
                            is not True
                        ):
                            raise ValueError(
                                f"{required_flag} was not verified"
                            )

                    if (
                        generated_summary.get(
                            "reserved_test_encoded"
                        )
                        is not False
                    ):
                        raise ValueError(
                            "Reserved test must not be encoded"
                        )

                    lineage = verify_parent_lineage(
                        config=config,
                        source_id=source_id,
                        language=language,
                        seed=seed,
                        factorization=factorization,
                        new_key=generated_key,
                        new_summary=generated_summary,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                    )
                    lineage_rows.append(
                        lineage
                    )

                    mechanism_rows.append(
                        flatten_mechanism_diagnostics(
                            source_id=source_id,
                            language=language,
                            factorization=factorization,
                            seed=seed,
                            generated_summary=generated_summary,
                            master_key_sha256=generated_key[
                                "master_key_sha256"
                            ],
                            parent_master_key_sha256=generated_key[
                                "parent_phase3a4_master_key_sha256"
                            ],
                        )
                    )

                    corpus_hash = sha256_file(
                        corpus_path
                    )
                    run_dir = (
                        result_run_root
                        / source_id
                        / safe_factor
                        / f"seed_{seed}"
                    )
                    relations_path = (
                        run_dir
                        / "relations.csv"
                    )
                    result_summary_path = (
                        run_dir
                        / "summary.json"
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
                            "phase3a5_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "control_corpus_sha256": corpus_hash,
                            "level2_config_sha256": level2_hash,
                        }
                        for (
                            key_name,
                            value,
                        ) in required.items():
                            if (
                                saved.get(
                                    key_name
                                )
                                != value
                            ):
                                raise ValueError(
                                    f"Existing Phase-3A.5 result has "
                                    f"different {key_name}: {run_dir}"
                                )

                        relation_rows = [
                            normalize_relation_row(
                                row
                            )
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
                                    "compositional_affix_"
                                    + factorization
                                ),
                                control_seed=seed,
                                train_units=train_units,
                                validation_units=validation_units,
                                config=level2,
                            )
                        )

                        for row in relation_rows:
                            row[
                                "factorization"
                            ] = factorization
                            row[
                                "phase3a5_mechanism"
                            ] = (
                                "compositional_affix_factorization"
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
                            "factorization": factorization,
                            "seed": seed,
                            "phase3a5_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                            "level2_config_sha256": level2_hash,
                            "control_corpus_sha256": corpus_hash,
                            "master_key_sha256": generated_key[
                                "master_key_sha256"
                            ],
                            "parent_phase3a4_master_key_sha256": (
                                generated_key[
                                    "parent_phase3a4_master_key_sha256"
                                ]
                            ),
                            "parent_lineage_verified": True,
                            "relations": relation_rows,
                            "model_selection_detail": detail,
                            "reserved_test_encoded": False,
                            "reserved_test_parsed_or_scored": False,
                            "voynich_locked_test_accessed": False,
                            "interpretation": (
                                "Exploratory representation-geometry result. "
                                "Do not promote a factorization to confirmatory "
                                "status without a separately frozen Phase-3B protocol."
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

                        (
                            run_dir
                            / "SHA256SUMS"
                        ).write_text(
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
                        row[
                            "relation"
                        ]
                        for row in relation_rows
                    } != expected_relations:
                        raise ValueError(
                            f"Relation set mismatch: {run_dir}"
                        )

                    for row in relation_rows:
                        row[
                            "factorization"
                        ] = factorization
                        row[
                            "phase3a5_mechanism"
                        ] = (
                            "compositional_affix_factorization"
                        )

                    all_relations.extend(
                        relation_rows
                    )
                    completed += 1

                    token_row = next(
                        row
                        for row in relation_rows
                        if row[
                            "relation"
                        ]
                        == "token_markov"
                    )
                    same_surface = generated_summary[
                        "validation_diagnostics"
                    ][
                        "same_surface_probability_given_same_plaintext_word"
                    ]

                    print(
                        f"  [{completed:2d}/{expected_runs}] "
                        f"F={factorization:<4} seed={seed} "
                        f"{generation_status}/{evaluation_status} "
                        f"LINEAGE-PASS "
                        f"same_surface={float(same_surface):.3f} "
                        f"tokenΔ="
                        f"{float(token_row['delta_competitor_minus_trigram_bits_per_event']):+.4f}"
                    )

            print()

        if (
            completed
            != expected_runs
        ):
            raise RuntimeError(
                f"Expected {expected_runs} runs; completed {completed}"
            )

        expected_relation_rows = int(
            config[
                "expected_relation_rows"
            ]
        )
        if (
            len(
                all_relations
            )
            != expected_relation_rows
        ):
            raise RuntimeError(
                f"Expected {expected_relation_rows} relation rows; "
                f"found {len(all_relations)}"
            )

        if len(lineage_rows) != expected_runs:
            raise RuntimeError(
                "Expected one parent-lineage verification per run"
            )

        verify_factorization_invariants(
            mechanism_rows,
            factorization_labels=factorization_labels,
        )

        languages = [
            source[
                "language"
            ]
            for source in config[
                "sources"
            ].values()
        ]

        (
            language_summary,
            global_summary,
        ) = summarize_relations(
            all_relations,
            factorization_labels=factorization_labels,
            languages=languages,
            relations=config[
                "operational_relations"
            ],
        )

        (
            replication_rows,
            full_hierarchy,
        ) = summarize_replication(
            language_summary,
            factorization_labels=factorization_labels,
            relations=config[
                "operational_relations"
            ],
        )

        mechanism_summary = (
            summarize_mechanism(
                mechanism_rows,
                factorization_labels=factorization_labels,
                languages=languages,
            )
        )

        trajectory = trajectory_summary(
            global_summary,
            factorization_labels=factorization_labels,
            full_hierarchy=full_hierarchy,
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
        lineage_path = (
            result_root
            / "parent_lineage_verification.csv"
        )
        mechanism_summary_path = (
            result_root
            / "factorization_diagnostics_summary.csv"
        )
        language_path = (
            result_root
            / "factorization_language_relation_summary.csv"
        )
        global_path = (
            result_root
            / "factorization_global_relation_summary.csv"
        )
        replication_path = (
            result_root
            / "factorization_replication_summary.csv"
        )
        summary_path = (
            result_root
            / "summary.json"
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
            lineage_path,
            lineage_rows,
        )
        write_csv(
            mechanism_summary_path,
            mechanism_summary,
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
            replication_path,
            replication_rows,
        )

        top_summary = {
            "schema_version": "1.0",
            "experiment_id": config[
                "experiment_id"
            ],
            "status": "exploratory",
            "phase3a5_config_sha256": config_hash,
            "generator_sha256": generator_hash,
            **parent_freeze,
            "level2_config_sha256": level2_hash,
            "runs": completed,
            "relation_rows": len(
                all_relations
            ),
            "factorizations": factorization_labels,
            "whole_token_variant_count": 16,
            "suffix_length_glyphs": 2,
            "coverage_fraction": 1.0,
            "base_form_preservation_verified": True,
            "phase3a4_k16_variant_schedule_verified": True,
            "abstract_exact_token_partition_invariant_across_factorizations": True,
            "parent_lineage_verifications": len(
                lineage_rows
            ),
            "trajectory": trajectory,
            "inherited_replication_rule": config[
                "inherited_replication_rule"
            ],
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
                "If a factorization reaches the inherited full five-relation "
                "hierarchy on development validation, stop mechanism tuning and "
                "write/freeze a separate Phase-3B confirmatory protocol before "
                "reserved-test access. Otherwise retain the result and design a "
                "new exploratory mechanism without adding post-hoc factorizations."
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
            lineage_path,
            mechanism_summary_path,
            language_path,
            global_path,
            replication_path,
            summary_path,
        ]
        (
            result_root
            / "SHA256SUMS"
        ).write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in primary_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print(
            "PHASE 3A.5 EXPLORATION COMPLETE"
        )
        print("=" * 96)
        print(
            f"Runs:              {completed}"
        )
        print(
            f"Relations:         {len(all_relations)}"
        )
        print(
            f"Run matrix:        {relation_matrix_path}"
        )
        print(
            f"Mechanism diag:    {mechanism_path}"
        )
        print(
            f"Parent lineage:    {lineage_path}"
        )
        print(
            f"Replication:       {replication_path}"
        )
        print(
            f"Trajectory:        {summary_path}"
        )
        print(
            f"Lineage checks:    {len(lineage_rows)}/{len(lineage_rows)} PASS"
        )
        print(
            "Exact token partition: INVARIANT ACROSS FACTORIZATIONS"
        )
        print(
            "Control test:      NOT ENCODED / NOT SCORED"
        )
        print(
            "Voynich test:      NOT ACCESSED"
        )
        print()
        print(
            "Token-Markov mean delta by factorization:"
        )
        for (
            factorization,
            value,
        ) in trajectory[
            "token_markov_mean_delta_by_factorization"
        ].items():
            print(
                f"  {factorization}: {float(value):+.6f} bits/event"
            )
        print()
        print(
            "Inherited full hierarchy by factorization:"
        )
        for (
            factorization,
            value,
        ) in full_hierarchy.items():
            print(
                f"  {factorization}: {'YES' if value else 'NO'}"
            )
        print()
        print(
            "Do not add post-hoc factorizations or access reserved data "
            "from this runner. Phase 3A.5 is exploratory."
        )
        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A.5 INTERRUPTED. Existing hash-matched controls/results "
            "are reusable; rerun the same command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A.5 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
