#!/usr/bin/env python3
"""Run Phase 3A.6 recurrence-avoiding surface-allocation exploration.

Intended repository destination:
    scripts/run_recurrence_avoiding_surface_allocation_exploration.py

Grid:
    4 historical-language controls x 5 frozen seeds = 20 new controls
    20 controls x 5 frozen Level-2 relations = 100 relation outcomes

The intervention keeps the Phase-3A.4 K=16 base encoding, suffix inventory,
coverage, and token length fixed. It changes only allocation from independent
with-replacement occurrence hashing to a deterministic per-lexical-base
without-replacement 16-cycle.

Reserved control test is not encoded or scored. Voynich test_LOCKED is not
read.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.ciphers import recurrence_avoiding_surface_allocation as mechanism
from src.ciphers.recurrence_avoiding_surface_allocation import (
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
    "configs/mechanisms/recurrence_avoiding_surface_allocation_v1.json"
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
    output = dict(row)

    for key in (
        "control_seed",
        "selected_hyperparameter",
        "bootstrap_units",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        if (
            key in output
            and output[key]
            not in ("", None)
        ):
            output[key] = int(
                output[key]
            )

    for key in (
        "competitor_bits_per_event",
        "trigram_bits_per_event",
        "delta_competitor_minus_trigram_bits_per_event",
        "relative_nll_reduction_trigram_vs_competitor",
        "ci_95_lower",
        "ci_95_upper",
        "bootstrap_probability_trigram_better",
        "bootstrap_probability_competitor_better",
    ):
        if (
            key in output
            and output[key]
            not in ("", None)
        ):
            output[key] = float(
                output[key]
            )

    for key in (
        "direction_match_voynich",
        "strong_direction_match_voynich",
    ):
        if key in output:
            output[key] = as_bool(
                output[key]
            )

    return output


def verify_frozen_inputs(
    config: Mapping[str, object],
) -> dict:
    parent = config[
        "parent_phase3a4"
    ]
    diagnostic = config[
        "diagnostic_lineage"
    ]

    paths_and_expected = [
        (
            "parent_phase3a4_config",
            Path(
                parent[
                    "config_path"
                ]
            ),
            parent[
                "expected_config_sha256"
            ],
        ),
        (
            "parent_phase3a4_generator",
            Path(
                parent[
                    "generator_path"
                ]
            ),
            parent[
                "expected_generator_sha256"
            ],
        ),
        (
            "level2_config",
            Path(
                config[
                    "level2_config_path"
                ]
            ),
            config[
                "expected_level2_config_sha256"
            ],
        ),
        (
            "phase3ad1_config",
            Path(
                diagnostic[
                    "phase3ad1_config_path"
                ]
            ),
            diagnostic[
                "expected_phase3ad1_config_sha256"
            ],
        ),
        (
            "phase3ad1_analysis",
            Path(
                diagnostic[
                    "phase3ad1_analysis_path"
                ]
            ),
            diagnostic[
                "expected_phase3ad1_analysis_sha256"
            ],
        ),
    ]

    observed = {}

    for (
        name,
        path,
        expected_sha,
    ) in paths_and_expected:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required frozen input missing: {path}"
            )
        digest = sha256_file(
            path
        )
        if digest != expected_sha:
            raise ValueError(
                f"{name} SHA mismatch: expected {expected_sha}, observed {digest}"
            )
        observed[
            f"{name}_sha256"
        ] = digest

    d1_summary_path = (
        Path(
            diagnostic[
                "phase3ad1_result_root"
            ]
        )
        / "summary.json"
    )
    if not d1_summary_path.is_file():
        raise FileNotFoundError(
            f"Phase-3A.D1 result summary missing: {d1_summary_path}"
        )

    d1_summary = json.loads(
        d1_summary_path.read_text(
            encoding="utf-8"
        )
    )

    if int(
        d1_summary.get(
            "parity_runs",
            -1,
        )
    ) != int(
        diagnostic[
            "required_parity_runs"
        ]
    ):
        raise ValueError(
            "Phase-3A.D1 summary does not report the required parity-run count"
        )

    if int(
        d1_summary.get(
            "parity_runs_passed",
            -1,
        )
    ) != int(
        diagnostic[
            "required_parity_passes"
        ]
    ):
        raise ValueError(
            "Phase-3A.D1 aggregate parity was not fully verified"
        )

    relation_sha = d1_summary.get(
        "phase3a5_relation_matrix_sha256"
    )
    expected_relation_sha = diagnostic[
        "phase3a5_relation_matrix_sha256_observed_in_diagnostic"
    ]
    if relation_sha != expected_relation_sha:
        raise ValueError(
            "Phase-3A.D1 summary points to a different Phase-3A.5 relation matrix"
        )

    observed[
        "phase3ad1_summary_sha256"
    ] = sha256_file(
        d1_summary_path
    )
    observed[
        "phase3ad1_parity_runs"
    ] = int(
        d1_summary[
            "parity_runs"
        ]
    )
    observed[
        "phase3ad1_parity_passes"
    ] = int(
        d1_summary[
            "parity_runs_passed"
        ]
    )

    level2 = json.loads(
        Path(
            config[
                "level2_config_path"
            ]
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        level2.get(
            "freeze_id"
        )
        != "level2-predictive-controls-v1"
    ):
        raise ValueError(
            "Unexpected Level-2 evaluator freeze_id"
        )

    if list(
        level2[
            "operational_relations"
        ]
    ) != list(
        config[
            "operational_relations"
        ]
    ):
        raise ValueError(
            "Phase-3A.6 relation set differs from frozen Level 2"
        )

    return observed


def _unit_dict(
    units: Sequence[tuple],
) -> dict:
    output = {}
    for line_index, tokens in units:
        line_index = int(
            line_index
        )
        if line_index in output:
            raise ValueError(
                f"Duplicate line index in units: {line_index}"
            )
        output[
            line_index
        ] = tuple(
            tuple(token)
            for token in tokens
        )
    return output


def exact_seen_in_train_fraction(
    train_units: Sequence[tuple],
    validation_units: Sequence[tuple],
) -> tuple[int, int, float]:
    train_tokens = {
        tuple(token)
        for _, line in train_units
        for token in line
    }

    validation_tokens = [
        tuple(token)
        for _, line in validation_units
        for token in line
    ]

    seen = sum(
        token in train_tokens
        for token in validation_tokens
    )
    total = len(
        validation_tokens
    )

    return (
        int(seen),
        int(total),
        (
            seen / total
            if total
            else 0.0
        ),
    )


def verify_parent_lineage_and_collisions(
    *,
    source_id: str,
    language: str,
    seed: int,
    config: Mapping[str, object],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    expected_line_count: int,
    new_corpus_path: Path,
    new_key_path: Path,
) -> tuple[dict, dict]:
    parent_root = Path(
        config[
            "parent_phase3a4"
        ][
            "control_root"
        ]
    )
    parent_condition = config[
        "parent_phase3a4"
    ][
        "condition"
    ]

    parent_dir = (
        parent_root
        / source_id
        / parent_condition
        / f"seed_{seed}"
    )
    parent_corpus = (
        parent_dir
        / "corpus.jsonl"
    )
    parent_key_path = (
        parent_dir
        / "key.json"
    )

    if not parent_corpus.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.4 K16 corpus missing: {parent_corpus}"
        )
    if not parent_key_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase-3A.4 K16 key missing: {parent_key_path}"
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
        new_key[
            "mono_forward"
        ]
        != parent_key[
            "mono_forward"
        ]
    ):
        raise ValueError(
            f"{source_id}/seed={seed}: inherited mono base mapping differs"
        )

    if (
        list(
            new_key[
                "master_suffixes"
            ]
        )
        != list(
            parent_key[
                "master_suffixes"
            ]
        )
    ):
        raise ValueError(
            f"{source_id}/seed={seed}: inherited K16 suffix inventory/order differs"
        )

    if (
        new_key[
            "parent_phase3a4_master_key_sha256"
        ]
        != parent_key[
            "master_key_sha256"
        ]
    ):
        raise ValueError(
            f"{source_id}/seed={seed}: parent master-key fingerprint mismatch"
        )

    parent_train, parent_validation = (
        read_control_units(
            parent_corpus,
            train_indices=train_indices,
            validation_indices=validation_indices,
            expected_line_count=expected_line_count,
        )
    )
    new_train, new_validation = (
        read_control_units(
            new_corpus_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
            expected_line_count=expected_line_count,
        )
    )

    parent_train_dict = _unit_dict(
        parent_train
    )
    parent_validation_dict = _unit_dict(
        parent_validation
    )
    new_train_dict = _unit_dict(
        new_train
    )
    new_validation_dict = _unit_dict(
        new_validation
    )

    if (
        set(
            parent_train_dict
        )
        != set(
            new_train_dict
        )
        or set(
            parent_validation_dict
        )
        != set(
            new_validation_dict
        )
    ):
        raise ValueError(
            f"{source_id}/seed={seed}: parent/new split memberships differ"
        )

    base_comparisons = 0

    for (
        parent_split,
        new_split,
    ) in (
        (
            parent_train_dict,
            new_train_dict,
        ),
        (
            parent_validation_dict,
            new_validation_dict,
        ),
    ):
        for line_index in sorted(
            parent_split
        ):
            parent_line = parent_split[
                line_index
            ]
            new_line = new_split[
                line_index
            ]

            if len(
                parent_line
            ) != len(
                new_line
            ):
                raise ValueError(
                    f"{source_id}/seed={seed}/line={line_index}: token count differs"
                )

            for parent_token, new_token in zip(
                parent_line,
                new_line,
            ):
                if tuple(
                    parent_token[
                        :-1
                    ]
                ) != tuple(
                    new_token[
                        :-1
                    ]
                ):
                    raise ValueError(
                        f"{source_id}/seed={seed}/line={line_index}: "
                        "encoded base differs from Phase 3A.4"
                    )
                if len(
                    parent_token
                ) != len(
                    new_token
                ):
                    raise ValueError(
                        f"{source_id}/seed={seed}/line={line_index}: token length differs"
                    )
                base_comparisons += 1

    (
        parent_seen,
        parent_total,
        parent_fraction,
    ) = exact_seen_in_train_fraction(
        parent_train,
        parent_validation,
    )
    (
        new_seen,
        new_total,
        new_fraction,
    ) = exact_seen_in_train_fraction(
        new_train,
        new_validation,
    )

    if parent_total != new_total:
        raise ValueError(
            "Parent/new validation token totals differ"
        )

    lineage = {
        "source_id": source_id,
        "language": language,
        "seed": int(
            seed
        ),
        "base_mapping_identical_to_phase3a4": True,
        "master_suffix_inventory_order_identical_to_phase3a4_K16": True,
        "parent_master_key_identical": True,
        "encoded_base_identical_train_validation": True,
        "token_length_identical_train_validation": True,
        "base_token_comparisons": int(
            base_comparisons
        ),
        "parent_corpus_sha256": sha256_file(
            parent_corpus
        ),
        "parent_key_sha256": sha256_file(
            parent_key_path
        ),
        "new_corpus_sha256": sha256_file(
            new_corpus_path
        ),
        "new_key_sha256": sha256_file(
            new_key_path
        ),
    }

    collision = {
        "source_id": source_id,
        "language": language,
        "seed": int(
            seed
        ),
        "validation_tokens": int(
            parent_total
        ),
        "phase3a4_K16_validation_exact_tokens_seen_in_train": int(
            parent_seen
        ),
        "phase3a4_K16_validation_seen_in_train_fraction": float(
            parent_fraction
        ),
        "balanced_validation_exact_tokens_seen_in_train": int(
            new_seen
        ),
        "balanced_validation_seen_in_train_fraction": float(
            new_fraction
        ),
        "absolute_seen_in_train_fraction_change": float(
            new_fraction
            - parent_fraction
        ),
        "relative_seen_in_train_fraction_change": (
            (
                new_fraction
                - parent_fraction
            )
            / parent_fraction
            if parent_fraction
            else None
        ),
    }

    return (
        lineage,
        collision,
    )


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    languages: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], List[dict], List[dict], bool]:
    language_rows: List[dict] = []
    relation_rows: List[dict] = []
    global_rows: List[dict] = []

    for relation in relations:
        relation_group = [
            row
            for row in rows
            if row[
                "relation"
            ] == relation
        ]
        if len(
            relation_group
        ) != 20:
            raise ValueError(
                f"{relation}: expected 20 runs, found {len(relation_group)}"
            )

        deltas = [
            float(
                row[
                    "delta_competitor_minus_trigram_bits_per_event"
                ]
            )
            for row in relation_group
        ]

        global_rows.append(
            {
                "relation": relation,
                "runs": len(
                    relation_group
                ),
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
                    for row in relation_group
                ),
                "strong_direction_runs": sum(
                    bool(
                        row[
                            "strong_direction_match_voynich"
                        ]
                    )
                    for row in relation_group
                ),
            }
        )

        replicated_languages = []

        for language in languages:
            group = [
                row
                for row in relation_group
                if row[
                    "language"
                ] == language
            ]
            if len(
                group
            ) != 5:
                raise ValueError(
                    f"{relation}/{language}: expected five seeds"
                )

            language_deltas = [
                float(
                    row[
                        "delta_competitor_minus_trigram_bits_per_event"
                    ]
                )
                for row in group
            ]
            strong = sum(
                bool(
                    row[
                        "strong_direction_match_voynich"
                    ]
                )
                for row in group
            )
            replicated = (
                strong >= 4
            )
            if replicated:
                replicated_languages.append(
                    language
                )

            language_rows.append(
                {
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
                    "strong_direction_seeds": strong,
                    "language_relation_replicated": replicated,
                }
            )

        robust = (
            len(
                replicated_languages
            )
            >= 3
        )

        relation_rows.append(
            {
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

    full_hierarchy = all(
        bool(
            row[
                "relation_robust_3_of_4"
            ]
        )
        for row in relation_rows
    )

    return (
        language_rows,
        relation_rows,
        global_rows,
        full_hierarchy,
    )


def summarize_collisions(
    rows: Sequence[Mapping[str, object]],
) -> List[dict]:
    groups = defaultdict(
        list
    )
    for row in rows:
        groups[
            row[
                "language"
            ]
        ].append(
            row
        )

    output = []
    for language in sorted(
        groups
    ):
        group = groups[
            language
        ]
        if len(
            group
        ) != 5:
            raise ValueError(
                f"{language}: expected five collision-comparison seeds"
            )

        parent = [
            float(
                row[
                    "phase3a4_K16_validation_seen_in_train_fraction"
                ]
            )
            for row in group
        ]
        balanced = [
            float(
                row[
                    "balanced_validation_seen_in_train_fraction"
                ]
            )
            for row in group
        ]
        changes = [
            b - p
            for p, b in zip(
                parent,
                balanced,
            )
        ]

        output.append(
            {
                "language": language,
                "seeds": 5,
                "mean_phase3a4_K16_validation_seen_in_train_fraction": mean(
                    parent
                ),
                "mean_balanced_validation_seen_in_train_fraction": mean(
                    balanced
                ),
                "mean_absolute_fraction_change": mean(
                    changes
                ),
                "sd_absolute_fraction_change": pstdev(
                    changes
                ),
                "seeds_with_reduced_exact_seen_in_train_fraction": sum(
                    change < 0.0
                    for change in changes
                ),
            }
        )

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate Phase-3A.6 recurrence-avoiding "
            "base-preserving K=16 controls."
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
            != "recurrence-avoiding-surface-allocation-exploration-v1"
        ):
            raise ValueError(
                "Unexpected Phase-3A.6 experiment_id"
            )
        if (
            config.get(
                "status"
            )
            != "exploratory"
        ):
            raise ValueError(
                "Phase 3A.6 must remain exploratory"
            )

        if int(
            config[
                "mechanism"
            ][
                "variant_count"
            ]
        ) != 16:
            raise ValueError(
                "Phase 3A.6 K must remain exactly 16"
            )
        if int(
            config[
                "mechanism"
            ][
                "suffix_length_glyphs"
            ]
        ) != 1:
            raise ValueError(
                "Phase 3A.6 must append exactly one suffix glyph"
            )

        frozen = verify_frozen_inputs(
            config
        )

        generator_hash = sha256_file(
            Path(
                mechanism.__file__
            )
        )
        level2_path = Path(
            config[
                "level2_config_path"
            ]
        )
        level2 = json.loads(
            level2_path.read_text(
                encoding="utf-8"
            )
        )

        seeds = [
            int(x)
            for x in config[
                "seeds"
            ]
        ]
        if len(
            seeds
        ) != 5:
            raise ValueError(
                "Expected exactly five frozen seeds"
            )

        expected_runs = int(
            config[
                "expected_runs"
            ]
        )
        if expected_runs != (
            len(
                config[
                    "sources"
                ]
            )
            * len(
                seeds
            )
        ):
            raise ValueError(
                "expected_runs does not match source/seed grid"
            )

        source_root = Path(
            config[
                "source_root"
            ]
        )
        split_root = Path(
            config[
                "split_root"
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
        run_root = (
            result_root
            / "runs"
        )
        run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        relation_rows: List[dict] = []
        lineage_rows: List[dict] = []
        collision_rows: List[dict] = []
        mechanism_rows: List[dict] = []
        completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A.6 — RECURRENCE-AVOIDING SURFACE ALLOCATION"
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
            f"{frozen['parent_phase3a4_config_sha256']}"
        )
        print(
            "Parent 3A.4 gen:  "
            f"{frozen['parent_phase3a4_generator_sha256']}"
        )
        print(
            "Phase 3A.D1 cfg:  "
            f"{frozen['phase3ad1_config_sha256']}"
        )
        print(
            "Phase 3A.D1 anal: "
            f"{frozen['phase3ad1_analysis_sha256']}"
        )
        print(
            "D1 parity:        "
            f"{frozen['phase3ad1_parity_passes']}/"
            f"{frozen['phase3ad1_parity_runs']} PASS"
        )
        print(
            f"Level-2 config:   {frozen['level2_config_sha256']}"
        )
        print(
            "Variant count:    16 (fixed)"
        )
        print(
            "Suffix length:    1 glyph (fixed)"
        )
        print(
            "Allocation:       balanced without replacement per lexical base"
        )
        print(
            f"Expected runs:    {expected_runs}"
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
            if len(
                lines
            ) != int(
                source_cfg[
                    "expected_lines"
                ]
            ):
                raise ValueError(
                    f"{source_id}: source line-count mismatch"
                )
            if token_count(
                lines
            ) != int(
                source_cfg[
                    "expected_tokens"
                ]
            ):
                raise ValueError(
                    f"{source_id}: source token-count mismatch"
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
                    f"{source_id}: split/source SHA mismatch"
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

            for seed in seeds:
                control_dir = (
                    control_root
                    / source_id
                    / "balanced16"
                    / f"seed_{seed}"
                )
                corpus_path = (
                    control_dir
                    / "corpus.jsonl"
                )
                key_path = (
                    control_dir
                    / "key.json"
                )
                control_summary_path = (
                    control_dir
                    / "summary.json"
                )

                provenance = {
                    "phase3a6_config_path": str(
                        args.config
                    ),
                    "phase3a6_config_sha256": config_hash,
                    "generator_path": str(
                        Path(
                            mechanism.__file__
                        )
                    ),
                    "generator_sha256": generator_hash,
                    "source_path": str(
                        source_path
                    ),
                    "source_sha256": source_hash,
                    "split_path": str(
                        split_path
                    ),
                    "split_sha256": split_hash,
                    "level2_config_path": str(
                        level2_path
                    ),
                    "level2_config_sha256": frozen[
                        "level2_config_sha256"
                    ],
                    "parent_phase3a4_config_sha256": frozen[
                        "parent_phase3a4_config_sha256"
                    ],
                    "parent_phase3a4_generator_sha256": frozen[
                        "parent_phase3a4_generator_sha256"
                    ],
                    "phase3ad1_config_sha256": frozen[
                        "phase3ad1_config_sha256"
                    ],
                    "phase3ad1_analysis_sha256": frozen[
                        "phase3ad1_analysis_sha256"
                    ],
                    "phase3ad1_summary_sha256": frozen[
                        "phase3ad1_summary_sha256"
                    ],
                }

                if (
                    corpus_path.is_file()
                    and key_path.is_file()
                    and control_summary_path.is_file()
                ):
                    saved = json.loads(
                        control_summary_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    saved_provenance = saved.get(
                        "provenance",
                        {},
                    )
                    for (
                        key_name,
                        value,
                    ) in provenance.items():
                        if (
                            saved_provenance.get(
                                key_name
                            )
                            != value
                        ):
                            raise ValueError(
                                f"Existing control provenance differs for "
                                f"{key_name}: {control_dir}"
                            )
                    generation_status = "REUSED"
                else:
                    generated = generate_control(
                        lines,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        reserved_indices=reserved_indices,
                        seed=seed,
                    )
                    write_generated_control(
                        control_dir,
                        generated,
                        provenance=provenance,
                    )
                    generation_status = "GENERATED"

                control_summary = json.loads(
                    control_summary_path.read_text(
                        encoding="utf-8"
                    )
                )

                for required_flag in (
                    "round_trip_verified_train_validation",
                    "base_form_preservation_verified_train_validation",
                    "token_length_control_verified_train_validation",
                    "balanced_without_replacement_schedule_verified",
                    "validation_continues_frozen_train_state",
                ):
                    if (
                        control_summary.get(
                            required_flag
                        )
                        is not True
                    ):
                        raise ValueError(
                            f"{required_flag} was not verified"
                        )

                if (
                    control_summary.get(
                        "reserved_test_encoded"
                    )
                    is not False
                ):
                    raise ValueError(
                        "Reserved control test must remain unencoded"
                    )

                continuation = control_summary[
                    "continuation_diagnostics"
                ]
                if int(
                    continuation[
                        "validation_avoidable_collisions"
                    ]
                ) != 0:
                    raise ValueError(
                        "Balanced mechanism produced an avoidable validation collision"
                    )

                (
                    lineage,
                    collision,
                ) = verify_parent_lineage_and_collisions(
                    source_id=source_id,
                    language=language,
                    seed=seed,
                    config=config,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    expected_line_count=len(
                        lines
                    ),
                    new_corpus_path=corpus_path,
                    new_key_path=key_path,
                )
                lineage_rows.append(
                    lineage
                )
                collision_rows.append(
                    collision
                )

                mechanism_rows.append(
                    {
                        "source_id": source_id,
                        "language": language,
                        "seed": int(
                            seed
                        ),
                        "variant_count": 16,
                        "suffix_length_glyphs": 1,
                        "train_exact_cipher_token_recurrence_fraction": (
                            control_summary[
                                "train_diagnostics"
                            ][
                                "exact_cipher_token_recurrence_fraction"
                            ]
                        ),
                        "validation_exact_cipher_token_recurrence_fraction": (
                            control_summary[
                                "validation_diagnostics"
                            ][
                                "exact_cipher_token_recurrence_fraction"
                            ]
                        ),
                        "validation_exact_token_seen_in_train_fraction": (
                            control_summary[
                                "validation_diagnostics"
                            ][
                                "validation_exact_token_seen_in_train_fraction"
                            ]
                        ),
                        "validation_avoidable_collision_opportunities": (
                            continuation[
                                "validation_avoidable_collision_opportunities"
                            ]
                        ),
                        "validation_avoidable_collisions": (
                            continuation[
                                "validation_avoidable_collisions"
                            ]
                        ),
                        "validation_unavoidable_train_surface_reuses": (
                            continuation[
                                "validation_unavoidable_train_surface_reuses"
                            ]
                        ),
                        "train_same_surface_probability_given_same_plaintext_word": (
                            control_summary[
                                "train_diagnostics"
                            ][
                                "same_surface_probability_given_same_plaintext_word"
                            ]
                        ),
                        "validation_same_surface_probability_given_same_plaintext_word": (
                            control_summary[
                                "validation_diagnostics"
                            ][
                                "same_surface_probability_given_same_plaintext_word"
                            ]
                        ),
                    }
                )

                run_dir = (
                    run_root
                    / source_id
                    / "balanced16"
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
                corpus_hash = sha256_file(
                    corpus_path
                )

                if (
                    relations_path.is_file()
                    and result_summary_path.is_file()
                ):
                    result_summary = json.loads(
                        result_summary_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    required = {
                        "phase3a6_config_sha256": config_hash,
                        "generator_sha256": generator_hash,
                        "control_corpus_sha256": corpus_hash,
                        "level2_config_sha256": frozen[
                            "level2_config_sha256"
                        ],
                    }
                    for (
                        key_name,
                        value,
                    ) in required.items():
                        if (
                            result_summary.get(
                                key_name
                            )
                            != value
                        ):
                            raise ValueError(
                                f"Existing result provenance differs for "
                                f"{key_name}: {run_dir}"
                            )
                    run_relations = [
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

                    run_relations, detail = evaluate_one_control(
                        source_id=source_id,
                        language=language,
                        family="recurrence_avoiding_surface_allocation",
                        control_seed=seed,
                        train_units=train_units,
                        validation_units=validation_units,
                        config=level2,
                    )

                    for row in run_relations:
                        row[
                            "phase3a6_allocation"
                        ] = "balanced16"

                    run_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    write_csv(
                        relations_path,
                        run_relations,
                    )

                    result_summary = {
                        "schema_version": "1.0",
                        "experiment_id": config[
                            "experiment_id"
                        ],
                        "status": "exploratory",
                        "source_id": source_id,
                        "language": language,
                        "seed": int(
                            seed
                        ),
                        "allocation": "balanced16",
                        "phase3a6_config_sha256": config_hash,
                        "generator_sha256": generator_hash,
                        "source_sha256": source_hash,
                        "split_sha256": split_hash,
                        "level2_config_sha256": frozen[
                            "level2_config_sha256"
                        ],
                        "control_corpus_sha256": corpus_hash,
                        "control_key_sha256": sha256_file(
                            key_path
                        ),
                        "parent_lineage_verified": True,
                        "zero_avoidable_validation_collisions_verified": True,
                        "relations": run_relations,
                        "model_selection_detail": detail,
                        "reserved_test_encoded": False,
                        "reserved_test_parsed_or_scored": False,
                        "voynich_locked_test_accessed": False,
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
                    for row in run_relations
                } != expected_relations:
                    raise ValueError(
                        f"Unexpected relation set: {run_dir}"
                    )

                for row in run_relations:
                    row[
                        "phase3a6_allocation"
                    ] = "balanced16"

                relation_rows.extend(
                    run_relations
                )
                completed += 1

                token_row = next(
                    row
                    for row in run_relations
                    if row[
                        "relation"
                    ]
                    == "token_markov"
                )

                print(
                    f"  [{completed:2d}/{expected_runs}] "
                    f"seed={seed} "
                    f"{generation_status}/{evaluation_status} "
                    f"LINEAGE-PASS "
                    f"seen_train "
                    f"{float(collision['phase3a4_K16_validation_seen_in_train_fraction']):.3f}"
                    f"→{float(collision['balanced_validation_seen_in_train_fraction']):.3f} "
                    f"tokenΔ="
                    f"{float(token_row['delta_competitor_minus_trigram_bits_per_event']):+.4f}"
                )

            print()

        if completed != expected_runs:
            raise RuntimeError(
                f"Expected {expected_runs} runs; completed {completed}"
            )

        if len(
            relation_rows
        ) != int(
            config[
                "expected_relation_rows"
            ]
        ):
            raise RuntimeError(
                f"Expected {config['expected_relation_rows']} relation rows; "
                f"found {len(relation_rows)}"
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
            replication_summary,
            global_summary,
            full_hierarchy,
        ) = summarize_relations(
            relation_rows,
            languages=languages,
            relations=config[
                "operational_relations"
            ],
        )

        collision_summary = summarize_collisions(
            collision_rows
        )

        result_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_tables = {
            "run_relation_matrix.csv": relation_rows,
            "parent_lineage_verification.csv": lineage_rows,
            "collision_comparison.csv": collision_rows,
            "collision_language_summary.csv": collision_summary,
            "mechanism_diagnostics.csv": mechanism_rows,
            "language_relation_summary.csv": language_summary,
            "relation_replication_summary.csv": replication_summary,
            "global_relation_summary.csv": global_summary,
        }

        written = []

        for filename, rows in output_tables.items():
            path = (
                result_root
                / filename
            )
            write_csv(
                path,
                rows,
            )
            written.append(
                path
            )

        token_language = {
            row[
                "language"
            ]: {
                "mean_delta_bits_per_event": row[
                    "mean_delta_bits_per_event"
                ],
                "positive_direction_seeds": row[
                    "positive_direction_seeds"
                ],
                "strong_direction_seeds": row[
                    "strong_direction_seeds"
                ],
                "language_relation_replicated": row[
                    "language_relation_replicated"
                ],
            }
            for row in language_summary
            if row[
                "relation"
            ]
            == "token_markov"
        }

        summary_path = (
            result_root
            / "summary.json"
        )
        summary = {
            "schema_version": "1.0",
            "experiment_id": config[
                "experiment_id"
            ],
            "status": "exploratory",
            "phase3a6_config_sha256": config_hash,
            "generator_sha256": generator_hash,
            **frozen,
            "runs": completed,
            "relation_rows": len(
                relation_rows
            ),
            "variant_count": 16,
            "suffix_length_glyphs": 1,
            "allocation": "balanced_without_replacement_per_lexical_base",
            "parent_lineage_checks": len(
                lineage_rows
            ),
            "parent_lineage_all_passed": True,
            "zero_avoidable_validation_collisions_verified": all(
                int(
                    row[
                        "validation_avoidable_collisions"
                    ]
                )
                == 0
                for row in mechanism_rows
            ),
            "token_markov_by_language": token_language,
            "full_inherited_level2_hierarchy": bool(
                full_hierarchy
            ),
            "primary_prediction": config[
                "primary_prediction"
            ],
            "inherited_replication_rule": config[
                "inherited_replication_rule"
            ],
            "stop_rule": config[
                "stop_rule"
            ],
            "reserved_control_test_encoded": False,
            "reserved_control_test_parsed_or_scored": False,
            "voynich_locked_test_accessed": False,
            "automatic_confirmatory_promotion": False,
            "next_step": (
                "If and only if all five inherited relations are robust on "
                "development validation, stop exploratory mechanism development "
                "and write/freeze a separate Phase-3B confirmatory protocol before "
                "reserved-test access. Otherwise retain this result and formulate "
                "a distinct new mechanism hypothesis without tuning this allocation."
            ),
        }
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(
            summary_path
        )

        checksum_path = (
            result_root
            / "SHA256SUMS"
        )
        checksum_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in written
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print(
            "PHASE 3A.6 EXPLORATION COMPLETE"
        )
        print("=" * 96)
        print(
            f"Runs:              {completed}"
        )
        print(
            f"Relations:         {len(relation_rows)}"
        )
        print(
            f"Parent lineage:    {len(lineage_rows)}/{len(lineage_rows)} PASS"
        )
        print(
            "Avoidable collisions: 0 by construction and verification"
        )
        print(
            f"Full hierarchy:    {'YES' if full_hierarchy else 'NO'}"
        )
        print(
            "Control test:      NOT ENCODED / NOT SCORED"
        )
        print(
            "Voynich test:      NOT ACCESSED"
        )
        print()
        print(
            "Token-Markov by language:"
        )
        for language in languages:
            row = token_language[
                language
            ]
            print(
                f"  {language:<10} "
                f"meanΔ={float(row['mean_delta_bits_per_event']):+.6f} "
                f"strong={int(row['strong_direction_seeds'])}/5 "
                f"replicated={'YES' if row['language_relation_replicated'] else 'NO'}"
            )
        print()
        print(
            "Do not add K>16 or post-hoc allocation variants. "
            "Phase 3A.6 remains exploratory."
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A.6 INTERRUPTED. Hash-matched generated controls/results "
            "can be reused by rerunning the same frozen command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A.6 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
