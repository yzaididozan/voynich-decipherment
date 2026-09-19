#!/usr/bin/env python3
"""Run VOYAGER Phase 3A.D1 token-Markov failure localization.

Intended repository destination:
    scripts/run_token_markov_failure_localization.py

This diagnostic re-scores the already-frozen Phase-3A.5 development controls
and decomposes the exact token-Markov versus matched-trigram likelihood
difference token by token.

No new cipher is generated.
No new model hyperparameter is selected.
Reserved control rows are streamed past unparsed by the frozen Level-2 reader.
Voynich test_LOCKED is not read or required.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, pstdev
import sys
from typing import List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.analysis import token_markov_failure_localization as localization
from src.analysis.token_markov_failure_localization import (
    compact_transition_rows,
    read_selected_plaintext_units,
    score_validation_contributions,
    sha256_file,
    summarize_language,
    summarize_rows,
    summarize_top_frequency_sets,
    verify_aggregate_parity,
)
from src.evaluation.level2_predictive_controls import (
    read_control_units,
    write_csv,
)


DEFAULT_CONFIG = Path(
    "configs/diagnostics/token_markov_failure_localization_v1.json"
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


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def relation_key(
    row: Mapping[str, object],
) -> tuple:
    return (
        row["source_id"],
        row.get("factorization", ""),
        int(row["control_seed"]),
    )


def load_token_relation_rows(
    path: Path,
    *,
    factorization_labels: Sequence[str],
    seeds: Sequence[int],
    source_ids: Sequence[str],
) -> dict:
    rows = read_csv(path)

    selected = [
        row
        for row in rows
        if row.get("relation")
        == "token_markov"
    ]

    expected = (
        len(factorization_labels)
        * len(seeds)
        * len(source_ids)
    )
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} Phase-3A.5 token-Markov relation rows; "
            f"found {len(selected)}"
        )

    index = {}
    for row in selected:
        key = relation_key(row)
        if key in index:
            raise ValueError(
                f"Duplicate token-Markov relation row: {key}"
            )

        factorization = row.get(
            "factorization"
        )
        if factorization not in factorization_labels:
            raise ValueError(
                f"Unexpected factorization in relation matrix: {factorization!r}"
            )

        context_order = int(
            row["selected_hyperparameter"]
        )
        if context_order not in {
            1,
            2,
        }:
            raise ValueError(
                f"Unexpected selected token context order: {context_order}"
            )

        index[key] = row

    expected_keys = {
        (
            source_id,
            factorization,
            int(seed),
        )
        for source_id in source_ids
        for factorization in factorization_labels
        for seed in seeds
    }
    if set(index) != expected_keys:
        missing = sorted(
            expected_keys - set(index)
        )
        extra = sorted(
            set(index) - expected_keys
        )
        raise ValueError(
            f"Phase-3A.5 relation matrix identity mismatch. "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    return index


def verify_parent_inputs(
    config: Mapping[str, object],
) -> dict:
    inputs = config["inputs"]

    phase3a5_config_path = Path(
        inputs["phase3a5_config_path"]
    )
    phase3a5_generator_path = Path(
        inputs["phase3a5_generator_path"]
    )
    level2_config_path = Path(
        inputs["level2_config_path"]
    )

    for path in (
        phase3a5_config_path,
        phase3a5_generator_path,
        level2_config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required frozen input is missing: {path}"
            )

    observed_phase3a5_config = sha256_file(
        phase3a5_config_path
    )
    observed_phase3a5_generator = sha256_file(
        phase3a5_generator_path
    )
    observed_level2 = sha256_file(
        level2_config_path
    )

    expected = {
        "phase3a5_config": inputs[
            "expected_phase3a5_config_sha256"
        ],
        "phase3a5_generator": inputs[
            "expected_phase3a5_generator_sha256"
        ],
        "level2_config": inputs[
            "expected_level2_config_sha256"
        ],
    }
    observed = {
        "phase3a5_config": observed_phase3a5_config,
        "phase3a5_generator": observed_phase3a5_generator,
        "level2_config": observed_level2,
    }

    for name in expected:
        if observed[name] != expected[name]:
            raise ValueError(
                f"Frozen {name} SHA mismatch: "
                f"expected {expected[name]}, observed {observed[name]}"
            )

    phase3a5 = json.loads(
        phase3a5_config_path.read_text(
            encoding="utf-8"
        )
    )
    level2 = json.loads(
        level2_config_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        phase3a5.get("experiment_id")
        != "compositional-affix-factorization-exploration-v1"
    ):
        raise ValueError(
            "Unexpected Phase-3A.5 experiment_id"
        )
    if (
        level2.get("freeze_id")
        != "level2-predictive-controls-v1"
    ):
        raise ValueError(
            "Unexpected Level-2 freeze_id"
        )

    if (
        int(level2["token_markov"]["max_context"])
        != int(config["model"]["max_context"])
    ):
        raise ValueError(
            "Diagnostic max_context differs from frozen Level-2 config"
        )
    if (
        int(level2["token_markov"]["char_base_order"])
        != int(config["model"]["char_base_order"])
    ):
        raise ValueError(
            "Diagnostic char_base_order differs from frozen Level-2 config"
        )

    return {
        "phase3a5_config_sha256": observed_phase3a5_config,
        "phase3a5_generator_sha256": observed_phase3a5_generator,
        "level2_config_sha256": observed_level2,
        "phase3a5_config": phase3a5,
        "level2_config": level2,
    }


def factorization_robustness_summary(
    relation_index: Mapping[tuple, Mapping[str, object]],
    *,
    config: Mapping[str, object],
    factorization_labels: Sequence[str],
    seeds: Sequence[int],
) -> List[dict]:
    output = []

    for factorization in factorization_labels:
        for source_id, source_cfg in config[
            "sources"
        ].items():
            language = source_cfg[
                "language"
            ]

            rows = [
                relation_index[
                    (
                        source_id,
                        factorization,
                        int(seed),
                    )
                ]
                for seed in seeds
            ]

            deltas = [
                float(
                    row[
                        "delta_competitor_minus_trigram_bits_per_event"
                    ]
                )
                for row in rows
            ]
            contexts = [
                int(
                    row[
                        "selected_hyperparameter"
                    ]
                )
                for row in rows
            ]

            output.append(
                {
                    "factorization": factorization,
                    "source_id": source_id,
                    "language": language,
                    "phase3a5_token_relation_status": source_cfg[
                        "phase3a5_token_relation_status"
                    ],
                    "seeds": len(rows),
                    "mean_delta_bits_per_event": mean(
                        deltas
                    ),
                    "sd_delta_bits_per_event": (
                        pstdev(deltas)
                        if len(deltas) > 1
                        else 0.0
                    ),
                    "positive_seeds": sum(
                        float(
                            row[
                                "delta_competitor_minus_trigram_bits_per_event"
                            ]
                        )
                        > 0.0
                        for row in rows
                    ),
                    "strong_seeds": sum(
                        parse_bool(
                            row[
                                "strong_direction_match_voynich"
                            ]
                        )
                        for row in rows
                    ),
                    "selected_context_1_seeds": sum(
                        value == 1
                        for value in contexts
                    ),
                    "selected_context_2_seeds": sum(
                        value == 2
                        for value in contexts
                    ),
                }
            )

    return output


def compact_primary_summary(
    *,
    language_rows: Sequence[Mapping[str, object]],
    bin_summaries: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict:
    """Record descriptive localization highlights without selecting a mechanism."""
    language_payload = {
        row["language"]: {
            "phase3a5_token_relation_status": row[
                "phase3a5_token_relation_status"
            ],
            "mean_seed_delta_bits_per_event": row[
                "mean_seed_delta_bits_per_event"
            ],
            "token_markov_better_token_fraction": row[
                "token_markov_better_token_fraction"
            ],
            "token_markov_advantage_bits": row[
                "token_markov_advantage_bits"
            ],
            "trigram_advantage_bits": row[
                "trigram_advantage_bits"
            ],
        }
        for row in language_rows
    }

    highlights = {}

    for summary_name, rows in bin_summaries.items():
        per_language = {}

        languages = sorted(
            {
                row["language"]
                for row in rows
            }
        )

        for language in languages:
            subset = [
                row
                for row in rows
                if row[
                    "language"
                ] == language
            ]
            ranked = sorted(
                subset,
                key=lambda row: (
                    -float(
                        row[
                            "share_of_language_token_markov_advantage_bits"
                        ]
                    ),
                    str(row),
                ),
            )

            top = []
            for row in ranked[:3]:
                label_fields = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "runs",
                        "tokens",
                        "events",
                        "total_delta_bits",
                        "delta_bits_per_event_ratio_of_sums",
                        "mean_seed_delta_bits_per_event",
                        "sd_seed_delta_bits_per_event",
                        "tokens_token_markov_better",
                        "token_markov_better_token_fraction",
                        "token_markov_advantage_bits",
                        "trigram_advantage_bits",
                        "share_of_language_token_markov_advantage_bits",
                    }
                }
                top.append(
                    {
                        "bin": label_fields,
                        "share_of_language_token_markov_advantage_bits": row[
                            "share_of_language_token_markov_advantage_bits"
                        ],
                        "delta_bits_per_event_ratio_of_sums": row[
                            "delta_bits_per_event_ratio_of_sums"
                        ],
                        "tokens": row[
                            "tokens"
                        ],
                    }
                )

            per_language[
                language
            ] = top

        highlights[
            summary_name
        ] = per_language

    return {
        "language_summary": language_payload,
        "top_token_markov_advantage_mass_bins": highlights,
        "warning": (
            "These are descriptive concentration summaries, not independent "
            "causal effects and not an automatic Phase-3A.6 mechanism choice."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Localize Phase-3A.5 exact-token Markov residual predictive "
            "advantage using TRAIN-derived features only."
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
            != "token-markov-failure-localization-v1"
        ):
            raise ValueError(
                "Unexpected diagnostic experiment_id"
            )
        if (
            config.get("status")
            != "exploratory_diagnostic"
        ):
            raise ValueError(
                "Phase 3A.D1 must remain explicitly exploratory"
            )

        frozen = verify_parent_inputs(
            config
        )

        factorization_labels = list(
            config[
                "factorizations"
            ][
                "aggregate_parity_checks"
            ]
        )
        primary_factorization = config[
            "factorizations"
        ][
            "primary_localization"
        ]

        if factorization_labels != [
            "1x16",
            "2x8",
            "4x4",
        ]:
            raise ValueError(
                "Parity factorization grid must remain [1x16,2x8,4x4]"
            )
        if (
            primary_factorization
            != "1x16"
        ):
            raise ValueError(
                "Primary localization factorization must remain 1x16"
            )

        seeds = [
            int(x)
            for x in config[
                "seeds"
            ]
        ]
        if len(seeds) != 5:
            raise ValueError(
                "Expected exactly five frozen seeds"
            )

        relation_matrix_path = Path(
            config[
                "inputs"
            ][
                "phase3a5_relation_matrix"
            ]
        )
        if not relation_matrix_path.is_file():
            raise FileNotFoundError(
                f"Phase-3A.5 relation matrix missing: {relation_matrix_path}"
            )

        relation_matrix_sha = sha256_file(
            relation_matrix_path
        )
        relation_index = (
            load_token_relation_rows(
                relation_matrix_path,
                factorization_labels=factorization_labels,
                seeds=seeds,
                source_ids=list(
                    config[
                        "sources"
                    ]
                ),
            )
        )

        source_root = Path(
            config[
                "inputs"
            ][
                "source_root"
            ]
        )
        split_root = Path(
            config[
                "inputs"
            ][
                "split_root"
            ]
        )
        control_root = Path(
            config[
                "inputs"
            ][
                "phase3a5_control_root"
            ]
        )
        output_root = Path(
            config[
                "output_root"
            ]
        )
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        analysis_module_sha = sha256_file(
            Path(
                localization.__file__
            )
        )

        parity_rows: List[dict] = []
        primary_rows: List[dict] = []
        completed = 0
        primary_completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A.D1 — TOKEN-MARKOV FAILURE LOCALIZATION"
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
            f"Analysis SHA:     {analysis_module_sha}"
        )
        print(
            "Phase-3A.5 cfg:   "
            f"{frozen['phase3a5_config_sha256']}"
        )
        print(
            "Phase-3A.5 gen:   "
            f"{frozen['phase3a5_generator_sha256']}"
        )
        print(
            f"Relation matrix:  {relation_matrix_sha}"
        )
        print(
            f"Level-2 config:   {frozen['level2_config_sha256']}"
        )
        print(
            "Detailed condition: 1x16"
        )
        print(
            "Parity conditions:   1x16, 2x8, 4x4"
        )
        print(
            "New cipher:       NO"
        )
        print(
            "New hyperparameter selection: NO"
        )
        print(
            "Control reserved: NOT PARSED / NOT SCORED"
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
            expected_lines = int(
                source_cfg[
                    "expected_lines"
                ]
            )
            source_path = (
                source_root
                / source_id
                / "source.txt"
            )

            if (
                sha256_file(
                    source_path
                )
                != source_cfg[
                    "source_sha256"
                ]
            ):
                raise ValueError(
                    f"{source_id}: frozen source SHA mismatch"
                )

            split_path = (
                split_root
                / f"{source_id}.json"
            )
            split = json.loads(
                split_path.read_text(
                    encoding="utf-8"
                )
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

            if set(
                train_indices
            ) & set(
                validation_indices
            ):
                raise ValueError(
                    f"{source_id}: train/validation overlap"
                )

            plain_train = (
                read_selected_plaintext_units(
                    source_path,
                    selected_indices=train_indices,
                    expected_line_count=expected_lines,
                )
            )
            plain_validation = (
                read_selected_plaintext_units(
                    source_path,
                    selected_indices=validation_indices,
                    expected_line_count=expected_lines,
                )
            )

            print(
                f"[{language}] {source_id}: "
                f"train={len(train_indices)} "
                f"validation={len(validation_indices)}"
            )

            for factorization in factorization_labels:
                safe_factor = (
                    "F"
                    + factorization
                )

                for seed in seeds:
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
                    key_path = (
                        control_dir
                        / "key.json"
                    )

                    if not corpus_path.is_file():
                        raise FileNotFoundError(
                            f"Generated Phase-3A.5 corpus missing: {corpus_path}"
                        )
                    if not key_path.is_file():
                        raise FileNotFoundError(
                            f"Generated Phase-3A.5 key missing: {key_path}"
                        )

                    cipher_train, cipher_validation = (
                        read_control_units(
                            corpus_path,
                            train_indices=train_indices,
                            validation_indices=validation_indices,
                            expected_line_count=expected_lines,
                        )
                    )

                    key_payload = json.loads(
                        key_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    relation = relation_index[
                        (
                            source_id,
                            factorization,
                            seed,
                        )
                    ]
                    selected_context = int(
                        relation[
                            "selected_hyperparameter"
                        ]
                    )

                    token_rows, aggregate = (
                        score_validation_contributions(
                            source_id=source_id,
                            language=language,
                            factorization=factorization,
                            seed=seed,
                            plain_train_units=plain_train,
                            plain_validation_units=plain_validation,
                            cipher_train_units=cipher_train,
                            cipher_validation_units=cipher_validation,
                            key=key_payload,
                            selected_context_order=selected_context,
                            max_context=int(
                                config[
                                    "model"
                                ][
                                    "max_context"
                                ]
                            ),
                            char_base_order=int(
                                config[
                                    "model"
                                ][
                                    "char_base_order"
                                ]
                            ),
                            bins=config[
                                "bins"
                            ],
                            top_n_frequency_sets=config[
                                "top_n_frequency_sets"
                            ],
                        )
                    )

                    parity = (
                        verify_aggregate_parity(
                            aggregate,
                            relation,
                            tolerance=float(
                                config[
                                    "model"
                                ][
                                    "aggregate_parity_tolerance"
                                ]
                            ),
                        )
                    )
                    parity.update(
                        {
                            "phase3a5_strong_direction_match": (
                                parse_bool(
                                    relation[
                                        "strong_direction_match_voynich"
                                    ]
                                )
                            ),
                            "phase3a5_direction_match": (
                                parse_bool(
                                    relation[
                                        "direction_match_voynich"
                                    ]
                                )
                            ),
                            "phase3a5_ci_95_lower": float(
                                relation[
                                    "ci_95_lower"
                                ]
                            ),
                            "phase3a5_ci_95_upper": float(
                                relation[
                                    "ci_95_upper"
                                ]
                            ),
                            "control_corpus_sha256": (
                                sha256_file(
                                    corpus_path
                                )
                            ),
                            "control_key_sha256": (
                                sha256_file(
                                    key_path
                                )
                            ),
                        }
                    )
                    parity_rows.append(
                        parity
                    )

                    if (
                        factorization
                        == primary_factorization
                    ):
                        status = source_cfg[
                            "phase3a5_token_relation_status"
                        ]
                        for row in token_rows:
                            row[
                                "phase3a5_token_relation_status"
                            ] = status
                        primary_rows.extend(
                            token_rows
                        )
                        primary_completed += 1

                    completed += 1

                    print(
                        f"  [{completed:2d}/60] "
                        f"F={factorization:<4} seed={seed} "
                        f"context={selected_context} "
                        f"delta={float(parity['recomputed_delta_bits_per_event']):+.6f} "
                        f"parity_error={float(parity['delta_abs_error']):.2e} "
                        f"{'DETAIL' if factorization == primary_factorization else 'PARITY'}"
                    )

            print()

        if completed != int(
            config[
                "expected_runs"
            ][
                "aggregate_parity_checks"
            ]
        ):
            raise RuntimeError(
                f"Expected 60 parity runs; completed {completed}"
            )

        if primary_completed != int(
            config[
                "expected_runs"
            ][
                "detailed_primary_factorization_runs"
            ]
        ):
            raise RuntimeError(
                f"Expected 20 detailed primary runs; completed {primary_completed}"
            )

        if not all(
            bool(
                row[
                    "parity_pass"
                ]
            )
            for row in parity_rows
        ):
            raise RuntimeError(
                "At least one aggregate parity check failed"
            )

        # -------------------------
        # Frozen descriptive bins.
        # -------------------------
        frequency_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_plain_current_frequency_bin",
            ),
        )
        bigram_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_plain_bigram_frequency_bin",
            ),
        )
        cipher_recurrence_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_cipher_current_frequency_bin",
            ),
        )
        cipher_bigram_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_cipher_bigram_frequency_bin",
            ),
        )
        plain_conditional_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_plain_conditional_probability_bin",
            ),
        )
        cipher_conditional_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "train_cipher_conditional_probability_bin",
            ),
        )
        token_length_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "plaintext_token_length_bin",
            ),
        )
        line_position_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "line_position_class",
            ),
        )
        variant_summary = summarize_rows(
            primary_rows,
            group_fields=(
                "language",
                "factorization",
                "current_variant_index",
            ),
        )
        top_frequency_summary = (
            summarize_top_frequency_sets(
                primary_rows,
                top_n_frequency_sets=config[
                    "top_n_frequency_sets"
                ],
            )
        )

        status_by_language = {
            source_cfg[
                "language"
            ]: source_cfg[
                "phase3a5_token_relation_status"
            ]
            for source_cfg in config[
                "sources"
            ].values()
        }
        language_summary = summarize_language(
            primary_rows,
            relation_status_by_language=status_by_language,
        )

        robustness_summary = (
            factorization_robustness_summary(
                relation_index,
                config=config,
                factorization_labels=factorization_labels,
                seeds=seeds,
            )
        )

        transition_rows = compact_transition_rows(
            primary_rows
        )

        # -------------------------
        # Write outputs.
        # -------------------------
        paths = {
            "aggregate_parity_checks.csv": (
                parity_rows
            ),
            "token_contributions.csv": (
                primary_rows
            ),
            "transition_contributions.csv": (
                transition_rows
            ),
            "frequency_bin_summary.csv": (
                frequency_summary
            ),
            "bigram_frequency_bin_summary.csv": (
                bigram_summary
            ),
            "cipher_recurrence_bin_summary.csv": (
                cipher_recurrence_summary
            ),
            "cipher_bigram_frequency_bin_summary.csv": (
                cipher_bigram_summary
            ),
            "plain_conditional_probability_bin_summary.csv": (
                plain_conditional_summary
            ),
            "cipher_conditional_probability_bin_summary.csv": (
                cipher_conditional_summary
            ),
            "token_length_summary.csv": (
                token_length_summary
            ),
            "line_position_summary.csv": (
                line_position_summary
            ),
            "variant_index_summary.csv": (
                variant_summary
            ),
            "top_frequency_summary.csv": (
                top_frequency_summary
            ),
            "language_summary.csv": (
                language_summary
            ),
            "factorization_robustness_summary.csv": (
                robustness_summary
            ),
        }

        written_paths = []

        for filename, rows in paths.items():
            path = (
                output_root
                / filename
            )
            write_csv(
                path,
                rows,
            )
            written_paths.append(
                path
            )

        compact = compact_primary_summary(
            language_rows=language_summary,
            bin_summaries={
                "plaintext_frequency": frequency_summary,
                "plaintext_bigram_frequency": bigram_summary,
                "ciphertext_recurrence": cipher_recurrence_summary,
                "ciphertext_bigram_frequency": cipher_bigram_summary,
                "plaintext_conditional_probability": plain_conditional_summary,
                "ciphertext_conditional_probability": cipher_conditional_summary,
                "plaintext_token_length": token_length_summary,
                "line_position": line_position_summary,
            },
        )

        summary_path = (
            output_root
            / "summary.json"
        )
        summary = {
            "schema_version": "1.0",
            "experiment_id": config[
                "experiment_id"
            ],
            "status": config[
                "status"
            ],
            "diagnostic_config_sha256": config_hash,
            "analysis_module_sha256": analysis_module_sha,
            "phase3a5_config_sha256": frozen[
                "phase3a5_config_sha256"
            ],
            "phase3a5_generator_sha256": frozen[
                "phase3a5_generator_sha256"
            ],
            "phase3a5_relation_matrix_sha256": relation_matrix_sha,
            "level2_config_sha256": frozen[
                "level2_config_sha256"
            ],
            "parity_runs": len(
                parity_rows
            ),
            "parity_runs_passed": sum(
                bool(
                    row[
                        "parity_pass"
                    ]
                )
                for row in parity_rows
            ),
            "primary_factorization": primary_factorization,
            "detailed_primary_runs": primary_completed,
            "detailed_validation_token_rows": len(
                primary_rows
            ),
            "transition_rows": len(
                transition_rows
            ),
            "max_aggregate_parity_abs_error": max(
                max(
                    float(
                        row[
                            "competitor_abs_error"
                        ]
                    ),
                    float(
                        row[
                            "trigram_abs_error"
                        ]
                    ),
                    float(
                        row[
                            "delta_abs_error"
                        ]
                    ),
                )
                for row in parity_rows
            ),
            "localization": compact,
            "guardrails": config[
                "guardrails"
            ],
            "reserved_control_ciphertext_json_parsed": False,
            "reserved_control_units_scored": False,
            "voynich_locked_test_accessed": False,
            "followup_rule": config[
                "followup_rule"
            ],
        }
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        written_paths.append(
            summary_path
        )

        checksums_path = (
            output_root
            / "SHA256SUMS"
        )
        checksums_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in written_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print(
            "PHASE 3A.D1 LOCALIZATION COMPLETE"
        )
        print("=" * 96)
        print(
            f"Parity runs:       {len(parity_rows)}/60 PASS"
        )
        print(
            f"Detailed runs:     {primary_completed}/20 (1x16)"
        )
        print(
            f"Token rows:        {len(primary_rows)}"
        )
        print(
            f"Transition rows:   {len(transition_rows)}"
        )
        print(
            "Max parity error:  "
            f"{summary['max_aggregate_parity_abs_error']:.3e}"
        )
        print(
            f"Output root:       {output_root}"
        )
        print(
            "Control reserved:  NOT PARSED / NOT SCORED"
        )
        print(
            "Voynich test:      NOT ACCESSED"
        )
        print()
        print(
            "Read language_summary.csv and the frozen bin summaries before "
            "formulating Phase 3A.6. Do not let this diagnostic automatically "
            "select a mechanism."
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A.D1 INTERRUPTED. No new control data are generated; "
            "rerun the same frozen command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A.D1 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
