#!/usr/bin/env python3
"""Run Phase 3A.2 lexical-coverage dose-response exploration.

Intended repository destination:
    scripts/run_latent_variant_coverage_exploration.py

Grid:
    4 languages x N={64,256,1024} x 5 seeds = 60 controls
    60 controls x 5 frozen predictive relations = 300 relation rows

The runner reuses the frozen Level-2 predictive evaluator on CONTROL
TRAIN/VALIDATION only. Reserved control-test lines are not encoded, parsed,
fitted, or scored. Voynich test_LOCKED.txt is not read or required.
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


from src.ciphers import latent_variant_coverage
from src.ciphers.latent_variant_coverage import (
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
    "configs/mechanisms/latent_variant_coverage_v1.json"
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
        "target_count",
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


def flatten_mechanism_diagnostics(
    *,
    source_id: str,
    language: str,
    target_count: int,
    seed: int,
    generated_summary: Mapping[str, object],
    master_codebook_sha256: str,
) -> dict:
    train = generated_summary["train_diagnostics"]
    validation = generated_summary["validation_diagnostics"]

    row = {
        "source_id": source_id,
        "language": language,
        "target_count": int(target_count),
        "seed": int(seed),
        "master_codebook_sha256": master_codebook_sha256,
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


def verify_master_codebook_invariance(
    diagnostic_rows: Sequence[Mapping[str, object]],
    *,
    target_counts: Sequence[int],
) -> None:
    """Ensure each source/seed uses one identical master key across N."""
    groups = {}

    for row in diagnostic_rows:
        key = (
            row["source_id"],
            int(row["seed"]),
        )
        groups.setdefault(key, []).append(row)

    for (source_id, seed), rows in groups.items():
        observed_n = sorted(
            int(row["target_count"])
            for row in rows
        )
        if observed_n != sorted(
            int(x) for x in target_counts
        ):
            raise ValueError(
                f"{source_id}/seed={seed}: expected N grid "
                f"{list(target_counts)}, found {observed_n}"
            )

        fingerprints = {
            row["master_codebook_sha256"]
            for row in rows
        }
        if len(fingerprints) != 1:
            raise ValueError(
                f"{source_id}/seed={seed}: master codebook changed across N"
            )


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    target_counts: Sequence[int],
    languages: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], List[dict]]:
    language_rows: List[dict] = []
    global_rows: List[dict] = []

    for n in target_counts:
        for relation in relations:
            global_group = [
                row
                for row in rows
                if int(row["target_count"]) == int(n)
                and row["relation"] == relation
            ]

            if len(global_group) != 20:
                raise ValueError(
                    f"N={n}/{relation}: expected 20 runs; "
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
                    "target_count": int(n),
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
                        f"N={n}/{relation}/{language}: "
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
                        "target_count": int(n),
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


def summarize_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    target_counts: Sequence[int],
    languages: Sequence[str],
) -> List[dict]:
    output = []

    for n in target_counts:
        for language in languages:
            group = [
                row
                for row in rows
                if int(row["target_count"]) == int(n)
                and row["language"] == language
            ]

            if len(group) != 5:
                raise ValueError(
                    f"N={n}/{language}: expected 5 diagnostics; "
                    f"found {len(group)}"
                )

            fields = [
                "validation_targeted_token_fraction",
                "validation_mean_realized_variants_per_target_word_present",
                "validation_targeted_variant_utilization_fraction",
                "validation_same_surface_probability_given_same_target_plaintext_word",
                "validation_exact_cipher_token_recurrence_fraction",
            ]

            row = {
                "target_count": int(n),
                "language": language,
                "seeds": 5,
            }

            for field in fields:
                values = [
                    float(item[field])
                    for item in group
                    if item[field] not in ("", None)
                ]
                row[f"mean_{field}"] = (
                    mean(values)
                    if values else None
                )

            output.append(row)

    return output


def trajectory_summary(
    global_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    *,
    target_counts: Sequence[int],
) -> dict:
    token_rows = {
        int(row["target_count"]): row
        for row in global_rows
        if row["relation"] == "token_markov"
    }

    token_means = [
        float(
            token_rows[int(n)][
                "mean_delta_bits_per_event"
            ]
        )
        for n in target_counts
    ]

    non_token_relations = {
        "hmm_space_free",
        "hmm_token_aware",
        "slot_grammar",
        "copy_edit",
    }

    preserved = {}
    for n in target_counts:
        relevant = [
            row
            for row in global_rows
            if int(row["target_count"]) == int(n)
            and row["relation"] in non_token_relations
        ]
        preserved[str(n)] = {
            row["relation"]: {
                "mean_delta_bits_per_event": float(
                    row["mean_delta_bits_per_event"]
                ),
                "strong_direction_runs_20": int(
                    row["strong_direction_runs"]
                ),
            }
            for row in relevant
        }

    coverage_by_n = {}
    for n in target_counts:
        relevant = [
            row
            for row in coverage_rows
            if int(row["target_count"]) == int(n)
        ]
        fractions = [
            float(
                row[
                    "mean_validation_targeted_token_fraction"
                ]
            )
            for row in relevant
        ]
        coverage_by_n[str(n)] = {
            "mean_across_languages": mean(fractions),
            "min_language_mean": min(fractions),
            "max_language_mean": max(fractions),
        }

    return {
        "token_markov_mean_delta_by_n": {
            str(n): token_means[i]
            for i, n in enumerate(target_counts)
        },
        "token_markov_mean_delta_monotone_non_decreasing": all(
            right >= left
            for left, right in zip(
                token_means,
                token_means[1:],
            )
        ),
        "validation_targeted_token_fraction_by_n": coverage_by_n,
        "other_four_relations_by_n": preserved,
        "interpretation": (
            "Exploratory trajectory only. Do not select a confirmatory N "
            "or access reserved-test data without a separate Phase-3B freeze."
        ),
    }


def require_level2_compatibility(
    phase3a2: Mapping[str, object],
    level2: Mapping[str, object],
) -> None:
    if level2.get("freeze_id") != "level2-predictive-controls-v1":
        raise ValueError(
            "Unexpected Level-2 evaluator config freeze_id"
        )

    if (
        list(level2["operational_relations"])
        != list(phase3a2["operational_relations"])
    ):
        raise ValueError(
            "Phase-3A.2 operational relations differ from Level 2"
        )

    for source_id, source in phase3a2["sources"].items():
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
            "Generate and evaluate Phase-3A.2 lexical-coverage "
            "dose-response controls on frozen Level-2 TRAIN/VALIDATION."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        config_hash = config_sha256(args.config)
        config = json.loads(
            args.config.read_text(encoding="utf-8")
        )

        if (
            config.get("experiment_id")
            != "latent-variant-coverage-exploration-v1"
        ):
            raise ValueError(
                "Unexpected Phase-3A.2 experiment_id"
            )
        if config.get("status") != "exploratory":
            raise ValueError(
                "Phase 3A.2 must remain explicitly exploratory"
            )

        mechanism = config["mechanism"]
        target_counts = [
            int(x)
            for x in mechanism["coverage_target_counts"]
        ]
        if target_counts != [64, 256, 1024]:
            raise ValueError(
                "Coverage grid must be exactly [64,256,1024]"
            )
        if int(mechanism["master_target_count"]) != 1024:
            raise ValueError(
                "master_target_count must be exactly 1024"
            )
        if int(mechanism["variant_count"]) != 4:
            raise ValueError(
                "variant_count must be exactly 4"
            )
        if int(mechanism["code_alphabet_size"]) != 16:
            raise ValueError(
                "code_alphabet_size must be exactly 16"
            )
        if int(mechanism["code_token_length"]) != 4:
            raise ValueError(
                "code_token_length must be exactly 4"
            )

        seeds = [
            int(x)
            for x in config["seeds"]
        ]
        if len(seeds) != 5:
            raise ValueError(
                "Phase 3A.2 requires exactly five frozen seeds"
            )

        expected_runs = int(config["expected_runs"])
        calculated_runs = (
            len(config["sources"])
            * len(target_counts)
            * len(seeds)
        )
        if expected_runs != calculated_runs:
            raise ValueError(
                "expected_runs does not match source/N/seed cardinality"
            )

        generator_hash = sha256_file(
            Path(latent_variant_coverage.__file__)
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
        completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A.2 — LEXICAL COVERAGE DOSE–RESPONSE"
        )
        print("=" * 96)
        print(f"Experiment:       {config['experiment_id']}")
        print(f"Status:           {config['status'].upper()}")
        print(f"Config SHA:       {config_hash}")
        print(f"Generator SHA:    {generator_hash}")
        print(f"Level-2 config:   {level2_hash}")
        print("Languages:        4")
        print("Coverage N grid:  64, 256, 1024")
        print("K:                4 (fixed)")
        print("Code alphabet:    16 (fixed)")
        print("Code length:      4 (fixed)")
        print("Seeds/N/source:   5")
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
                int(x) for x in split["validation"]
            ]
            reserved_indices = [
                int(x) for x in split["reserved_test"]
            ]

            print(
                f"[{language}] {source_id}: "
                f"train={len(train_indices)} "
                f"validation={len(validation_indices)} "
                f"reserved={len(reserved_indices)}"
            )

            for n in target_counts:
                for seed in seeds:
                    control_dir = (
                        control_root
                        / source_id
                        / f"N{n}"
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
                            "phase3a2_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                        }
                        for key, value in required.items():
                            if provenance.get(key) != value:
                                raise ValueError(
                                    f"Existing generated control has "
                                    f"different {key}: {control_dir}"
                                )

                        if int(
                            existing["target_count"]
                        ) != n or int(
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
                            target_count=n,
                            master_target_count=int(
                                mechanism[
                                    "master_target_count"
                                ]
                            ),
                            code_alphabet_size=int(
                                mechanism[
                                    "code_alphabet_size"
                                ]
                            ),
                            variant_count=int(
                                mechanism[
                                    "variant_count"
                                ]
                            ),
                        )

                        provenance = {
                            "phase3a2_config_path": str(
                                args.config
                            ),
                            "phase3a2_config_sha256": config_hash,
                            "generator_path": str(
                                Path(
                                    latent_variant_coverage.__file__
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

                    mechanism_rows.append(
                        flatten_mechanism_diagnostics(
                            source_id=source_id,
                            language=language,
                            target_count=n,
                            seed=seed,
                            generated_summary=generated_summary,
                            master_codebook_sha256=generated_key[
                                "master_codebook_sha256"
                            ],
                        )
                    )

                    corpus_hash = sha256_file(
                        corpus_path
                    )
                    run_dir = (
                        result_run_root
                        / source_id
                        / f"N{n}"
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
                            "phase3a2_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "control_corpus_sha256": corpus_hash,
                            "level2_config_sha256": level2_hash,
                        }
                        for key, value in required.items():
                            if saved.get(key) != value:
                                raise ValueError(
                                    f"Existing Phase-3A.2 result has "
                                    f"different {key}: {run_dir}"
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
                                    f"latent_variant_coverage_N{n}"
                                ),
                                control_seed=seed,
                                train_units=train_units,
                                validation_units=validation_units,
                                config=level2,
                            )
                        )

                        for row in relation_rows:
                            row["target_count"] = n
                            row["phase3a2_mechanism"] = (
                                "related_multi_code_nomenclator_coverage"
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
                            "target_count": n,
                            "variant_count": 4,
                            "seed": seed,
                            "phase3a2_config_sha256": config_hash,
                            "generator_sha256": generator_hash,
                            "source_sha256": source_hash,
                            "split_sha256": split_hash,
                            "level2_config_sha256": level2_hash,
                            "control_corpus_sha256": corpus_hash,
                            "master_codebook_sha256": generated_key[
                                "master_codebook_sha256"
                            ],
                            "relations": relation_rows,
                            "model_selection_detail": detail,
                            "reserved_test_encoded": False,
                            "reserved_test_parsed_or_scored": False,
                            "voynich_locked_test_accessed": False,
                            "interpretation": (
                                "Exploratory coverage dose-response result. "
                                "Do not promote an N to confirmatory status "
                                "without a separate Phase-3B freeze."
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
                        row["target_count"] = n
                        row["phase3a2_mechanism"] = (
                            "related_multi_code_nomenclator_coverage"
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
                    coverage = float(
                        generated_summary[
                            "validation_diagnostics"
                        ][
                            "targeted_token_fraction"
                        ]
                    )

                    print(
                        f"  [{completed:2d}/{expected_runs}] "
                        f"N={n:<4d} seed={seed} "
                        f"{generation_status}/{evaluation_status}  "
                        f"coverage={coverage:.3f}  "
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

        verify_master_codebook_invariance(
            mechanism_rows,
            target_counts=target_counts,
        )

        languages = [
            source["language"]
            for source in config["sources"].values()
        ]

        language_summary, global_summary = (
            summarize_relations(
                all_relations,
                target_counts=target_counts,
                languages=languages,
                relations=config[
                    "operational_relations"
                ],
            )
        )
        coverage_summary = summarize_coverage(
            mechanism_rows,
            target_counts=target_counts,
            languages=languages,
        )
        trajectory = trajectory_summary(
            global_summary,
            coverage_summary,
            target_counts=target_counts,
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
        language_path = (
            result_root
            / "n_language_relation_summary.csv"
        )
        global_path = (
            result_root
            / "n_global_relation_summary.csv"
        )
        coverage_path = (
            result_root
            / "coverage_summary.csv"
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
            language_path,
            language_summary,
        )
        write_csv(
            global_path,
            global_summary,
        )
        write_csv(
            coverage_path,
            coverage_summary,
        )

        top_summary = {
            "schema_version": "1.0",
            "experiment_id": config[
                "experiment_id"
            ],
            "status": "exploratory",
            "phase3a2_config_sha256": config_hash,
            "generator_sha256": generator_hash,
            "level2_config_sha256": level2_hash,
            "runs": completed,
            "relation_rows": len(
                all_relations
            ),
            "target_counts": target_counts,
            "variant_count": 4,
            "code_alphabet_size": 16,
            "code_token_length": 4,
            "master_codebook_invariance_across_n": True,
            "trajectory": trajectory,
            "primary_prediction": config[
                "primary_prediction"
            ],
            "reserved_control_test_encoded": False,
            "reserved_control_test_parsed_or_scored": False,
            "voynich_locked_test_accessed": False,
            "no_single_similarity_score": True,
            "automatic_winner_selected": False,
            "next_step": (
                "Interpret the coverage trajectory. If a stable candidate "
                "region emerges, write and freeze a separate Phase-3B "
                "confirmatory protocol before any reserved-test use. If the "
                "trajectory plateaus or disrupts other relations, retain the "
                "negative result and design the next exploratory intervention."
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
            language_path,
            global_path,
            coverage_path,
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
        print("PHASE 3A.2 EXPLORATION COMPLETE")
        print("=" * 96)
        print(f"Runs:              {completed}")
        print(f"Relations:         {len(all_relations)}")
        print(f"Run matrix:        {relation_matrix_path}")
        print(f"Mechanism diag:    {mechanism_path}")
        print(f"Coverage summary:  {coverage_path}")
        print(f"N summary:         {global_path}")
        print(f"Trajectory:        {summary_path}")
        print("Master keys:       INVARIANT ACROSS N (verified)")
        print("Control test:      NOT ENCODED / NOT SCORED")
        print("Voynich test:      NOT ACCESSED")
        print()
        print("Token-Markov mean delta by N:")
        for n, value in (
            trajectory[
                "token_markov_mean_delta_by_n"
            ].items()
        ):
            print(
                f"  N={n}: {float(value):+.6f} bits/event"
            )
        print()
        print(
            "Do not select/freeze a confirmatory N from this runner. "
            "Phase 3A.2 is exploratory."
        )
        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A.2 INTERRUPTED. Existing hash-matched controls/results "
            "are reusable; rerun the same command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A.2 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
