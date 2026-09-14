#!/usr/bin/env python3
"""Run Phase 3A constrained lexical-surface-variation exploration.

Intended repository destination:
    scripts/run_latent_variant_exploration.py

Prerequisites:
- Phase-3 config at configs/mechanisms/latent_variant_v1.json
- generator at src/ciphers/latent_variant.py
- frozen open-language source corpora
- frozen Level-2 control splits
- frozen Level-2 predictive evaluator

This runner evaluates 4 languages x 3 K values x 5 seeds = 60 controls.
It reuses the frozen Level-2 model procedures on CONTROL TRAIN/VALIDATION only.
Reserved control-test lines are never encoded, parsed, fitted, or scored.
Voynich test_LOCKED.txt is never read or required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.ciphers import latent_variant
from src.ciphers.latent_variant import (
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
    "configs/mechanisms/latent_variant_v1.json"
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


def _flatten_mechanism_diagnostics(
    *,
    source_id: str,
    language: str,
    variant_count: int,
    seed: int,
    generated_summary: Mapping[str, object],
) -> dict:
    train = generated_summary["train_diagnostics"]
    validation = generated_summary["validation_diagnostics"]

    row = {
        "source_id": source_id,
        "language": language,
        "variant_count": int(variant_count),
        "seed": int(seed),
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


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
    languages: Sequence[str],
    relations: Sequence[str],
) -> tuple[List[dict], List[dict]]:
    """Summarize without declaring an exploratory winner."""
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

            global_deltas = [
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
                    "mean_delta_bits_per_event": mean(global_deltas),
                    "median_delta_bits_per_event": median(global_deltas),
                    "sd_delta_bits_per_event": pstdev(global_deltas),
                    "min_delta_bits_per_event": min(global_deltas),
                    "max_delta_bits_per_event": max(global_deltas),
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

                deltas = [
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
                        "mean_delta_bits_per_event": mean(deltas),
                        "sd_delta_bits_per_event": pstdev(deltas),
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


def trajectory_summary(
    global_rows: Sequence[Mapping[str, object]],
    *,
    variant_counts: Sequence[int],
) -> dict:
    token_rows = {
        int(row["variant_count"]): row
        for row in global_rows
        if row["relation"] == "token_markov"
    }

    token_means = [
        float(token_rows[int(k)]["mean_delta_bits_per_event"])
        for k in variant_counts
    ]

    non_token_relations = {
        "hmm_space_free",
        "hmm_token_aware",
        "slot_grammar",
        "copy_edit",
    }

    preserved = {}
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
                "strong_direction_runs_20": int(
                    row["strong_direction_runs"]
                ),
            }
            for row in relevant
        }

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
        "other_four_relations_by_k": preserved,
        "interpretation": (
            "Exploratory trajectory only. Do not select a confirmatory K "
            "without writing a separate Phase-3B protocol first."
        ),
    }


def require_level2_compatibility(
    phase3: Mapping[str, object],
    level2: Mapping[str, object],
) -> None:
    if level2.get("freeze_id") != "level2-predictive-controls-v1":
        raise ValueError(
            "Unexpected Level-2 evaluator config freeze_id"
        )

    if (
        list(level2["operational_relations"])
        != list(phase3["operational_relations"])
    ):
        raise ValueError(
            "Phase-3 operational relations differ from Level 2"
        )

    for source_id, source in phase3["sources"].items():
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
            "Generate and evaluate Phase-3A related lexical-variant "
            "controls on the already-frozen Level-2 TRAIN/VALIDATION split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        phase3_sha = config_sha256(args.config)
        phase3 = json.loads(
            args.config.read_text(encoding="utf-8")
        )

        if (
            phase3.get("experiment_id")
            != "latent-variant-exploration-v1"
        ):
            raise ValueError("Unexpected Phase-3 experiment_id")
        if phase3.get("status") != "exploratory":
            raise ValueError(
                "Phase 3A must remain explicitly exploratory"
            )

        generator_sha = sha256_file(
            Path(latent_variant.__file__)
        )

        level2_config_path = Path(
            phase3["level2_config_path"]
        )
        level2_sha = config_sha256(
            level2_config_path
        )
        level2 = json.loads(
            level2_config_path.read_text(
                encoding="utf-8"
            )
        )
        require_level2_compatibility(
            phase3,
            level2,
        )

        mechanism = phase3["mechanism"]
        variant_counts = [
            int(x)
            for x in mechanism["variant_counts"]
        ]
        if variant_counts != [1, 2, 4]:
            raise ValueError(
                "Phase-3A variant grid must be exactly [1,2,4]"
            )

        seeds = [
            int(x)
            for x in phase3["seeds"]
        ]
        if len(seeds) != 5:
            raise ValueError(
                "Phase-3A requires exactly five frozen seeds"
            )

        expected_runs = int(
            phase3["expected_runs"]
        )
        if expected_runs != (
            len(phase3["sources"])
            * len(variant_counts)
            * len(seeds)
        ):
            raise ValueError(
                "expected_runs does not match source/K/seed cardinality"
            )

        source_root = Path(
            phase3["source_root"]
        )
        split_root = Path(
            phase3["level2_split_root"]
        )
        control_root = Path(
            phase3["control_output_root"]
        )
        result_root = Path(
            phase3["result_output_root"]
        )
        result_run_root = result_root / "runs"

        result_run_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        all_relations: List[dict] = []
        mechanism_rows: List[dict] = []
        completed = 0

        print("=" * 96)
        print(
            "VOYAGER PHASE 3A — CONSTRAINED LEXICAL SURFACE VARIATION"
        )
        print("=" * 96)
        print(f"Experiment:       {phase3['experiment_id']}")
        print(f"Status:           {phase3['status'].upper()}")
        print(f"Phase-3 config:   {phase3_sha}")
        print(f"Generator SHA:    {generator_sha}")
        print(f"Level-2 config:   {level2_sha}")
        print("Languages:        4")
        print("Variant K grid:   1, 2, 4")
        print("Seeds/K/source:   5")
        print(f"Expected runs:    {expected_runs}")
        print("Control test:     NOT ENCODED / NOT SCORED")
        print("Voynich test:     NOT ACCESSED")
        print()

        for source_id, source_cfg in phase3[
            "sources"
        ].items():
            language = source_cfg["language"]
            source_path = (
                source_root
                / source_id
                / "source.txt"
            )
            source_sha = sha256_file(
                source_path
            )

            if (
                source_sha
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
            split_sha = sha256_file(
                split_path
            )
            split = json.loads(
                split_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                split.get("source_sha256")
                != source_sha
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
                int(x) for x in split["reserved_test"]
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
                    control_summary_path = (
                        control_dir / "summary.json"
                    )
                    corpus_path = (
                        control_dir / "corpus.jsonl"
                    )

                    generated_summary = None

                    if (
                        control_summary_path.is_file()
                        and corpus_path.is_file()
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
                            "phase3_config_sha256": phase3_sha,
                            "generator_sha256": generator_sha,
                            "source_sha256": source_sha,
                            "split_sha256": split_sha,
                        }
                        for key, value in required.items():
                            if provenance.get(key) != value:
                                raise ValueError(
                                    f"Existing generated control has "
                                    f"different {key}: {control_dir}"
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
                        generation_status = "REUSED"
                    else:
                        generated = generate_control(
                            lines,
                            train_indices=train_indices,
                            validation_indices=validation_indices,
                            reserved_indices=reserved_indices,
                            seed=seed,
                            variant_count=k,
                            top_n=int(
                                mechanism[
                                    "nomenclator_words"
                                ]
                            ),
                            code_alphabet_size=int(
                                mechanism[
                                    "code_alphabet_size"
                                ]
                            ),
                            family_slot_width=int(
                                mechanism[
                                    "family_slot_width"
                                ]
                            ),
                        )

                        provenance = {
                            "phase3_config_path": str(
                                args.config
                            ),
                            "phase3_config_sha256": phase3_sha,
                            "generator_path": str(
                                Path(
                                    latent_variant.__file__
                                )
                            ),
                            "generator_sha256": generator_sha,
                            "source_path": str(
                                source_path
                            ),
                            "source_sha256": source_sha,
                            "split_path": str(
                                split_path
                            ),
                            "split_sha256": split_sha,
                            "level2_config_path": str(
                                level2_config_path
                            ),
                            "level2_config_sha256": level2_sha,
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
                        generation_status = "GENERATED"

                    if generated_summary.get(
                        "reserved_test_encoded"
                    ) is not False:
                        raise ValueError(
                            "Reserved test must not be encoded"
                        )

                    mechanism_rows.append(
                        _flatten_mechanism_diagnostics(
                            source_id=source_id,
                            language=language,
                            variant_count=k,
                            seed=seed,
                            generated_summary=generated_summary,
                        )
                    )

                    corpus_sha = sha256_file(
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
                            "phase3_config_sha256": phase3_sha,
                            "generator_sha256": generator_sha,
                            "control_corpus_sha256": corpus_sha,
                            "level2_config_sha256": level2_sha,
                        }
                        for key, value in required.items():
                            if saved.get(key) != value:
                                raise ValueError(
                                    f"Existing Phase-3 result has "
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
                                    f"latent_variant_K{k}"
                                ),
                                control_seed=seed,
                                train_units=train_units,
                                validation_units=validation_units,
                                config=level2,
                            )
                        )

                        for row in relation_rows:
                            row["variant_count"] = k
                            row["phase3_mechanism"] = (
                                "related_multi_code_nomenclator"
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
                            "experiment_id": phase3[
                                "experiment_id"
                            ],
                            "status": "exploratory",
                            "source_id": source_id,
                            "language": language,
                            "variant_count": k,
                            "seed": seed,
                            "phase3_config_sha256": phase3_sha,
                            "generator_sha256": generator_sha,
                            "source_sha256": source_sha,
                            "split_sha256": split_sha,
                            "level2_config_sha256": level2_sha,
                            "control_corpus_sha256": corpus_sha,
                            "relations": relation_rows,
                            "model_selection_detail": detail,
                            "reserved_test_encoded": False,
                            "reserved_test_parsed_or_scored": False,
                            "voynich_locked_test_accessed": False,
                            "interpretation": (
                                "Exploratory result. Do not promote a K "
                                "to confirmatory status without a separate "
                                "Phase-3B freeze."
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
                        phase3[
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

                    # Make sure reused CSV rows also carry Phase-3 identity.
                    for row in relation_rows:
                        row["variant_count"] = k
                        row["phase3_mechanism"] = (
                            "related_multi_code_nomenclator"
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

                    print(
                        f"  [{completed:2d}/{expected_runs}] "
                        f"K={k} seed={seed} "
                        f"{generation_status}/{evaluation_status}  "
                        f"tokenΔ="
                        f"{float(token_row['delta_competitor_minus_trigram_bits_per_event']):+.4f}"
                    )

            print()

        if completed != expected_runs:
            raise RuntimeError(
                f"Expected {expected_runs} runs; completed {completed}"
            )

        if len(all_relations) != expected_runs * 5:
            raise RuntimeError(
                f"Expected {expected_runs * 5} relation rows; "
                f"found {len(all_relations)}"
            )

        languages = [
            source["language"]
            for source in phase3["sources"].values()
        ]

        language_summary, global_summary = (
            summarize_relations(
                all_relations,
                variant_counts=variant_counts,
                languages=languages,
                relations=phase3[
                    "operational_relations"
                ],
            )
        )

        trajectory = trajectory_summary(
            global_summary,
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
        language_path = (
            result_root
            / "k_language_relation_summary.csv"
        )
        global_path = (
            result_root
            / "k_global_relation_summary.csv"
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

        top_summary = {
            "schema_version": "1.0",
            "experiment_id": phase3[
                "experiment_id"
            ],
            "status": "exploratory",
            "phase3_config_sha256": phase3_sha,
            "generator_sha256": generator_sha,
            "level2_config_sha256": level2_sha,
            "runs": completed,
            "relation_rows": len(
                all_relations
            ),
            "variant_counts": variant_counts,
            "trajectory": trajectory,
            "primary_prediction": phase3[
                "primary_prediction"
            ],
            "reserved_control_test_encoded": False,
            "reserved_control_test_parsed_or_scored": False,
            "voynich_locked_test_accessed": False,
            "no_single_similarity_score": True,
            "automatic_winner_selected": False,
            "next_step": (
                "If a stable mechanism region is scientifically "
                "interesting, write and freeze a separate Phase-3B "
                "confirmatory protocol before using any reserved-test data."
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
        print("PHASE 3A EXPLORATION COMPLETE")
        print("=" * 96)
        print(f"Runs:              {completed}")
        print(f"Relations:         {len(all_relations)}")
        print(f"Run matrix:        {relation_matrix_path}")
        print(f"Mechanism diag:    {mechanism_path}")
        print(f"K summary:         {global_path}")
        print(f"Trajectory:        {summary_path}")
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
            "Do not select/freeze a confirmatory K from this runner. "
            "Phase 3A is exploratory."
        )
        return 0

    except KeyboardInterrupt:
        print(
            "\nPHASE-3A INTERRUPTED. Existing hash-matched controls/results "
            "are reusable; rerun the same command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"PHASE-3A ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
