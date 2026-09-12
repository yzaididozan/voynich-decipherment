#!/usr/bin/env python3
"""Run frozen Level-2 predictive hierarchy across all 140 controls.

Operational relations:
  trigram > HMM (space-free)
  trigram > HMM (token-aware)
  trigram > slot grammar
  trigram > exact token Markov
  trigram > local copy-edit

All seven cipher families, all four languages, and all five frozen cipher keys
are evaluated. Completed control runs are resumable: a rerun verifies and
reuses their result artifacts rather than refitting them.

This is intentionally computationally expensive because it repeats the frozen
Voynich model-selection grids instead of using the Level-1 result to select a
favored cipher family.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.level2_predictive_controls import (
    evaluate_one_control,
    read_control_units,
    sha256_file,
    summarize_level2,
    write_csv,
)
from src.models.ngram import (
    BaselineSequence,
    WittenBellNGram,
    evaluate_model,
)


DEFAULT_CONFIG = Path(
    "configs/evaluation/level2_predictive_controls_v1.json"
)


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true", "1", "yes",
    }


def normalize_loaded_relation(row: dict) -> dict:
    output = dict(row)

    for key in (
        "control_seed",
        "selected_hyperparameter",
        "bootstrap_units",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        if key in output and output[key] != "":
            output[key] = int(output[key])

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
        if key in output and output[key] != "":
            output[key] = float(output[key])

    for key in (
        "direction_match_voynich",
        "strong_direction_match_voynich",
    ):
        if key in output:
            output[key] = as_bool(output[key])

    return output


def require_single_order_ngram_support() -> None:
    """Fail early if the repository predates the known single-order fix."""
    model = WittenBellNGram(max_order=3).fit(
        [("A", "B", "C"), ("A", "B", "D")]
    )
    example = BaselineSequence(
        leaf_group="preflight",
        folio="preflight",
        locus="preflight",
        symbols=("A", "B", "C"),
        is_boundary=(False, False, False),
    )
    try:
        rows, _, _ = evaluate_model(
            model,
            [example],
            orders=(3,),
            view="level2_preflight",
        )
    except StopIteration as exc:
        raise ValueError(
            "src/models/ngram.py predates the project's existing "
            "single-nonunigram-order evaluation fix. Apply the already-"
            "frozen ngram evaluate_model patch before Level 2."
        ) from exc

    if len(rows) != 1 or rows[0]["order"] != 3:
        raise ValueError(
            "Unexpected ngram single-order evaluation behavior"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        require_single_order_ngram_support()

        config_bytes = args.config.read_bytes()
        config_sha = sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes.decode("utf-8"))

        if config.get("freeze_id") != "level2-predictive-controls-v1":
            raise ValueError("Unexpected Level-2 freeze_id")

        expected_runs = int(
            config["expected_control_runs"]
        )
        expected_relations = list(
            config["operational_relations"]
        )

        split_root = Path(config["split_root"])
        split_manifest_path = split_root / "manifest.json"
        if not split_manifest_path.is_file():
            raise ValueError(
                "Frozen Level-2 split manifest missing. Run:\n"
                "  python scripts/freeze_level2_control_splits.py"
            )

        split_manifest = json.loads(
            split_manifest_path.read_text(
                encoding="utf-8"
            )
        )
        if split_manifest.get("config_sha256") != config_sha:
            raise ValueError(
                "Frozen split manifest config hash differs from "
                "the Level-2 config"
            )

        control_root = Path(config["control_root"])
        output_root = Path(config["output_root"])
        run_root = output_root / "runs"
        run_root.mkdir(parents=True, exist_ok=True)

        all_relations: List[dict] = []
        run_rows: List[dict] = []
        completed_runs = 0

        print("=" * 96)
        print("VOYAGER LEVEL 2 — PREDICTIVE HIERARCHY CONTROL TEST")
        print("=" * 96)
        print(f"Freeze ID:        {config['freeze_id']}")
        print(f"Config SHA:       {config_sha}")
        print("Languages:        4")
        print("Cipher families:  7")
        print("Keys/family:      5")
        print(f"Control runs:     {expected_runs}")
        print("Relations/run:    5 operational (4 conceptual; HMM has 2 views)")
        print(
            "Bootstrap:        "
            f"{config['bootstrap']['replicates']:,} paired source-line replicates"
        )
        print("Reserved test:    NOT PARSED OR SCORED")
        print("Voynich test:     NOT ACCESSED")
        print()

        for source_id, source_cfg in config[
            "source_units"
        ].items():
            language = source_cfg["language"]

            split_path = split_root / f"{source_id}.json"
            if not split_path.is_file():
                raise ValueError(
                    f"Split file missing: {split_path}"
                )
            split_payload = json.loads(
                split_path.read_text(encoding="utf-8")
            )
            if split_payload.get("config_sha256") != config_sha:
                raise ValueError(
                    f"{source_id}: split/config hash mismatch"
                )
            if (
                split_payload.get("source_sha256")
                != source_cfg["source_sha256"]
            ):
                raise ValueError(
                    f"{source_id}: split/source hash mismatch"
                )

            control_manifest_path = (
                control_root / source_id / "manifest.json"
            )
            if not control_manifest_path.is_file():
                raise ValueError(
                    f"Control manifest missing: "
                    f"{control_manifest_path}"
                )
            control_manifest = json.loads(
                control_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            if (
                control_manifest.get("source_sha256")
                != source_cfg["source_sha256"]
            ):
                raise ValueError(
                    f"{source_id}: controls were generated from "
                    "different source bytes"
                )

            runs = control_manifest.get("runs", [])
            expected_pairs = {
                (family, int(seed))
                for family in config["families"]
                for seed in config["control_seeds"]
            }
            observed_pairs = {
                (str(run["family"]), int(run["seed"]))
                for run in runs
            }
            if observed_pairs != expected_pairs:
                missing = sorted(
                    expected_pairs - observed_pairs
                )
                extra = sorted(
                    observed_pairs - expected_pairs
                )
                raise ValueError(
                    f"{source_id}: control-run set mismatch; "
                    f"missing={missing}, extra={extra}"
                )

            run_lookup = {
                (
                    str(run["family"]),
                    int(run["seed"]),
                ): run
                for run in runs
            }

            print(
                f"[{language}] {source_id}: "
                f"validation units="
                f"{split_payload['counts']['validation']}"
            )

            for family in config["families"]:
                for control_seed in config[
                    "control_seeds"
                ]:
                    control_seed = int(control_seed)
                    run = run_lookup[
                        (family, control_seed)
                    ]
                    input_run_dir = Path(run["path"])
                    corpus_path = (
                        input_run_dir / "corpus.jsonl"
                    )
                    control_summary_path = (
                        input_run_dir / "summary.json"
                    )

                    if (
                        not corpus_path.is_file()
                        or not control_summary_path.is_file()
                    ):
                        raise ValueError(
                            f"Missing control input for "
                            f"{source_id}/{family}/{control_seed}"
                        )

                    control_summary = json.loads(
                        control_summary_path.read_text(
                            encoding="utf-8"
                        )
                    )
                    if (
                        control_summary.get(
                            "round_trip_verified"
                        )
                        is not True
                    ):
                        raise ValueError(
                            f"{source_id}/{family}/{control_seed}: "
                            "cipher round-trip is not verified"
                        )
                    if (
                        control_summary.get("source_sha256")
                        != source_cfg["source_sha256"]
                    ):
                        raise ValueError(
                            f"{source_id}/{family}/{control_seed}: "
                            "input source hash mismatch"
                        )

                    corpus_sha = sha256_file(corpus_path)
                    out_dir = (
                        run_root
                        / source_id
                        / family
                        / f"seed_{control_seed}"
                    )
                    summary_path = out_dir / "summary.json"
                    relations_path = out_dir / "relations.csv"

                    if (
                        summary_path.is_file()
                        and relations_path.is_file()
                    ):
                        saved = json.loads(
                            summary_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        if (
                            saved.get("config_sha256")
                            != config_sha
                        ):
                            raise ValueError(
                                f"Existing Level-2 result uses "
                                f"different config: {out_dir}"
                            )
                        if (
                            saved.get("control_corpus_sha256")
                            != corpus_sha
                        ):
                            raise ValueError(
                                f"Existing Level-2 result input hash "
                                f"changed: {out_dir}"
                            )

                        relation_rows = [
                            normalize_loaded_relation(row)
                            for row in read_csv(relations_path)
                        ]
                        if {
                            row["relation"]
                            for row in relation_rows
                        } != set(expected_relations):
                            raise ValueError(
                                f"Saved relation set mismatch: {out_dir}"
                            )

                        all_relations.extend(
                            relation_rows
                        )
                        run_rows.append(
                            {
                                "source_id": source_id,
                                "language": language,
                                "family": family,
                                "control_seed": control_seed,
                                "full_run_direction_match": bool(
                                    saved[
                                        "full_run_direction_match"
                                    ]
                                ),
                                "full_run_strong_direction_match": bool(
                                    saved[
                                        "full_run_strong_direction_match"
                                    ]
                                ),
                                "reused_existing_result": True,
                            }
                        )
                        completed_runs += 1
                        print(
                            f"  [{completed_runs:3d}/{expected_runs}] "
                            f"{family} seed={control_seed}: REUSED"
                        )
                        continue

                    train_units, validation_units = (
                        read_control_units(
                            corpus_path,
                            train_indices=split_payload[
                                "train"
                            ],
                            validation_indices=split_payload[
                                "validation"
                            ],
                            expected_line_count=int(
                                split_payload[
                                    "source_line_count"
                                ]
                            ),
                        )
                    )

                    print(
                        f"  [{completed_runs + 1:3d}/{expected_runs}] "
                        f"{family} seed={control_seed}: FITTING..."
                    )

                    relation_rows, detailed = evaluate_one_control(
                        source_id=source_id,
                        language=language,
                        family=family,
                        control_seed=control_seed,
                        train_units=train_units,
                        validation_units=validation_units,
                        config=config,
                    )

                    out_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    write_csv(
                        relations_path,
                        relation_rows,
                    )

                    summary = {
                        "schema_version": "1.0",
                        "freeze_id": config["freeze_id"],
                        "config_path": str(args.config),
                        "config_sha256": config_sha,
                        "source_id": source_id,
                        "language": language,
                        "source_sha256": (
                            source_cfg["source_sha256"]
                        ),
                        "family": family,
                        "control_seed": control_seed,
                        "control_corpus_path": str(
                            corpus_path
                        ),
                        "control_corpus_sha256": corpus_sha,
                        "split_path": str(split_path),
                        "split_sha256": sha256_file(
                            split_path
                        ),
                        "train_units": len(train_units),
                        "validation_units": len(
                            validation_units
                        ),
                        "reserved_test_units": int(
                            split_payload["counts"][
                                "reserved_test"
                            ]
                        ),
                        "reserved_test_parsed_or_scored": False,
                        "voynich_locked_test_accessed": False,
                        "relations": relation_rows,
                        "full_run_direction_match": detailed[
                            "full_run_direction_match"
                        ],
                        "full_run_strong_direction_match": detailed[
                            "full_run_strong_direction_match"
                        ],
                        "model_selection_detail": detailed,
                        "selection_caveat": config[
                            "selection_caveat"
                        ],
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

                    (out_dir / "SHA256SUMS").write_text(
                        "\n".join(
                            [
                                f"{sha256_file(relations_path)}  relations.csv",
                                f"{sha256_file(summary_path)}  summary.json",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    all_relations.extend(
                        relation_rows
                    )
                    run_rows.append(
                        {
                            "source_id": source_id,
                            "language": language,
                            "family": family,
                            "control_seed": control_seed,
                            "full_run_direction_match": detailed[
                                "full_run_direction_match"
                            ],
                            "full_run_strong_direction_match": detailed[
                                "full_run_strong_direction_match"
                            ],
                            "reused_existing_result": False,
                        }
                    )
                    completed_runs += 1

                    strong_count = sum(
                        bool(
                            row[
                                "strong_direction_match_voynich"
                            ]
                        )
                        for row in relation_rows
                    )
                    print(
                        f"      strong Voynich-direction relations: "
                        f"{strong_count}/5"
                    )

            print()

        if completed_runs != expected_runs:
            raise RuntimeError(
                f"Expected {expected_runs} completed runs; "
                f"found {completed_runs}"
            )
        if len(all_relations) != (
            expected_runs * len(expected_relations)
        ):
            raise RuntimeError(
                "Unexpected relation-row count: "
                f"{len(all_relations)}"
            )

        languages = [
            config["source_units"][source_id][
                "language"
            ]
            for source_id in config["source_units"]
        ]

        language_rows, family_rows = summarize_level2(
            all_relations,
            families=config["families"],
            relations=expected_relations,
            languages=languages,
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )
        relation_matrix_path = (
            output_root / "run_relation_matrix.csv"
        )
        run_summary_path = (
            output_root / "control_run_summary.csv"
        )
        language_summary_path = (
            output_root
            / "language_relation_replication.csv"
        )
        family_summary_path = (
            output_root
            / "family_hierarchy_summary.csv"
        )

        write_csv(
            relation_matrix_path,
            all_relations,
        )
        write_csv(
            run_summary_path,
            run_rows,
        )
        write_csv(
            language_summary_path,
            language_rows,
        )
        write_csv(
            family_summary_path,
            family_rows,
        )

        full_outcomes = {
            row["family"]: bool(
                row[
                    "family_full_hierarchy_reproduction"
                ]
            )
            for row in family_rows
            if row["relation"] == "__FULL_HIERARCHY__"
        }

        top_summary = {
            "schema_version": "1.0",
            "freeze_id": config["freeze_id"],
            "config_sha256": config_sha,
            "control_runs": completed_runs,
            "operational_relations": expected_relations,
            "relation_rows": len(
                all_relations
            ),
            "replication_rule": config[
                "replication_rule"
            ],
            "selection_caveat": config[
                "selection_caveat"
            ],
            "voynich_reference": config[
                "voynich_reference"
            ],
            "family_full_hierarchy_reproduction": (
                full_outcomes
            ),
            "reserved_test_parsed_or_scored": False,
            "voynich_locked_test_accessed": False,
            "interpretation_guardrails": config[
                "interpretation_guardrails"
            ],
            "artifacts": [
                "run_relation_matrix.csv",
                "control_run_summary.csv",
                "language_relation_replication.csv",
                "family_hierarchy_summary.csv",
                "runs/<source>/<family>/seed_<seed>/...",
            ],
        }
        top_summary_path = output_root / "summary.json"
        top_summary_path.write_text(
            json.dumps(
                top_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        checksum_paths = (
            relation_matrix_path,
            run_summary_path,
            language_summary_path,
            family_summary_path,
            top_summary_path,
        )
        (output_root / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print("LEVEL-2 PREDICTIVE HIERARCHY COMPLETE")
        print("=" * 96)
        print(f"Control runs:   {completed_runs}")
        print(f"Relations:      {len(all_relations)}")
        print(f"Run matrix:     {relation_matrix_path}")
        print(f"Family summary: {family_summary_path}")
        print("Reserved test:  NOT PARSED OR SCORED")
        print("Voynich test:   NOT ACCESSED")
        print()
        print("Full hierarchy reproduction:")
        for family in config["families"]:
            print(
                f"  {family:18s} "
                f"{'YES' if full_outcomes[family] else 'NO'}"
            )
        return 0

    except KeyboardInterrupt:
        print(
            "\nLEVEL-2 INTERRUPTED. Completed run directories are "
            "resumable; rerun the same command.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"LEVEL-2 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
