#!/usr/bin/env python3
"""Run VOYAGER Phase 3B locked reserved-control confirmation.

Default behavior is PRE-FLIGHT ONLY. The reserved split is encoded/scored only
when --execute-reserved-test is supplied after the protocol has been frozen.

The confirmatory representation is the exact Phase 3A.6 K=16 recurrence-
avoiding mechanism. TRAIN is encoded first; RESERVED starts from the final
TRAIN occurrence counters. Development VALIDATION does not advance counters and
is never supplied to the evaluator.

Voynich test_LOCKED is never accessed by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Phase 3B config is YAML. Install the repository's YAML dependency "
        "(PyYAML) before running this script."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ciphers import recurrence_avoiding_surface_allocation as mechanism
from src.evaluation.level2_predictive_controls import evaluate_one_control, write_csv

DEFAULT_CONFIG = Path("configs/phase_3b_confirmatory.yaml")
GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_relation_row(row: Mapping[str, object]) -> dict:
    output = dict(row)
    for key in (
        "control_seed",
        "selected_hyperparameter",
        "bootstrap_units",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        if key in output and output[key] not in ("", None):
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
        if key in output and output[key] not in ("", None):
            output[key] = float(output[key])

    for key in ("direction_match_voynich", "strong_direction_match_voynich"):
        if key in output:
            output[key] = as_bool(output[key])
    return output


def verify_exact_mapping(observed: Mapping[str, object], expected: Mapping[str, object], *, prefix: str = "") -> None:
    for key, expected_value in expected.items():
        label = f"{prefix}.{key}" if prefix else key
        if key not in observed:
            raise ValueError(f"Required field missing from Phase 3A.6 summary: {label}")
        observed_value = observed[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(observed_value, Mapping):
                raise ValueError(f"Expected mapping at {label}")
            verify_exact_mapping(observed_value, expected_value, prefix=label)
        elif observed_value != expected_value:
            raise ValueError(
                f"Frozen Phase 3A.6 summary mismatch at {label}: "
                f"expected {expected_value!r}, observed {observed_value!r}"
            )


def preflight(config_path: Path, config: Mapping[str, object]) -> dict:
    if config.get("experiment_id") != "phase-3b-reserved-control-confirmation-v1":
        raise ValueError("Unexpected Phase 3B experiment_id")
    if config.get("status") != "confirmatory":
        raise ValueError("Phase 3B status must be confirmatory")

    candidate = config["candidate_mechanism"]
    if int(candidate["variant_count"]) != 16:
        raise ValueError("Phase 3B K must remain exactly 16")
    if int(candidate["suffix_length_glyphs"]) != 1:
        raise ValueError("Phase 3B suffix length must remain exactly one glyph")

    phase3a6 = config["frozen_phase3a6"]
    level2_cfg = config["frozen_level2"]

    phase3a6_config_path = Path(phase3a6["config_path"])
    phase3a6_generator_path = Path(phase3a6["generator_path"])
    level2_path = Path(level2_cfg["config_path"])

    checks = (
        (phase3a6_config_path, phase3a6["expected_config_sha256"], "Phase 3A.6 config"),
        (phase3a6_generator_path, phase3a6["expected_generator_sha256"], "Phase 3A.6 generator"),
        (level2_path, level2_cfg["expected_config_sha256"], "Level-2 config"),
    )
    observed_hashes = {}
    for path, expected_hash, label in checks:
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise ValueError(f"{label} SHA mismatch: expected {expected_hash}, observed {digest}")
        observed_hashes[label] = digest

    # Verify the imported module is the exact frozen source file.
    imported_generator = Path(mechanism.__file__).resolve()
    if sha256_file(imported_generator) != phase3a6["expected_generator_sha256"]:
        raise ValueError("Imported Phase 3A.6 generator does not match frozen SHA")

    exploration_summary_path = Path(phase3a6["exploration_result_root"]) / "summary.json"
    if not exploration_summary_path.is_file():
        raise FileNotFoundError(f"Completed Phase 3A.6 summary missing: {exploration_summary_path}")
    exploration_summary = json.loads(exploration_summary_path.read_text(encoding="utf-8"))
    verify_exact_mapping(
        exploration_summary,
        phase3a6["required_exploration_summary"],
    )
    if exploration_summary.get("phase3a6_config_sha256") != phase3a6["expected_config_sha256"]:
        raise ValueError("Phase 3A.6 result summary was produced by a different config")
    if exploration_summary.get("generator_sha256") != phase3a6["expected_generator_sha256"]:
        raise ValueError("Phase 3A.6 result summary was produced by a different generator")

    level2 = json.loads(level2_path.read_text(encoding="utf-8"))
    if level2.get("freeze_id") != level2_cfg["required_freeze_id"]:
        raise ValueError("Unexpected Level-2 freeze_id")
    if list(level2["operational_relations"]) != list(level2_cfg["operational_relations"]):
        raise ValueError("Phase 3B relation set differs from frozen Level 2")

    seeds = [int(x) for x in config["seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Phase 3B requires exactly five distinct frozen seeds")
    if int(config["expected_runs"]) != len(config["sources"]) * len(seeds):
        raise ValueError("expected_runs does not match source/seed grid")
    if int(config["expected_relation_rows"]) != int(config["expected_runs"]) * len(level2_cfg["operational_relations"]):
        raise ValueError("expected_relation_rows does not match run/relation grid")

    return {
        "phase3b_config_sha256": sha256_file(config_path),
        "phase3a6_config_sha256": observed_hashes["Phase 3A.6 config"],
        "phase3a6_generator_sha256": observed_hashes["Phase 3A.6 generator"],
        "phase3a6_exploration_summary_sha256": sha256_file(exploration_summary_path),
        "level2_config_sha256": observed_hashes["Level-2 config"],
        "level2": level2,
    }


def encode_partition(
    source_lines: Sequence[PlainLine],
    line_indices: Sequence[int],
    *,
    key: Mapping[str, object],
    seed: int,
    counters: MutableMapping[GlyphToken, int],
) -> List[tuple]:
    """Encode one split online using only public frozen Phase-3A.6 primitives."""
    units: List[tuple] = []
    for line_index in sorted(int(x) for x in line_indices):
        plain_line = source_lines[line_index - 1]
        cipher_line: List[GlyphToken] = []
        for plaintext_token in plain_line:
            base = tuple(mechanism.encode_base(plaintext_token, key=key))
            occurrence_number = int(counters.get(base, 0))
            cipher_token = tuple(
                mechanism.encode_token_for_occurrence(
                    plaintext_token,
                    key=key,
                    seed=int(seed),
                    occurrence_number=occurrence_number,
                )
            )
            counters[base] = occurrence_number + 1
            if tuple(mechanism.strip_suffix(cipher_token, key=key)) != base:
                raise RuntimeError("Base-preservation invariant failed")
            if mechanism.decode_token(cipher_token, key=key) != plaintext_token:
                raise RuntimeError("Round-trip invariant failed")
            if len(cipher_token) != len(base) + 1:
                raise RuntimeError("One-glyph suffix-length invariant failed")
            cipher_line.append(cipher_token)
        units.append((int(line_index), tuple(cipher_line)))
    return units


def write_confirmatory_control(
    output_dir: Path,
    *,
    total_lines: int,
    train_units: Sequence[tuple],
    reserved_units: Sequence[tuple],
    validation_indices: Sequence[int],
    key: Mapping[str, object],
    provenance: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = {int(i): tuple(tokens) for i, tokens in train_units}
    reserved = {int(i): tuple(tokens) for i, tokens in reserved_units}
    validation = {int(x) for x in validation_indices}

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for line_index in range(1, total_lines + 1):
            if line_index in train:
                row = {"line_index": line_index, "split": "train", "tokens": [list(t) for t in train[line_index]]}
            elif line_index in reserved:
                row = {"line_index": line_index, "split": "reserved_test", "tokens": [list(t) for t in reserved[line_index]]}
            elif line_index in validation:
                row = {"line_index": line_index, "development_validation_opaque": True}
            else:
                raise RuntimeError(f"Line {line_index} belongs to no Phase 3B split")
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    key_path = output_dir / "key.json"
    key_path.write_text(json.dumps(dict(key), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "experiment_id": "phase-3b-reserved-control-confirmation-v1",
        "status": "confirmatory",
        "provenance": dict(provenance),
        "train_lines_encoded": len(train_units),
        "reserved_lines_encoded": len(reserved_units),
        "development_validation_lines_encoded": 0,
        "reserved_continues_directly_from_final_train_state": True,
        "voynich_locked_test_accessed": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (corpus_path, key_path, summary_path)
        ) + "\n",
        encoding="utf-8",
    )


def verify_key_lineage(key: Mapping[str, object], phase3a6_key_path: Path) -> dict:
    if not phase3a6_key_path.is_file():
        raise FileNotFoundError(f"Frozen Phase 3A.6 key missing: {phase3a6_key_path}")
    parent = json.loads(phase3a6_key_path.read_text(encoding="utf-8"))
    exact_fields = (
        "mono_forward",
        "mono_reverse",
        "train_character_alphabet",
        "master_suffixes",
        "parent_phase3a4_master_key_sha256",
        "allocation_namespace",
        "variant_count",
        "suffix_length_glyphs",
        "allocation_key_sha256",
        "validation_unseen_character_policy",
    )
    for field in exact_fields:
        if key.get(field) != parent.get(field):
            raise ValueError(f"Confirmatory key differs from frozen Phase 3A.6 key at {field}")
    return {
        "phase3a6_key_sha256": sha256_file(phase3a6_key_path),
        "key_lineage_exact": True,
    }


def summarize_relations(
    rows: Sequence[Mapping[str, object]],
    *,
    languages: Sequence[str],
    relations: Sequence[str],
    seeds_per_language: int,
) -> tuple[List[dict], List[dict], List[dict], bool]:
    language_rows: List[dict] = []
    relation_rows: List[dict] = []
    global_rows: List[dict] = []

    for relation in relations:
        relation_group = [row for row in rows if row["relation"] == relation]
        expected = len(languages) * seeds_per_language
        if len(relation_group) != expected:
            raise ValueError(f"{relation}: expected {expected} runs, found {len(relation_group)}")
        deltas = [float(row["delta_competitor_minus_trigram_bits_per_event"]) for row in relation_group]
        global_rows.append({
            "relation": relation,
            "runs": len(relation_group),
            "mean_delta_bits_per_event": mean(deltas),
            "median_delta_bits_per_event": median(deltas),
            "sd_delta_bits_per_event": pstdev(deltas),
            "min_delta_bits_per_event": min(deltas),
            "max_delta_bits_per_event": max(deltas),
            "positive_direction_runs": sum(bool(row["direction_match_voynich"]) for row in relation_group),
            "strong_direction_runs": sum(bool(row["strong_direction_match_voynich"]) for row in relation_group),
        })

        replicated_languages: List[str] = []
        for language in languages:
            group = [row for row in relation_group if row["language"] == language]
            if len(group) != seeds_per_language:
                raise ValueError(f"{relation}/{language}: expected {seeds_per_language} seeds")
            language_deltas = [float(row["delta_competitor_minus_trigram_bits_per_event"]) for row in group]
            strong = sum(bool(row["strong_direction_match_voynich"]) for row in group)
            replicated = strong >= 4
            if replicated:
                replicated_languages.append(language)
            language_rows.append({
                "relation": relation,
                "language": language,
                "seeds": seeds_per_language,
                "mean_delta_bits_per_event": mean(language_deltas),
                "sd_delta_bits_per_event": pstdev(language_deltas),
                "positive_direction_seeds": sum(bool(row["direction_match_voynich"]) for row in group),
                "strong_direction_seeds": strong,
                "language_relation_replicated": replicated,
            })

        robust = len(replicated_languages) >= 3
        relation_rows.append({
            "relation": relation,
            "languages_replicated": len(replicated_languages),
            "replicated_languages": ";".join(replicated_languages),
            "relation_robust_3_of_4": robust,
        })

    full_hierarchy = all(bool(row["relation_robust_3_of_4"]) for row in relation_rows)
    return language_rows, relation_rows, global_rows, full_hierarchy


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VOYAGER Phase 3B reserved-control confirmation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--execute-reserved-test",
        action="store_true",
        help="After protocol freeze, unlock encoding/scoring of the reserved controls.",
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        frozen = preflight(args.config, config)

        print("=" * 96)
        print("VOYAGER PHASE 3B — LOCKED RESERVED-CONTROL CONFIRMATION")
        print("=" * 96)
        print(f"Experiment:       {config['experiment_id']}")
        print(f"Status:           {config['status'].upper()}")
        print(f"Phase 3B cfg SHA: {frozen['phase3b_config_sha256']}")
        print(f"Phase 3A.6 cfg:   {frozen['phase3a6_config_sha256']}")
        print(f"Phase 3A.6 gen:   {frozen['phase3a6_generator_sha256']}")
        print(f"Level-2 config:   {frozen['level2_config_sha256']}")
        print("Variant count:    16 (FROZEN)")
        print("Suffix length:    1 glyph (FROZEN)")
        print("Allocation:       balanced without replacement per lexical base (FROZEN)")
        print("Reserved state:   direct continuation from final TRAIN state")
        print("Dev validation:   BYPASSED / NOT USED")
        print("Voynich test:     NOT ACCESSED")
        print()

        if not args.execute_reserved_test:
            print("PRE-FLIGHT PASS — RESERVED HAS NOT BEEN ENCODED OR SCORED.")
            print("Freeze/commit this protocol and record its hashes before rerunning with:")
            print("  python scripts/run_phase_3b_confirmatory_control.py --execute-reserved-test")
            return 0

        source_root = Path(config["source_root"])
        split_root = Path(config["split_root"])
        phase3a6_control_root = Path(config["phase3a6_control_root"])
        confirmatory_control_root = Path(config["confirmatory_control_root"])
        result_root = Path(config["result_output_root"])
        run_root = result_root / "runs"
        result_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)

        final_summary_path = result_root / "summary.json"
        if final_summary_path.exists():
            raise RuntimeError(
                "Phase 3B already has a final summary. Refusing a second confirmatory execution "
                "against the same reserved set."
            )

        level2 = frozen["level2"]
        seeds = [int(x) for x in config["seeds"]]
        relations = list(config["frozen_level2"]["operational_relations"])

        relation_rows: List[dict] = []
        lineage_rows: List[dict] = []
        completed = 0

        for source_id, source_cfg in config["sources"].items():
            language = source_cfg["language"]
            source_path = source_root / source_id / "source.txt"
            if sha256_file(source_path) != source_cfg["source_sha256"]:
                raise ValueError(f"{source_id}: frozen source SHA mismatch")
            source_lines = mechanism.read_source_lines(source_path)
            if len(source_lines) != int(source_cfg["expected_lines"]):
                raise ValueError(f"{source_id}: source line-count mismatch")
            if mechanism.token_count(source_lines) != int(source_cfg["expected_tokens"]):
                raise ValueError(f"{source_id}: source token-count mismatch")

            split_path = split_root / f"{source_id}.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            if split.get("source_sha256") != source_cfg["source_sha256"]:
                raise ValueError(f"{source_id}: split/source SHA mismatch")
            train_indices = [int(x) for x in split["train"]]
            validation_indices = [int(x) for x in split["validation"]]
            reserved_indices = [int(x) for x in split["reserved_test"]]
            mechanism.validate_partition(len(source_lines), train_indices, validation_indices, reserved_indices)

            expected_counts = (
                (train_indices, "expected_train_lines", "TRAIN"),
                (validation_indices, "expected_validation_lines", "VALIDATION"),
                (reserved_indices, "expected_reserved_lines", "RESERVED"),
            )
            for indices, key_name, split_name in expected_counts:
                if len(indices) != int(source_cfg[key_name]):
                    raise ValueError(f"{source_id}: {split_name} line-count mismatch")

            print(
                f"[{language}] {source_id}: train={len(train_indices)} "
                f"dev_validation={len(validation_indices)} reserved={len(reserved_indices)}"
            )

            train_lines_for_fit = [source_lines[i - 1] for i in sorted(train_indices)]

            for seed in seeds:
                key = mechanism.build_master_key(train_lines_for_fit, seed=seed)
                phase3a6_key_path = (
                    phase3a6_control_root / source_id / "balanced16" / f"seed_{seed}" / "key.json"
                )
                lineage = verify_key_lineage(key, phase3a6_key_path)

                counters: Counter = Counter()
                train_units = encode_partition(
                    source_lines,
                    train_indices,
                    key=key,
                    seed=seed,
                    counters=counters,
                )
                final_train_state = dict(counters)
                reserved_units = encode_partition(
                    source_lines,
                    reserved_indices,
                    key=key,
                    seed=seed,
                    counters=counters,
                )

                # Confirm the first reserved occurrence for each base starts exactly
                # from its final TRAIN counter; encode_partition mutates only after use.
                # A stable digest records the frozen TRAIN state used for confirmation.
                train_state_payload = [
                    [list(base), int(count)]
                    for base, count in sorted(final_train_state.items(), key=lambda item: repr(item[0]))
                ]
                train_state_sha = sha256(
                    json.dumps(train_state_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ).hexdigest()

                control_dir = (
                    confirmatory_control_root / source_id / "balanced16" / f"seed_{seed}"
                )
                provenance = {
                    "phase3b_config_sha256": frozen["phase3b_config_sha256"],
                    "phase3a6_config_sha256": frozen["phase3a6_config_sha256"],
                    "phase3a6_generator_sha256": frozen["phase3a6_generator_sha256"],
                    "phase3a6_exploration_summary_sha256": frozen["phase3a6_exploration_summary_sha256"],
                    "level2_config_sha256": frozen["level2_config_sha256"],
                    "source_sha256": source_cfg["source_sha256"],
                    "split_sha256": sha256_file(split_path),
                    "frozen_train_occurrence_state_sha256": train_state_sha,
                    "reserved_state_policy": "direct_from_final_train_state_without_development_validation",
                }
                write_confirmatory_control(
                    control_dir,
                    total_lines=len(source_lines),
                    train_units=train_units,
                    reserved_units=reserved_units,
                    validation_indices=validation_indices,
                    key=key,
                    provenance=provenance,
                )

                run_dir = run_root / source_id / "balanced16" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                relations_path = run_dir / "relations.csv"
                run_summary_path = run_dir / "summary.json"

                run_relations, detail = evaluate_one_control(
                    source_id=source_id,
                    language=language,
                    family="phase3b_recurrence_avoiding_surface_allocation",
                    control_seed=seed,
                    train_units=train_units,
                    validation_units=reserved_units,  # API name only; these are RESERVED units.
                    config=level2,
                )
                for row in run_relations:
                    row["phase3b_evaluation_split"] = "reserved_test"
                    row["phase3b_allocation"] = "balanced16"
                write_csv(relations_path, run_relations)

                run_summary = {
                    "schema_version": "1.0",
                    "experiment_id": config["experiment_id"],
                    "status": "confirmatory",
                    "source_id": source_id,
                    "language": language,
                    "seed": seed,
                    **provenance,
                    **lineage,
                    "train_lines": len(train_units),
                    "reserved_lines": len(reserved_units),
                    "development_validation_lines_used": 0,
                    "relations": run_relations,
                    "model_selection_detail": detail,
                    "voynich_locked_test_accessed": False,
                }
                run_summary_path.write_text(
                    json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (run_dir / "SHA256SUMS").write_text(
                    f"{sha256_file(relations_path)}  relations.csv\n"
                    f"{sha256_file(run_summary_path)}  summary.json\n",
                    encoding="utf-8",
                )

                normalized = [normalize_relation_row(row) for row in run_relations]
                if {row["relation"] for row in normalized} != set(relations):
                    raise ValueError(f"Unexpected relation set: {run_dir}")
                relation_rows.extend(normalized)
                lineage_rows.append({
                    "source_id": source_id,
                    "language": language,
                    "seed": seed,
                    **lineage,
                    "frozen_train_occurrence_state_sha256": train_state_sha,
                    "development_validation_used": False,
                    "reserved_continues_directly_from_train": True,
                })
                completed += 1

                token_row = next(row for row in normalized if row["relation"] == "token_markov")
                print(
                    f"  [{completed:2d}/{config['expected_runs']}] seed={seed} "
                    f"LINEAGE-PASS reserved tokenΔ="
                    f"{float(token_row['delta_competitor_minus_trigram_bits_per_event']):+.4f} "
                    f"strong={'YES' if token_row['strong_direction_match_voynich'] else 'NO'}"
                )
            print()

        if completed != int(config["expected_runs"]):
            raise RuntimeError(f"Expected {config['expected_runs']} runs; completed {completed}")
        if len(relation_rows) != int(config["expected_relation_rows"]):
            raise RuntimeError(
                f"Expected {config['expected_relation_rows']} relation rows; found {len(relation_rows)}"
            )

        languages = [src["language"] for src in config["sources"].values()]
        language_summary, replication_summary, global_summary, full_hierarchy = summarize_relations(
            relation_rows,
            languages=languages,
            relations=relations,
            seeds_per_language=len(seeds),
        )

        outputs = {
            "run_relation_matrix.csv": relation_rows,
            "language_relation_summary.csv": language_summary,
            "relation_replication_summary.csv": replication_summary,
            "global_relation_summary.csv": global_summary,
            "lineage_verification.csv": lineage_rows,
        }
        written: List[Path] = []
        for filename, rows in outputs.items():
            path = result_root / filename
            write_csv(path, rows)
            written.append(path)

        decision = "PHASE_3B_PASS" if full_hierarchy else "PHASE_3B_FAIL"
        token_language = {
            row["language"]: {
                "mean_delta_bits_per_event": row["mean_delta_bits_per_event"],
                "positive_direction_seeds": row["positive_direction_seeds"],
                "strong_direction_seeds": row["strong_direction_seeds"],
                "language_relation_replicated": row["language_relation_replicated"],
            }
            for row in language_summary
            if row["relation"] == "token_markov"
        }
        summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": "confirmatory",
            "decision": decision,
            "phase3b_config_sha256": frozen["phase3b_config_sha256"],
            "phase3a6_config_sha256": frozen["phase3a6_config_sha256"],
            "phase3a6_generator_sha256": frozen["phase3a6_generator_sha256"],
            "phase3a6_exploration_summary_sha256": frozen["phase3a6_exploration_summary_sha256"],
            "level2_config_sha256": frozen["level2_config_sha256"],
            "runs": completed,
            "relation_rows": len(relation_rows),
            "variant_count": 16,
            "suffix_length_glyphs": 1,
            "allocation": "balanced_without_replacement_per_lexical_base",
            "confirmatory_split": "reserved_test",
            "reserved_state_policy": "direct_from_final_train_state_without_development_validation",
            "development_validation_used": False,
            "token_markov_by_language": token_language,
            "full_inherited_level2_hierarchy": bool(full_hierarchy),
            "confirmatory_decision_rule": config["confirmatory_decision_rule"],
            "voynich_locked_test_accessed": False,
            "post_result_tuning_authorized": False,
        }
        final_summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(final_summary_path)

        checksum_path = result_root / "SHA256SUMS"
        checksum_path.write_text(
            "\n".join(f"{sha256_file(path)}  {path.name}" for path in written) + "\n",
            encoding="utf-8",
        )

        print("=" * 96)
        print(f"{decision.replace('_', ' ')}")
        print("=" * 96)
        print(f"Runs:              {completed}")
        print(f"Relations:         {len(relation_rows)}")
        print(f"Full hierarchy:    {'YES' if full_hierarchy else 'NO'}")
        print("Dev validation:    NOT USED")
        print("Voynich test:      NOT ACCESSED")
        print()
        print("Token-Markov by language:")
        for language in languages:
            row = token_language[language]
            print(
                f"  {language:<10} meanΔ={float(row['mean_delta_bits_per_event']):+.6f} "
                f"strong={int(row['strong_direction_seeds'])}/5 "
                f"replicated={'YES' if row['language_relation_replicated'] else 'NO'}"
            )
        print()
        if full_hierarchy:
            print("PASS is retained as confirmation on reserved controls. Voynich remains locked.")
        else:
            print("FAIL is retained as the confirmatory outcome. Do not tune against this reserved set.")
        return 0

    except KeyboardInterrupt:
        print("\nPHASE-3B INTERRUPTED. Do not alter the frozen protocol before any exact-provenance resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"PHASE-3B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
