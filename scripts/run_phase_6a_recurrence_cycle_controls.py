#!/usr/bin/env python3
"""VOYAGER Phase 6A recurrence-cycle fingerprint control calibration.

This runner:
- verifies the frozen K=16 candidate and two prespecified matched null modules;
- reuses the already-accessed Phase-5A historical-language controls;
- fits the inherited base key on TRAIN only;
- encodes TRAIN then TEST with continuous per-base occurrence counters;
- measures observable suffix recurrence at lags 1..16;
- performs no Voynich access.

It is a mechanism-control calibration, not a Voynich test and not a fresh-data
prospective validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import yaml

# Ensure the repository root is importable when this file is executed directly:
#     python scripts/run_phase_6a_recurrence_cycle_controls.py
#
# Without this, Python may put only the scripts/ directory on sys.path, causing
# `from src...` imports to fail even when the command is run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ciphers import recurrence_avoiding_surface_allocation as fixed_cycle
from src.ciphers import recurrence_cycle_shuffle_null as cycle_shuffle
from src.ciphers import recurrence_iid_null as iid_null


DEFAULT_CONFIG = Path("configs/phase_6a_recurrence_cycle_prediction.yaml")
GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]

MODEL_MODULES = {
    "fixed_cycle_k16": fixed_cycle,
    "cycle_shuffle_k16": cycle_shuffle,
    "iid_k16": iid_null,
}

FORBIDDEN_VOYNICH_PATH_FRAGMENTS = (
    "data/raw/zl",
    "data/reference/sta1",
    "data/splits/v1",
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(
            f"{label} SHA-256 mismatch\n"
            f"  expected: {expected}\n"
            f"  observed: {observed}"
        )
    return observed


def lexical_path_is_forbidden(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(fragment in normalized for fragment in FORBIDDEN_VOYNICH_PATH_FRAGMENTS)


def validate_no_voynich_inputs(config: Mapping[str, object]) -> None:
    corpus = config["control_corpus"]
    candidate = config["frozen_candidate"]
    nulls = config["matched_nulls"]

    input_paths = [
        str(corpus["manifest_path"]),
        str(candidate["config_path"]),
        str(candidate["generator_path"]),
        str(nulls["cycle_shuffle"]["module_path"]),
        str(nulls["iid"]["module_path"]),
    ]
    for value in input_paths:
        if lexical_path_is_forbidden(value):
            raise ValueError(f"Voynich path is forbidden in Phase 6A: {value}")


def verify_config_contract(config: Mapping[str, object]) -> None:
    if config.get("experiment_id") != "phase-6a-recurrence-cycle-control-calibration-v1":
        raise ValueError("Unexpected Phase 6A experiment_id")
    if config.get("status") != "confirmatory_mechanism_control_calibration":
        raise ValueError("Unexpected Phase 6A status")

    seeds = [int(value) for value in config["seeds"]]
    if seeds != list(range(40814041438, 40814041443)):
        raise ValueError("Phase 6A seeds must remain exactly 40814041438..40814041442")

    candidate = config["frozen_candidate"]
    if int(candidate["variant_count"]) != 16:
        raise ValueError("Candidate K must remain 16")
    if int(candidate["suffix_length_glyphs"]) != 1:
        raise ValueError("Candidate suffix length must remain one glyph")

    stat = config["primary_statistic"]
    if stat["name"] != "cycle16_contrast":
        raise ValueError("Primary statistic name changed")
    if int(stat["eligible_family_min_occurrences"]) != 17:
        raise ValueError("Eligible-family threshold must remain 17")
    if [int(x) for x in stat["primary_lags"]] != list(range(1, 17)):
        raise ValueError("Primary lags must remain exactly 1..16")

    suff = config["data_sufficiency"]
    if int(suff["minimum_eligible_families_per_source_seed"]) != 10:
        raise ValueError("Minimum eligible families changed")
    if int(suff["minimum_lag16_pairs_per_source_seed"]) != 64:
        raise ValueError("Minimum lag-16 pairs changed")

    decision = config["primary_decision"]
    if decision["language_replication"] != "5/5 seeds must succeed":
        raise ValueError("Language replication rule changed")
    if "20/20" not in str(decision["phase6a_pass"]):
        raise ValueError("Phase 6A must require 20/20 source-seed successes")

    validate_no_voynich_inputs(config)


def verify_frozen_modules(config: Mapping[str, object]) -> dict:
    candidate = config["frozen_candidate"]
    nulls = config["matched_nulls"]

    candidate_cfg_sha = require_file_hash(
        Path(candidate["config_path"]),
        candidate["expected_config_sha256"],
        "Frozen K=16 mechanism config",
    )
    candidate_gen_sha = require_file_hash(
        Path(candidate["generator_path"]),
        candidate["expected_generator_sha256"],
        "Frozen K=16 generator",
    )
    shuffle_sha = require_file_hash(
        Path(nulls["cycle_shuffle"]["module_path"]),
        nulls["cycle_shuffle"]["expected_sha256"],
        "Cycle-shuffle null",
    )
    iid_sha = require_file_hash(
        Path(nulls["iid"]["module_path"]),
        nulls["iid"]["expected_sha256"],
        "IID K=16 null",
    )

    return {
        "candidate_config_sha256": candidate_cfg_sha,
        "candidate_generator_sha256": candidate_gen_sha,
        "cycle_shuffle_sha256": shuffle_sha,
        "iid_null_sha256": iid_sha,
    }


def allocator_self_check() -> None:
    """Check only scheduler invariants using a synthetic glyph-base tuple."""
    base = ("A", "B", "C")
    seed = 40814041438

    fixed_first = [
        fixed_cycle.variant_index_for_occurrence(
            base,
            seed=seed,
            occurrence_number=n,
        )
        for n in range(16)
    ]
    fixed_second = [
        fixed_cycle.variant_index_for_occurrence(
            base,
            seed=seed,
            occurrence_number=n,
        )
        for n in range(16, 32)
    ]
    if sorted(fixed_first) != list(range(16)) or fixed_first != fixed_second:
        raise RuntimeError("Frozen candidate failed period-16 allocator self-check")

    for cycle in range(4):
        observed = [
            cycle_shuffle.variant_index_for_occurrence(
                base,
                seed=seed,
                occurrence_number=cycle * 16 + position,
            )
            for position in range(16)
        ]
        if sorted(observed) != list(range(16)):
            raise RuntimeError("Cycle-shuffle null violated within-cycle balance")

    iid_values = [
        iid_null.variant_index_for_occurrence(
            base,
            seed=seed,
            occurrence_number=n,
        )
        for n in range(64)
    ]
    if any(value < 0 or value >= 16 for value in iid_values):
        raise RuntimeError("IID null emitted an out-of-range variant")


def validate_manifest(config: Mapping[str, object]) -> dict:
    corpus = config["control_corpus"]
    manifest_path = Path(str(corpus["manifest_path"]))

    observed_manifest_sha = require_file_hash(
        manifest_path,
        corpus["expected_manifest_sha256"],
        "Reused Phase-5A control manifest",
    )
    manifest = read_json(manifest_path)

    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("Reused control manifest must contain exactly four sources")

    required_languages = sorted(str(x) for x in corpus["required_languages"])
    observed_languages = sorted(str(row.get("language")) for row in sources)
    if observed_languages != required_languages:
        raise ValueError(
            f"Control languages changed: {observed_languages} != {required_languages}"
        )

    required_ids = sorted(str(x) for x in corpus["required_source_ids"])
    observed_ids = sorted(str(row.get("source_id")) for row in sources)
    if observed_ids != required_ids:
        raise ValueError(
            f"Control source IDs changed: {observed_ids} != {required_ids}"
        )

    checked = []
    for row in sources:
        source_id = str(row["source_id"])
        train_path = Path(str(row["train_path"]))
        test_path = Path(str(row["test_locked_path"]))

        for value in (str(train_path), str(test_path)):
            if lexical_path_is_forbidden(value):
                raise ValueError(f"{source_id}: Voynich-like path forbidden: {value}")

        require_file_hash(
            train_path,
            str(row["train_sha256"]),
            f"{source_id} TRAIN",
        )
        require_file_hash(
            test_path,
            str(row["test_locked_sha256"]),
            f"{source_id} TEST",
        )

        train_lines = fixed_cycle.read_source_lines(train_path)
        test_lines = fixed_cycle.read_source_lines(test_path)

        if len(train_lines) != int(row["train_lines"]):
            raise ValueError(f"{source_id}: TRAIN line-count mismatch")
        if fixed_cycle.token_count(train_lines) != int(row["train_tokens"]):
            raise ValueError(f"{source_id}: TRAIN token-count mismatch")
        if len(test_lines) != int(row["test_locked_lines"]):
            raise ValueError(f"{source_id}: TEST line-count mismatch")
        if fixed_cycle.token_count(test_lines) != int(row["test_locked_tokens"]):
            raise ValueError(f"{source_id}: TEST token-count mismatch")

        checked.append(
            {
                **dict(row),
                "_train_lines": train_lines,
                "_test_lines": test_lines,
            }
        )

    return {
        "manifest_path": manifest_path,
        "manifest_sha256": observed_manifest_sha,
        "sources": checked,
    }


def encode_partition(
    lines: Sequence[PlainLine],
    *,
    key: Mapping[str, object],
    seed: int,
    allocator_module,
    counters: MutableMapping[GlyphToken, int],
) -> List[GlyphToken]:
    output: List[GlyphToken] = []

    for plain_line in lines:
        if not plain_line:
            raise ValueError("Encountered an empty source line")
        for plaintext_token in plain_line:
            base = tuple(
                fixed_cycle.encode_base(
                    plaintext_token,
                    key=key,
                )
            )
            occurrence_number = int(counters.get(base, 0))
            cipher_token = tuple(
                allocator_module.encode_token_for_occurrence(
                    plaintext_token,
                    key=key,
                    seed=int(seed),
                    occurrence_number=occurrence_number,
                )
            )
            counters[base] = occurrence_number + 1

            if len(cipher_token) != len(base) + 1:
                raise RuntimeError("One-glyph suffix invariant failed")
            if tuple(cipher_token[:-1]) != base:
                raise RuntimeError("Encoded base changed under allocator")

            output.append(cipher_token)

    return output


def encoded_stream_hash(tokens: Sequence[GlyphToken]) -> str:
    payload = json.dumps(
        [list(token) for token in tokens],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def suffix_sequences_by_base(
    tokens: Sequence[GlyphToken],
) -> Dict[GlyphToken, List[str]]:
    groups: Dict[GlyphToken, List[str]] = defaultdict(list)
    for token in tokens:
        if len(token) < 2:
            raise RuntimeError("Cipher token too short to split base/suffix")
        base = tuple(token[:-1])
        suffix = str(token[-1])
        groups[base].append(suffix)
    return groups


def pooled_lag_rates(
    tokens: Sequence[GlyphToken],
    *,
    min_occurrences: int,
    lags: Sequence[int],
) -> tuple[dict, dict]:
    groups = suffix_sequences_by_base(tokens)
    eligible = {
        base: suffixes
        for base, suffixes in groups.items()
        if len(suffixes) >= int(min_occurrences)
    }

    rates: Dict[int, float] = {}
    pair_counts: Dict[int, int] = {}
    match_counts: Dict[int, int] = {}

    for lag in lags:
        lag = int(lag)
        pairs = 0
        matches = 0
        for suffixes in eligible.values():
            if len(suffixes) <= lag:
                continue
            for index in range(len(suffixes) - lag):
                pairs += 1
                if suffixes[index] == suffixes[index + lag]:
                    matches += 1

        if pairs <= 0:
            raise ValueError(
                f"No eligible same-family pairs at lag {lag}; "
                f"min_occurrences={min_occurrences}"
            )
        pair_counts[lag] = pairs
        match_counts[lag] = matches
        rates[lag] = matches / pairs

    diagnostics = {
        "eligible_family_count": len(eligible),
        "eligible_token_count": sum(len(values) for values in eligible.values()),
        "pair_counts": pair_counts,
        "match_counts": match_counts,
    }
    return rates, diagnostics


def optional_lag32_rate(tokens: Sequence[GlyphToken]) -> tuple[float | None, int, int]:
    groups = suffix_sequences_by_base(tokens)
    eligible = [
        suffixes
        for suffixes in groups.values()
        if len(suffixes) >= 33
    ]
    pairs = 0
    matches = 0
    for suffixes in eligible:
        for index in range(len(suffixes) - 32):
            pairs += 1
            if suffixes[index] == suffixes[index + 32]:
                matches += 1
    if pairs == 0:
        return None, len(eligible), 0
    return matches / pairs, len(eligible), pairs


def arithmetic_mean(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def score_model(
    *,
    model_name: str,
    allocator_module,
    train_lines: Sequence[PlainLine],
    test_lines: Sequence[PlainLine],
    key: Mapping[str, object],
    seed: int,
    config: Mapping[str, object],
) -> dict:
    counters: Counter = Counter()
    train_tokens = encode_partition(
        train_lines,
        key=key,
        seed=seed,
        allocator_module=allocator_module,
        counters=counters,
    )
    final_train_state = dict(counters)
    test_tokens = encode_partition(
        test_lines,
        key=key,
        seed=seed,
        allocator_module=allocator_module,
        counters=counters,
    )
    combined = [*train_tokens, *test_tokens]

    stat_cfg = config["primary_statistic"]
    min_occurrences = int(stat_cfg["eligible_family_min_occurrences"])
    lags = [int(x) for x in stat_cfg["primary_lags"]]
    rates, diagnostics = pooled_lag_rates(
        combined,
        min_occurrences=min_occurrences,
        lags=lags,
    )

    minimum_families = int(
        config["data_sufficiency"]["minimum_eligible_families_per_source_seed"]
    )
    minimum_lag16_pairs = int(
        config["data_sufficiency"]["minimum_lag16_pairs_per_source_seed"]
    )
    if diagnostics["eligible_family_count"] < minimum_families:
        raise ValueError(
            f"{model_name}: only {diagnostics['eligible_family_count']} eligible "
            f"families; require >= {minimum_families}"
        )
    if diagnostics["pair_counts"][16] < minimum_lag16_pairs:
        raise ValueError(
            f"{model_name}: only {diagnostics['pair_counts'][16]} lag-16 pairs; "
            f"require >= {minimum_lag16_pairs}"
        )

    short_mean = arithmetic_mean([rates[lag] for lag in range(1, 16)])
    contrast = rates[16] - short_mean
    lag32_rate, lag32_family_count, lag32_pair_count = optional_lag32_rate(combined)

    state_payload = json.dumps(
        [
            [list(base), int(count)]
            for base, count in sorted(
                final_train_state.items(),
                key=lambda item: json.dumps(
                    list(item[0]),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return {
        "model": model_name,
        "rates": rates,
        "pair_counts": diagnostics["pair_counts"],
        "match_counts": diagnostics["match_counts"],
        "eligible_family_count": diagnostics["eligible_family_count"],
        "eligible_token_count": diagnostics["eligible_token_count"],
        "lag1_15_mean_match_rate": short_mean,
        "lag16_match_rate": rates[16],
        "cycle16_contrast": contrast,
        "lag32_match_rate": lag32_rate,
        "lag32_eligible_family_count": lag32_family_count,
        "lag32_pair_count": lag32_pair_count,
        "encoded_token_count": len(combined),
        "encoded_stream_sha256": encoded_stream_hash(combined),
        "final_train_state_sha256": sha256(state_payload).hexdigest(),
    }


def candidate_exact_pattern(result: Mapping[str, object], tolerance: float) -> bool:
    rates = result["rates"]
    if abs(float(rates[16]) - 1.0) > tolerance:
        return False
    for lag in range(1, 16):
        if abs(float(rates[lag])) > tolerance:
            return False
    if abs(float(result["cycle16_contrast"]) - 1.0) > tolerance:
        return False
    return True


def prepare_result_dir(
    result_dir: Path,
    *,
    config_sha256: str,
    rerun_identical: bool,
) -> None:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        return

    if not rerun_identical:
        raise FileExistsError(
            f"Existing Phase 6A result found: {summary_path}\n"
            "Refusing to overwrite. Use --rerun-identical only for exact "
            "reproduction with the identical frozen config."
        )

    prior = read_json(summary_path)
    if prior.get("phase6a_config_sha256") != config_sha256:
        raise ValueError(
            "--rerun-identical refused: current config SHA differs from "
            "the existing result"
        )

    for name in (
        "per_lag_match_rates.csv",
        "per_run_cycle_statistics.csv",
        "language_replication.csv",
        "summary.json",
        "run_manifest.json",
        "SHA256SUMS",
    ):
        path = result_dir / name
        if path.exists():
            path.unlink()


def write_sha256sums(result_dir: Path) -> None:
    targets = sorted(
        path
        for path in result_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (result_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in targets
        )
        + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, *, rerun_identical: bool) -> int:
    config = load_yaml(config_path)
    verify_config_contract(config)
    config_sha = sha256_file(config_path)

    frozen_hashes = verify_frozen_modules(config)
    allocator_self_check()
    corpus = validate_manifest(config)

    result_dir = Path(str(config["outputs"]["result_dir"]))
    result_dir.mkdir(parents=True, exist_ok=True)
    prepare_result_dir(
        result_dir,
        config_sha256=config_sha,
        rerun_identical=rerun_identical,
    )

    tolerance = float(config["primary_statistic"]["numerical_tolerance"])
    seeds = [int(seed) for seed in config["seeds"]]

    per_lag_rows: List[dict] = []
    per_run_rows: List[dict] = []

    print("=" * 96)
    print("VOYAGER PHASE 6A — RECURRENCE-CYCLE FINGERPRINT CONTROL CALIBRATION")
    print("=" * 96)
    print(f"Experiment:          {config['experiment_id']}")
    print(f"Status:              {config['status'].upper()}")
    print(f"Config SHA-256:      {config_sha}")
    print(f"Control manifest:    {corpus['manifest_sha256']}")
    print("Voynich access:      PROHIBITED / NONE")
    print("Control freshness:   REUSED PHASE-5A DATA; NOT PROSPECTIVE")
    print("Primary statistic:   cycle16_contrast")
    print("Decision:            20/20 source-seed runs required")
    print()

    for source in corpus["sources"]:
        source_id = str(source["source_id"])
        language = str(source["language"])
        train_lines = source["_train_lines"]
        test_lines = source["_test_lines"]

        print(
            f"[{language}] {source_id}: "
            f"TRAIN={fixed_cycle.token_count(train_lines)} tokens, "
            f"TEST={fixed_cycle.token_count(test_lines)} tokens"
        )

        for seed in seeds:
            key = fixed_cycle.build_master_key(
                train_lines,
                seed=seed,
            )

            model_results = {}
            for model_name, module in MODEL_MODULES.items():
                model_results[model_name] = score_model(
                    model_name=model_name,
                    allocator_module=module,
                    train_lines=train_lines,
                    test_lines=test_lines,
                    key=key,
                    seed=seed,
                    config=config,
                )

                scored = model_results[model_name]
                for lag in range(1, 17):
                    per_lag_rows.append(
                        {
                            "source_id": source_id,
                            "language": language,
                            "seed": seed,
                            "model": model_name,
                            "lag": lag,
                            "match_rate": scored["rates"][lag],
                            "matches": scored["match_counts"][lag],
                            "pairs": scored["pair_counts"][lag],
                            "eligible_family_count": scored["eligible_family_count"],
                        }
                    )

            candidate = model_results["fixed_cycle_k16"]
            shuffled = model_results["cycle_shuffle_k16"]
            iid = model_results["iid_k16"]

            exact = candidate_exact_pattern(candidate, tolerance)
            best_null = max(
                float(shuffled["cycle16_contrast"]),
                float(iid["cycle16_contrast"]),
            )
            margin = float(candidate["cycle16_contrast"]) - best_null

            success = (
                exact
                and float(candidate["cycle16_contrast"])
                    > float(shuffled["cycle16_contrast"])
                and float(candidate["cycle16_contrast"])
                    > float(iid["cycle16_contrast"])
                and margin >= 0.50
            )

            row = {
                "source_id": source_id,
                "language": language,
                "seed": seed,
                "eligible_family_count": candidate["eligible_family_count"],
                "lag16_pair_count": candidate["pair_counts"][16],
                "candidate_lag1_15_mean": candidate["lag1_15_mean_match_rate"],
                "candidate_lag16": candidate["lag16_match_rate"],
                "candidate_cycle16_contrast": candidate["cycle16_contrast"],
                "cycle_shuffle_lag16": shuffled["lag16_match_rate"],
                "cycle_shuffle_cycle16_contrast": shuffled["cycle16_contrast"],
                "iid_lag16": iid["lag16_match_rate"],
                "iid_cycle16_contrast": iid["cycle16_contrast"],
                "candidate_minus_best_null": margin,
                "candidate_exact_pattern": exact,
                "success": success,
                "candidate_lag32": candidate["lag32_match_rate"],
                "cycle_shuffle_lag32": shuffled["lag32_match_rate"],
                "iid_lag32": iid["lag32_match_rate"],
                "candidate_encoded_stream_sha256": candidate["encoded_stream_sha256"],
                "cycle_shuffle_encoded_stream_sha256": shuffled["encoded_stream_sha256"],
                "iid_encoded_stream_sha256": iid["encoded_stream_sha256"],
                "final_train_state_sha256": candidate["final_train_state_sha256"],
            }
            per_run_rows.append(row)

            print(
                f"  seed={seed} "
                f"fixed={candidate['cycle16_contrast']:+.6f} "
                f"shuffle={shuffled['cycle16_contrast']:+.6f} "
                f"iid={iid['cycle16_contrast']:+.6f} "
                f"margin={margin:+.6f} "
                f"{'PASS' if success else 'FAIL'}"
            )

    language_rows = []
    for language in [str(x) for x in config["control_corpus"]["required_languages"]]:
        rows = [row for row in per_run_rows if row["language"] == language]
        successes = sum(bool(row["success"]) for row in rows)
        replicated = len(rows) == 5 and successes == 5
        language_rows.append(
            {
                "language": language,
                "successful_seeds": successes,
                "total_seeds": len(rows),
                "replicated_5_of_5": replicated,
            }
        )

    all_runs_success = len(per_run_rows) == 20 and all(
        bool(row["success"]) for row in per_run_rows
    )
    all_languages_replicated = len(language_rows) == 4 and all(
        bool(row["replicated_5_of_5"]) for row in language_rows
    )
    phase_pass = all_runs_success and all_languages_replicated

    decision = (
        config["primary_decision"]["pass_label"]
        if phase_pass
        else config["primary_decision"]["fail_label"]
    )

    write_csv(result_dir / "per_lag_match_rates.csv", per_lag_rows)
    write_csv(result_dir / "per_run_cycle_statistics.csv", per_run_rows)
    write_csv(result_dir / "language_replication.csv", language_rows)

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "decision": decision,
        "phase6a_config_sha256": config_sha,
        "control_manifest_sha256": corpus["manifest_sha256"],
        "control_data_status": config["control_corpus"]["reuse_status"],
        "voynich_accessed": False,
        "runs": len(per_run_rows),
        "successful_runs": sum(bool(row["success"]) for row in per_run_rows),
        "languages_replicated": sum(
            bool(row["replicated_5_of_5"]) for row in language_rows
        ),
        "required_success": "20/20 runs and 4/4 languages",
        "primary_statistic": "cycle16_contrast",
        "candidate_theoretical_contrast": 1.0,
        "minimum_candidate_minus_best_null": min(
            float(row["candidate_minus_best_null"])
            for row in per_run_rows
        ),
        "phase4_decision_retained": config["parent_context"]["phase4_decision_retained"],
        "phase5a_decision_retained": config["parent_context"]["phase5a_decision"],
        "interpretation_boundary": (
            "Control-side calibration only. A PASS shows that the frozen "
            "period-16 statistic separates the fixed-cycle K=16 allocator "
            "from the two prespecified matched allocation alternatives on "
            "the reused historical-language controls. It does not test Voynich."
        ),
    }
    write_json(result_dir / "summary.json", summary)

    manifest = {
        "schema_version": "1.0",
        "phase6a_config_sha256": config_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "control_manifest_path": str(corpus["manifest_path"]),
        "control_manifest_sha256": corpus["manifest_sha256"],
        **frozen_hashes,
        "voynich_accessed": False,
        "models": list(MODEL_MODULES),
        "seeds": seeds,
        "source_ids": [str(row["source_id"]) for row in corpus["sources"]],
        "output_files": [
            "per_lag_match_rates.csv",
            "per_run_cycle_statistics.csv",
            "language_replication.csv",
            "summary.json",
        ],
    }
    write_json(result_dir / "run_manifest.json", manifest)
    write_sha256sums(result_dir)

    print()
    print("=" * 96)
    print("VOYAGER PHASE 6A RESULT")
    print("=" * 96)
    print(f"Decision:             {decision}")
    print(
        f"Successful runs:      "
        f"{sum(bool(row['success']) for row in per_run_rows)}/20"
    )
    print(
        f"Languages replicated: "
        f"{sum(bool(row['replicated_5_of_5']) for row in language_rows)}/4"
    )
    print("Voynich files:        NOT ACCESSED")
    print(f"Saved:                {result_dir}")

    return 0 if phase_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--rerun-identical",
        action="store_true",
        help=(
            "Allow exact reproduction only when an existing summary records "
            "the identical current Phase-6A config SHA-256."
        ),
    )
    args = parser.parse_args()

    try:
        return run(
            args.config,
            rerun_identical=bool(args.rerun_identical),
        )
    except Exception as exc:
        print(
            f"PHASE 6A ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
