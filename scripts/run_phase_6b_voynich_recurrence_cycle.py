#!/usr/bin/env python3
"""VOYAGER Phase 6B — preregistered Voynich recurrence-cycle test.

Default invocation performs PRE-FLIGHT ONLY.

Execution:
    python scripts/run_phase_6b_voynich_recurrence_cycle.py --execute-analysis

Scientific status:
- the recurrence statistic/decision rule is preregistered before calculation;
- the Voynich corpus itself was already accessed in prior phases;
- this is not an untouched-manuscript test;
- Phase 4 and Phase 5A decisions remain permanently unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Dict, List, Mapping, MutableSequence, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import UNKNOWN_STA1, make_analytical_loci
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules


DEFAULT_CONFIG = Path("configs/phase_6b_voynich_recurrence_cycle.yaml")
GlyphToken = Tuple[str, ...]
BaseFamily = Tuple[str, ...]


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


def require_hash(path: Path, expected: str, label: str) -> str:
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


def verify_config_contract(config: Mapping[str, object]) -> None:
    if config.get("experiment_id") != "phase-6b-voynich-recurrence-cycle-v1":
        raise ValueError("Unexpected Phase 6B experiment_id")

    expected_status = (
        "preregistered_mechanism_derived_analysis_on_previously_accessed_voynich"
    )
    if config.get("status") != expected_status:
        raise ValueError("Unexpected Phase 6B status")

    stat = config["primary_statistic"]
    if stat["name"] != "cycle16_contrast":
        raise ValueError("Primary statistic changed")
    if int(stat["period"]) != 16:
        raise ValueError("Frozen period must remain 16")
    if int(stat["eligible_family_min_occurrences"]) != 17:
        raise ValueError("Eligible-family threshold must remain 17")
    if [int(x) for x in stat["primary_lags"]] != list(range(1, 17)):
        raise ValueError("Primary lags must remain exactly 1..16")

    suff = config["data_sufficiency"]
    if int(suff["minimum_eligible_families"]) != 10:
        raise ValueError("Minimum eligible families must remain 10")
    if int(suff["minimum_lag16_pairs"]) != 64:
        raise ValueError("Minimum lag-16 pairs must remain 64")

    null = config["within_family_permutation_null"]
    if int(null["replicates"]) != 10_000:
        raise ValueError("Permutation replicates must remain exactly 10000")
    if int(null["seed"]) != 40814041438:
        raise ValueError("Permutation seed must remain exactly 40814041438")
    if abs(float(null["alpha_each"]) - 0.001) > 1e-15:
        raise ValueError("Permutation alpha must remain exactly 0.001")

    source = config["voynich_input"]
    if source["corpus_scope"] != "WHOLE_ZL3B_CORPUS_IN_TRANSCRIPTION_ORDER":
        raise ValueError("Voynich corpus scope changed")
    if bool(source["split_files_used"]):
        raise ValueError("Phase 6B must not use frozen split files")

    observable = config["observable_mapping"]
    if observable["base_family"] != "tuple(token[:-1])":
        raise ValueError("Base-family mapping changed")
    if observable["suffix"] != "token[-1]":
        raise ValueError("Suffix mapping changed")


def verify_phase6a(config: Mapping[str, object]) -> dict:
    parent = config["parent_phase6a"]
    require_hash(
        Path(parent["config_path"]),
        parent["expected_config_sha256"],
        "Phase 6A config",
    )

    summary_path = Path(parent["summary_path"])
    if not summary_path.is_file():
        raise FileNotFoundError(f"Phase 6A summary missing: {summary_path}")
    summary = read_json(summary_path)

    if summary.get("decision") != parent["required_decision"]:
        raise ValueError(
            f"Phase 6A decision mismatch: {summary.get('decision')!r}"
        )
    if int(summary.get("successful_runs", -1)) != int(
        parent["required_successful_runs"]
    ):
        raise ValueError("Phase 6A successful-run count mismatch")
    if int(summary.get("languages_replicated", -1)) != int(
        parent["required_languages_replicated"]
    ):
        raise ValueError("Phase 6A language-replication count mismatch")
    if bool(summary.get("voynich_accessed")) != bool(
        parent["required_voynich_accessed"]
    ):
        raise ValueError("Phase 6A Voynich-access flag mismatch")
    if summary.get("phase6a_config_sha256") != parent["expected_config_sha256"]:
        raise ValueError("Phase 6A summary config SHA mismatch")

    return {
        "summary": summary,
        "summary_sha256": sha256_file(summary_path),
    }


def verify_inputs(config: Mapping[str, object]) -> dict:
    source = config["voynich_input"]
    source_path = Path(source["source_path"])
    rules_path = Path(source["sta1_rules_path"])

    source_sha = require_hash(
        source_path,
        source["expected_source_sha256"],
        "ZL3b source",
    )
    rules_sha = require_hash(
        rules_path,
        source["expected_sta1_rules_sha256"],
        "STA1 rules",
    )

    return {
        "source_path": source_path,
        "rules_path": rules_path,
        "source_sha256": source_sha,
        "rules_sha256": rules_sha,
    }


def prepare_execution(
    *,
    config: Mapping[str, object],
    config_sha: str,
    execute_analysis: bool,
    rerun_identical: bool,
) -> tuple[Path, bool]:
    execution = config["execution"]
    result_dir = Path(execution["result_dir"])
    marker_path = Path(execution["marker_path"])
    summary_path = result_dir / "summary.json"

    if not execute_analysis:
        return result_dir, False

    existing_sha = None
    if summary_path.is_file():
        prior = read_json(summary_path)
        existing_sha = prior.get("phase6b_config_sha256")
    elif marker_path.is_file():
        prior = read_json(marker_path)
        existing_sha = prior.get("phase6b_config_sha256")

    if summary_path.exists() or marker_path.exists():
        if not rerun_identical:
            raise FileExistsError(
                "Existing Phase 6B execution marker/result found. "
                "Refusing a second execution. Use --rerun-identical only for "
                "exact reproduction with the identical frozen config."
            )
        if existing_sha != config_sha:
            raise ValueError(
                "--rerun-identical refused: stored Phase 6B config SHA differs "
                "from the current config"
            )

        for name in (
            "observed_lag_profile.csv",
            "eligible_families.csv",
            "permutation_distribution.csv",
            "summary.json",
            "run_manifest.json",
            "SHA256SUMS",
        ):
            path = result_dir / name
            if path.exists():
                path.unlink()

    result_dir.mkdir(parents=True, exist_ok=True)

    if not marker_path.exists():
        write_json(
            marker_path,
            {
                "schema_version": "1.0",
                "experiment_id": config["experiment_id"],
                "phase6b_config_sha256": config_sha,
                "meaning": (
                    "Phase 6B recurrence-statistic execution has begun on the "
                    "previously accessed Voynich corpus."
                ),
                "voynich_data_status": config["voynich_input"]["data_status"],
            },
        )

    return result_dir, True


def extract_family_sequences(loci) -> tuple[Dict[BaseFamily, List[str]], dict]:
    families: Dict[BaseFamily, List[str]] = defaultdict(list)
    included = 0
    excluded_z1 = 0
    excluded_one_glyph = 0
    empty_tokens = 0

    for locus in loci:
        for token in locus.tokens:
            if not token:
                empty_tokens += 1
                continue

            glyphs = tuple(token)
            if UNKNOWN_STA1 in glyphs:
                excluded_z1 += 1
                continue
            if len(glyphs) < 2:
                excluded_one_glyph += 1
                continue

            base = tuple(glyphs[:-1])
            suffix = str(glyphs[-1])
            if not base:
                raise RuntimeError("Usable token unexpectedly produced empty base")
            families[base].append(suffix)
            included += 1

    return dict(families), {
        "included_tokens": included,
        "excluded_tokens_with_Z1": excluded_z1,
        "excluded_one_glyph_tokens": excluded_one_glyph,
        "empty_token_slots_skipped": empty_tokens,
        "observable_base_families": len(families),
    }


def eligible_families(
    families: Mapping[BaseFamily, Sequence[str]],
    *,
    minimum_occurrences: int,
) -> Dict[BaseFamily, Tuple[str, ...]]:
    return {
        base: tuple(suffixes)
        for base, suffixes in families.items()
        if len(suffixes) >= minimum_occurrences
    }


def lag_rate(
    families: Mapping[BaseFamily, Sequence[str]],
    lag: int,
) -> tuple[float, int, int]:
    pairs = 0
    matches = 0

    for suffixes in families.values():
        n = len(suffixes)
        if n <= lag:
            continue
        pairs += n - lag
        matches += sum(
            1
            for index in range(n - lag)
            if suffixes[index] == suffixes[index + lag]
        )

    if pairs <= 0:
        raise ValueError(f"No pooled pairs available at lag {lag}")
    return matches / pairs, matches, pairs


def observed_profile(
    families: Mapping[BaseFamily, Sequence[str]],
    lags: Sequence[int],
) -> tuple[dict, dict, dict]:
    rates = {}
    matches = {}
    pairs = {}
    for lag in lags:
        rate, matched, total = lag_rate(families, int(lag))
        rates[int(lag)] = rate
        matches[int(lag)] = matched
        pairs[int(lag)] = total
    return rates, matches, pairs


def cycle16_contrast(rates: Mapping[int, float]) -> tuple[float, float]:
    short_mean = sum(float(rates[k]) for k in range(1, 16)) / 15.0
    return float(rates[16]) - short_mean, short_mean


def shuffled_copy(
    families: Mapping[BaseFamily, Sequence[str]],
    rng: random.Random,
) -> Dict[BaseFamily, Tuple[str, ...]]:
    output = {}
    # Sorting makes RNG consumption deterministic independent of dict ordering.
    ordered = sorted(
        families.items(),
        key=lambda item: json.dumps(
            list(item[0]),
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )
    for base, suffixes in ordered:
        values = list(suffixes)
        rng.shuffle(values)
        output[base] = tuple(values)
    return output


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("Cannot compute quantile of empty sequence")
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def monte_carlo_p(null_values: Sequence[float], observed: float) -> float:
    exceed = sum(float(value) >= float(observed) for value in null_values)
    return (1 + exceed) / (len(null_values) + 1)


def run_permutations(
    families: Mapping[BaseFamily, Sequence[str]],
    *,
    replicates: int,
    seed: int,
    compute_lag32: bool,
) -> tuple[List[dict], list, list, list]:
    rows: List[dict] = []
    null_lag16: List[float] = []
    null_contrast: List[float] = []
    null_lag32: List[float] = []

    rng = random.Random(int(seed))

    for replicate in range(1, int(replicates) + 1):
        permuted = shuffled_copy(families, rng)
        rates, _, _ = observed_profile(permuted, range(1, 17))
        contrast, short_mean = cycle16_contrast(rates)

        lag32_value = None
        if compute_lag32:
            lag32_value, _, _ = lag_rate(permuted, 32)
            null_lag32.append(lag32_value)

        null_lag16.append(rates[16])
        null_contrast.append(contrast)

        rows.append(
            {
                "replicate": replicate,
                "lag16_match_rate": rates[16],
                "short_lag_mean": short_mean,
                "cycle16_contrast": contrast,
                "lag32_match_rate": lag32_value,
            }
        )

    return rows, null_lag16, null_contrast, null_lag32


def family_inventory_rows(
    families: Mapping[BaseFamily, Sequence[str]],
) -> List[dict]:
    rows = []
    for base, suffixes in sorted(
        families.items(),
        key=lambda item: (
            -len(item[1]),
            json.dumps(list(item[0]), separators=(",", ":"), ensure_ascii=False),
        ),
    ):
        rate16, matches16, pairs16 = lag_rate({base: suffixes}, 16)
        rows.append(
            {
                "base_family_json": json.dumps(
                    list(base),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                "occurrences": len(suffixes),
                "distinct_suffixes": len(set(suffixes)),
                "lag16_pairs": pairs16,
                "lag16_matches": matches16,
                "lag16_match_rate": rate16,
            }
        )
    return rows


def write_sha256sums(result_dir: Path) -> None:
    targets = sorted(
        path
        for path in result_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (result_dir / "SHA256SUMS").write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in targets) + "\n",
        encoding="utf-8",
    )


def run(
    config_path: Path,
    *,
    execute_analysis: bool,
    rerun_identical: bool,
) -> int:
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config_sha = sha256_file(config_path)
    config = load_yaml(config_path)
    verify_config_contract(config)

    phase6a = verify_phase6a(config)
    inputs = verify_inputs(config)

    result_dir, executing = prepare_execution(
        config=config,
        config_sha=config_sha,
        execute_analysis=execute_analysis,
        rerun_identical=rerun_identical,
    )

    print("=" * 96)
    print("VOYAGER PHASE 6B — PREREGISTERED VOYNICH RECURRENCE-CYCLE TEST")
    print("=" * 96)
    print(f"Experiment:          {config['experiment_id']}")
    print(f"Status:              {config['status'].upper()}")
    print(f"Config SHA-256:      {config_sha}")
    print(f"Phase 6A:            {phase6a['summary']['decision']}")
    print(f"ZL3b SHA-256:        {inputs['source_sha256']}")
    print(f"STA1 SHA-256:        {inputs['rules_sha256']}")
    print("Corpus scope:        WHOLE ZL3b / previously accessed")
    print("Base family:         token[:-1]")
    print("Suffix:              token[-1]")
    print("Period:              16")
    print("Eligible family:     >=17 occurrences")
    print("Permutation null:    10,000 within-family suffix shuffles")
    print("Primary alpha:       p_lag16 <= .001 AND p_contrast <= .001")
    print()

    if not executing:
        print("PRE-FLIGHT PASS — recurrence statistic NOT calculated.")
        print()
        print("Freeze/commit Phase 6B, then execute exactly:")
        print(
            "  python scripts/run_phase_6b_voynich_recurrence_cycle.py "
            "--execute-analysis"
        )
        return 0

    records = load_ivtff(
        inputs["source_path"],
        transcription_id=config["voynich_input"]["transcription_id"],
        strict=True,
    )
    rules = load_bitrans_rules(
        inputs["rules_path"],
        expected_native_alphabet="Eva-",
    )
    if rules.sha256 != inputs["rules_sha256"]:
        raise ValueError("Loaded STA1 rule hash differs from verified hash")

    loci = make_analytical_loci(records, rules)
    all_families, extraction = extract_family_sequences(loci)

    minimum_occurrences = int(
        config["primary_statistic"]["eligible_family_min_occurrences"]
    )
    eligible = eligible_families(
        all_families,
        minimum_occurrences=minimum_occurrences,
    )

    observed_rates, observed_matches, observed_pairs = observed_profile(
        eligible,
        range(1, 17),
    )
    observed_contrast, observed_short_mean = cycle16_contrast(observed_rates)

    minimum_families = int(config["data_sufficiency"]["minimum_eligible_families"])
    minimum_lag16_pairs = int(config["data_sufficiency"]["minimum_lag16_pairs"])

    sufficiency_pass = (
        len(eligible) >= minimum_families
        and observed_pairs[16] >= minimum_lag16_pairs
    )

    lag32_families = {
        base: suffixes
        for base, suffixes in all_families.items()
        if len(suffixes)
        >= int(config["secondary_descriptive"]["lag32"]["family_min_occurrences"])
    }
    observed_lag32 = None
    observed_lag32_pairs = 0
    if lag32_families:
        observed_lag32, _, observed_lag32_pairs = lag_rate(lag32_families, 32)

    # Primary null always uses the primary eligible-family set. The secondary
    # lag-32 permutation uses its own >=33 family set below.
    null_cfg = config["within_family_permutation_null"]
    permutation_rows, null_lag16, null_contrast, _ = run_permutations(
        eligible,
        replicates=int(null_cfg["replicates"]),
        seed=int(null_cfg["seed"]),
        compute_lag32=False,
    )

    null_lag32 = []
    if lag32_families:
        lag32_rng = random.Random(int(null_cfg["seed"]) + 32)
        for _ in range(int(null_cfg["replicates"])):
            permuted32 = shuffled_copy(lag32_families, lag32_rng)
            rate32, _, _ = lag_rate(permuted32, 32)
            null_lag32.append(rate32)

        for row, value in zip(permutation_rows, null_lag32):
            row["lag32_match_rate"] = value

    p_lag16 = monte_carlo_p(null_lag16, observed_rates[16])
    p_contrast = monte_carlo_p(null_contrast, observed_contrast)
    p_lag32 = (
        monte_carlo_p(null_lag32, observed_lag32)
        if observed_lag32 is not None
        else None
    )

    strict_lag16_peak = observed_rates[16] > max(
        observed_rates[k] for k in range(1, 16)
    )

    alpha = float(null_cfg["alpha_each"])
    detected = (
        sufficiency_pass
        and observed_contrast > 0.0
        and strict_lag16_peak
        and p_lag16 <= alpha
        and p_contrast <= alpha
    )

    decision_cfg = config["primary_decision"]
    decision = (
        decision_cfg["detected_label"]
        if detected
        else decision_cfg["not_detected_label"]
    )

    lag_rows = []
    for lag in range(1, 17):
        lag_rows.append(
            {
                "lag": lag,
                "match_rate": observed_rates[lag],
                "matches": observed_matches[lag],
                "pairs": observed_pairs[lag],
                "primary_period": lag == 16,
            }
        )
    if observed_lag32 is not None:
        lag_rows.append(
            {
                "lag": 32,
                "match_rate": observed_lag32,
                "matches": None,
                "pairs": observed_lag32_pairs,
                "primary_period": False,
            }
        )

    family_rows = family_inventory_rows(eligible)
    families_over_16_suffixes = sum(
        int(row["distinct_suffixes"]) > 16
        for row in family_rows
    )

    write_csv(result_dir / "observed_lag_profile.csv", lag_rows)
    write_csv(result_dir / "eligible_families.csv", family_rows)
    write_csv(result_dir / "permutation_distribution.csv", permutation_rows)

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "decision": decision,
        "phase6b_config_sha256": config_sha,
        "voynich_data_status": config["voynich_input"]["data_status"],
        "corpus_scope": config["voynich_input"]["corpus_scope"],
        "phase4_decision_retained": config["retained_prior_results"]["phase4_decision"],
        "phase5a_decision_retained": config["retained_prior_results"]["phase5a_decision"],
        "phase6a_decision": phase6a["summary"]["decision"],
        "extraction": extraction,
        "eligible_family_count": len(eligible),
        "lag16_pair_count": observed_pairs[16],
        "data_sufficiency_pass": sufficiency_pass,
        "observed_short_lag_mean": observed_short_mean,
        "observed_lag16_match_rate": observed_rates[16],
        "observed_cycle16_contrast": observed_contrast,
        "strict_lag16_peak": strict_lag16_peak,
        "permutation_replicates": int(null_cfg["replicates"]),
        "permutation_seed": int(null_cfg["seed"]),
        "p_lag16": p_lag16,
        "p_cycle16_contrast": p_contrast,
        "null_lag16_mean": statistics.fmean(null_lag16),
        "null_lag16_sd_population": statistics.pstdev(null_lag16),
        "null_lag16_q999": quantile(null_lag16, 0.999),
        "null_contrast_mean": statistics.fmean(null_contrast),
        "null_contrast_sd_population": statistics.pstdev(null_contrast),
        "null_contrast_q999": quantile(null_contrast, 0.999),
        "secondary_lag32_match_rate": observed_lag32,
        "secondary_lag32_pairs": observed_lag32_pairs,
        "secondary_lag32_permutation_p": p_lag32,
        "eligible_families_with_more_than_16_distinct_suffixes": (
            families_over_16_suffixes
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(result_dir / "summary.json", summary)

    run_manifest = {
        "schema_version": "1.0",
        "phase6b_config_sha256": config_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "phase6a_summary_sha256": phase6a["summary_sha256"],
        "zl3b_source_sha256": inputs["source_sha256"],
        "sta1_rules_sha256": inputs["rules_sha256"],
        "representation": config["voynich_input"]["representation"],
        "split_files_used": False,
        "periods_tested_for_primary": [16],
        "base_family_rule": config["observable_mapping"]["base_family"],
        "suffix_rule": config["observable_mapping"]["suffix"],
        "permutation_replicates": int(null_cfg["replicates"]),
        "permutation_seed": int(null_cfg["seed"]),
        "outputs": [
            "ANALYSIS_EXECUTED.json",
            "observed_lag_profile.csv",
            "eligible_families.csv",
            "permutation_distribution.csv",
            "summary.json",
        ],
    }
    write_json(result_dir / "run_manifest.json", run_manifest)
    write_sha256sums(result_dir)

    print("=" * 96)
    print("VOYAGER PHASE 6B RESULT")
    print("=" * 96)
    print(f"Decision:             {decision}")
    print(f"Eligible families:    {len(eligible)}")
    print(f"Lag-16 pairs:         {observed_pairs[16]}")
    print(f"Lag-16 match rate:    {observed_rates[16]:+.8f}")
    print(f"Short-lag mean:       {observed_short_mean:+.8f}")
    print(f"Cycle16 contrast:     {observed_contrast:+.8f}")
    print(f"Strict lag-16 peak:   {'YES' if strict_lag16_peak else 'NO'}")
    print(f"Permutation p lag16: {p_lag16:.8g}")
    print(f"Permutation p cycle: {p_contrast:.8g}")
    if observed_lag32 is not None:
        print(f"Secondary lag-32:     {observed_lag32:+.8f}")
        print(f"Secondary p lag-32:   {p_lag32:.8g}")
    print("Phase 4 decision:     RETAINED UNCHANGED")
    print("Phase 5A decision:    RETAINED UNCHANGED")
    print(f"Saved:                {result_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--execute-analysis",
        action="store_true",
        help="Calculate the frozen Phase 6B recurrence outcome.",
    )
    parser.add_argument(
        "--rerun-identical",
        action="store_true",
        help=(
            "Permit exact reproducibility rerun only when stored and current "
            "Phase-6B config SHA-256 values are identical."
        ),
    )
    args = parser.parse_args()

    if args.rerun_identical and not args.execute_analysis:
        print(
            "PHASE 6B ERROR: --rerun-identical requires --execute-analysis",
            file=sys.stderr,
        )
        return 1

    try:
        return run(
            args.config,
            execute_analysis=bool(args.execute_analysis),
            rerun_identical=bool(args.rerun_identical),
        )
    except Exception as exc:
        print(
            f"PHASE 6B ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
