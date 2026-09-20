#!/usr/bin/env python3
"""VOYAGER Phase 5A — prospective boundary-structure prediction.

Default mode is PRE-FLIGHT ONLY. Fresh locked TEST files are not touched until
--execute-prospective-test is supplied after the protocol, fresh-control
manifest, TRAIN files, and TEST files have been frozen.

Scientific design:
* exact frozen Phase-3A.6 recurrence-aware K=16 generator;
* fresh historical-language TRAIN -> locked TEST continuation;
* fixed HMM K=16 in space-free and token-aware views;
* fixed matched Witten-Bell order-3 comparator;
* exactly 15 deterministic contiguous token-balanced TEST units;
* primary feature H(final glyph, initial glyph) across token boundaries;
* primary outcome token-aware delta minus space-free delta;
* per-run success requires Pearson > 0 AND Spearman > 0;
* language replication requires >=4/5 seeds;
* Phase 5A support requires >=3/4 languages.

This script never reads any Voynich corpus or Voynich split file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Phase 5A config is YAML. Install PyYAML in the project environment."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ciphers import recurrence_avoiding_surface_allocation as mechanism
from src.models.hmm import CategoricalHMM, evaluate_hmm
from src.models.ngram import (
    BaselineSequence,
    WB,
    WittenBellNGram,
    evaluate_model,
)

DEFAULT_CONFIG = Path("configs/phase_5a_boundary_prediction.yaml")
GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]
CipherLine = Tuple[GlyphToken, ...]

FORBIDDEN_VOYNICH_PATH_FRAGMENTS = (
    "data/raw/zl",
    "data/splits/v1/train.txt",
    "data/splits/v1/validation.txt",
    "data/splits/v1/test_locked.txt",
)

REQUIRED_MANIFEST_SOURCE_FIELDS = {
    "source_id",
    "language",
    "corpus",
    "version",
    "license",
    "provenance",
    "train_path",
    "train_sha256",
    "train_lines",
    "train_tokens",
    "test_locked_path",
    "test_locked_sha256",
    "test_locked_lines",
    "test_locked_tokens",
    "prospective_status",
    "selection_frozen_before_outcomes",
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


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
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA mismatch\n"
            f"  expected: {expected}\n"
            f"  observed: {observed}"
        )
    return observed


def valid_sha256_text(value: object) -> bool:
    text = str(value)
    return (
        len(text) == 64
        and all(char in "0123456789abcdef" for char in text.lower())
    )


def lexical_path_is_forbidden(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(fragment in normalized for fragment in FORBIDDEN_VOYNICH_PATH_FRAGMENTS)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_variance(values: Sequence[float]) -> float | None:
    if not values:
        return None
    m = mean(values)
    assert m is not None
    return sum((value - m) ** 2 for value in values) / len(values)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = count / total
        result -= p * math.log2(p)
    return result


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def correlation(rows: Sequence[Mapping[str, object]], feature: str, outcome: str) -> tuple[float | None, float | None]:
    x: List[float] = []
    y: List[float] = []
    for row in rows:
        a = float(row[feature])
        b = float(row[outcome])
        if math.isfinite(a) and math.isfinite(b):
            x.append(a)
            y.append(b)
    return pearson(x, y), spearman(x, y)


def verify_d3(config: Mapping[str, object]) -> dict:
    parent = config["frozen_parent_d3"]
    config_path = Path(parent["config_path"])
    d3_config_hash = require_hash(
        config_path,
        parent["expected_config_sha256"],
        "Phase 4.D3 config",
    )

    summary_path = Path(parent["summary_path"])
    manifest_path = Path(parent["run_manifest_path"])
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Completed Phase 4.D3 outputs are required")

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)

    if summary.get("status") != parent["required_status"]:
        raise ValueError("Unexpected D3 status")
    if (
        summary.get("parent_phase4_decision_retained")
        != parent["required_phase4_decision_retained"]
    ):
        raise ValueError("D3 does not retain the required Phase-4 decision")
    if bool(summary.get("predictive_models_fit_or_rescored")):
        raise ValueError("D3 summary says predictive models were fit/rescored")
    if (
        bool(summary.get("d2_boundary_pair_reconstruction_parity_all_passed"))
        != bool(parent["require_d2_boundary_reconstruction_parity"])
    ):
        raise ValueError("D3 did not pass the required D2 reconstruction parity")
    # D2 and D3 use different field names for the same frozen empirical
    # quantity. D2 called it ``across_transition_pair_entropy_bits``. D3
    # reconstructs it as ``boundary_pair_entropy_bits = H(F,I)`` and requires
    # exact count/entropy parity with D2 before interpreting the anatomy.
    if (
        summary.get("preidentified_d2_feature")
        != parent["preidentified_d2_feature"]
    ):
        raise ValueError(
            "D3 preidentified D2 feature differs from the frozen D2 lineage"
        )

    d3_config = read_json(config_path)
    association = d3_config.get("association_analysis", {})
    d3_feature = parent["d3_reconstructed_equivalent_feature"]

    if d3_feature not in association.get("features", []):
        raise ValueError(
            "D3 reconstructed H(F,I) feature is absent from the frozen "
            "association-analysis feature set"
        )
    if (
        association.get("outcome")
        != parent["required_d3_association_outcome"]
    ):
        raise ValueError("D3 association outcome differs from the Phase 5A freeze")

    formulae = d3_config.get("formulae", {})
    if formulae.get(d3_feature) != parent["required_d3_formula"]:
        raise ValueError(
            "D3 boundary-pair entropy formula differs from the Phase 5A freeze"
        )

    if not bool(summary.get("d2_boundary_pair_reconstruction_parity_all_passed")):
        raise ValueError(
            "D3 did not establish D2-to-D3 H(F,I) reconstruction parity"
        )
    if not bool(manifest.get("d2_pair_entropy_parity_all_passed")):
        raise ValueError(
            "D3 run manifest does not record D2 H(F,I) parity as passed"
        )

    if bool(manifest.get("predictive_models_fit_or_rescored_by_d3")):
        raise ValueError("D3 manifest says predictive models were fit/rescored")

    return {
        "d3_config_sha256": d3_config_hash,
        "d3_summary_sha256": sha256_file(summary_path),
        "d3_run_manifest_sha256": sha256_file(manifest_path),
    }


def verify_model_configs(config: Mapping[str, object]) -> dict:
    mechanism_cfg = config["frozen_mechanism"]
    hmm_cfg = config["frozen_hmm"]
    ngram_cfg = config["frozen_ngram"]

    observed = {
        "mechanism_config_sha256": require_hash(
            Path(mechanism_cfg["config_path"]),
            mechanism_cfg["expected_config_sha256"],
            "Phase 3A.6 mechanism config",
        ),
        "mechanism_generator_sha256": require_hash(
            Path(mechanism_cfg["generator_path"]),
            mechanism_cfg["expected_generator_sha256"],
            "Phase 3A.6 generator",
        ),
        "hmm_config_sha256": require_hash(
            Path(hmm_cfg["config_path"]),
            hmm_cfg["expected_config_sha256"],
            "HMM config",
        ),
        "ngram_config_sha256": require_hash(
            Path(ngram_cfg["config_path"]),
            ngram_cfg["expected_config_sha256"],
            "n-gram config",
        ),
    }

    if sha256_file(Path(mechanism.__file__).resolve()) != mechanism_cfg["expected_generator_sha256"]:
        raise ValueError("Imported recurrence-aware generator differs from frozen SHA")

    if int(mechanism_cfg["variant_count"]) != 16:
        raise ValueError("Phase 5A mechanism K must remain 16")
    if int(mechanism_cfg["suffix_length_glyphs"]) != 1:
        raise ValueError("Phase 5A suffix length must remain one glyph")

    hmm = read_json(Path(hmm_cfg["config_path"]))
    required_hmm = {
        "hidden_states": [2, 4, 8, 16],
        "views": ["space_free", "token_aware"],
        "restarts": 5,
        "restart_selection": "highest_training_log_likelihood",
        "pseudocount": 0.5,
        "seed": 40814041438,
    }
    for key, expected in required_hmm.items():
        if hmm.get(key) != expected:
            raise ValueError(
                f"Frozen HMM config mismatch at {key}: "
                f"{hmm.get(key)!r} != {expected!r}"
            )

    if int(hmm_cfg["hidden_states"]["space_free"]) != 16:
        raise ValueError("space_free HMM must remain K=16")
    if int(hmm_cfg["hidden_states"]["token_aware"]) != 16:
        raise ValueError("token_aware HMM must remain K=16")

    ngram = read_json(Path(ngram_cfg["config_path"]))
    if ngram.get("orders") != [1, 2, 3, 4, 5]:
        raise ValueError("Frozen n-gram order grid changed")
    if ngram.get("views") != ["space_free", "token_aware"]:
        raise ValueError("Frozen n-gram views changed")
    if int(ngram_cfg["matched_order"]) != 3:
        raise ValueError("Matched comparator must remain order 3")

    observed["hmm_config"] = hmm
    observed["ngram_config"] = ngram
    return observed


def validate_fresh_manifest_without_test_access(
    config: Mapping[str, object],
) -> dict:
    """Validate manifest and TRAIN only. Never touch a TEST path here."""
    fresh = config["fresh_controls"]
    manifest_path = Path(fresh["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Fresh-control manifest missing: {manifest_path}\n"
            "Install and freeze the Phase 5A fresh controls before execution."
        )

    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unexpected fresh-control manifest schema")
    if manifest.get("freeze_id") != fresh["required_freeze_id"]:
        raise ValueError("Unexpected fresh-control freeze_id")
    if (
        bool(manifest.get("selection_frozen_before_outcomes"))
        != bool(fresh["require_selection_frozen_before_outcomes"])
    ):
        raise ValueError("Fresh source selection was not frozen before outcomes")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("Fresh-control manifest must contain exactly four sources")

    required_languages = list(fresh["required_languages"])
    observed_languages = [str(source.get("language")) for source in sources]
    if sorted(observed_languages) != sorted(required_languages):
        raise ValueError(
            f"Fresh languages must be exactly {required_languages}; "
            f"observed {observed_languages}"
        )
    if len(set(observed_languages)) != 4:
        raise ValueError("Exactly one fresh source per language is required")

    source_ids = [str(source.get("source_id")) for source in sources]
    if len(set(source_ids)) != 4:
        raise ValueError("Fresh source IDs must be distinct")
    forbidden = set(fresh["forbidden_prior_source_ids"])
    overlap = forbidden & set(source_ids)
    if overlap:
        raise ValueError(
            "Previously used Phase-3 source IDs are forbidden in Phase 5A: "
            + ", ".join(sorted(overlap))
        )

    checked_sources = []
    for source in sources:
        missing = REQUIRED_MANIFEST_SOURCE_FIELDS - set(source)
        if missing:
            raise ValueError(
                f"{source.get('source_id', '<unknown>')}: manifest fields missing: "
                f"{sorted(missing)}"
            )

        source_id = str(source["source_id"])
        train_path_string = str(source["train_path"])
        test_path_string = str(source["test_locked_path"])

        if lexical_path_is_forbidden(train_path_string) or lexical_path_is_forbidden(test_path_string):
            raise ValueError(f"{source_id}: Voynich paths are prohibited in Phase 5A")

        # Purely lexical TEST checks. Do NOT construct/use filesystem metadata
        # operations on the TEST path in preflight.
        if not test_path_string.replace("\\", "/").endswith(
            str(fresh["require_test_filename_suffix"])
        ):
            raise ValueError(
                f"{source_id}: locked test filename must end with "
                f"{fresh['require_test_filename_suffix']!r}"
            )
        if not valid_sha256_text(source["test_locked_sha256"]):
            raise ValueError(f"{source_id}: invalid frozen TEST SHA text")
        if int(source["test_locked_lines"]) <= 0 or int(source["test_locked_tokens"]) <= 0:
            raise ValueError(f"{source_id}: invalid frozen TEST counts in manifest")
        if source["prospective_status"] != fresh["required_prospective_status"]:
            raise ValueError(f"{source_id}: prospective status is not untouched")
        if not bool(source["selection_frozen_before_outcomes"]):
            raise ValueError(f"{source_id}: source selection was not frozen")

        train_path = Path(train_path_string)
        if not train_path.is_file():
            raise FileNotFoundError(f"{source_id}: TRAIN missing: {train_path}")
        observed_train_sha = sha256_file(train_path)
        if observed_train_sha != source["train_sha256"]:
            raise ValueError(f"{source_id}: TRAIN SHA mismatch")

        train_lines = mechanism.read_source_lines(train_path)
        if len(train_lines) != int(source["train_lines"]):
            raise ValueError(f"{source_id}: TRAIN line-count mismatch")
        if mechanism.token_count(train_lines) != int(source["train_tokens"]):
            raise ValueError(f"{source_id}: TRAIN token-count mismatch")
        if len(train_lines) < 15:
            raise ValueError(f"{source_id}: TRAIN unexpectedly small")

        checked_sources.append(
            {
                **dict(source),
                "_train_lines": train_lines,
                "_observed_train_sha256": observed_train_sha,
            }
        )

    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "sources": checked_sources,
    }


def preflight(config_path: Path, config: Mapping[str, object]) -> dict:
    if config.get("experiment_id") != "phase-5a-prospective-boundary-prediction-v1":
        raise ValueError("Unexpected Phase 5A experiment_id")
    if config.get("status") != "confirmatory_prospective_control_prediction":
        raise ValueError("Phase 5A must remain confirmatory_prospective_control_prediction")

    seeds = [int(seed) for seed in config["seeds"]]
    if seeds != list(range(40814041438, 40814041443)):
        raise ValueError("Phase 5A seeds must remain exactly 40814041438..40814041442")

    primary = config["primary_prediction"]
    if primary["feature"] != "boundary_pair_entropy_bits":
        raise ValueError("Phase 5A primary feature cannot change")
    if primary["outcome"] != "boundary_boost_bits_per_event":
        raise ValueError("Phase 5A primary outcome cannot change")
    if primary["per_run_direction_match"] != "pearson_r > 0 and spearman_rho > 0":
        raise ValueError("Phase 5A per-run decision rule changed")

    lineage = verify_d3(config)
    frozen = verify_model_configs(config)
    fresh = validate_fresh_manifest_without_test_access(config)

    expected_runs = int(config["expected_grid"]["runs"])
    if expected_runs != 4 * 5:
        raise ValueError("Phase 5A run grid must remain 4 x 5 = 20")
    if int(config["test_unitization"]["unit_count_per_source_seed"]) != 15:
        raise ValueError("Phase 5A TEST unit count must remain 15")

    return {
        "phase5a_config_sha256": sha256_file(config_path),
        **lineage,
        **frozen,
        **fresh,
    }


def encode_lines(
    lines: Sequence[PlainLine],
    *,
    key: Mapping[str, object],
    seed: int,
    counters: MutableMapping[GlyphToken, int],
) -> List[Tuple[int, CipherLine]]:
    output: List[Tuple[int, CipherLine]] = []
    for line_index, plain_line in enumerate(lines, start=1):
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
                raise RuntimeError("One-glyph suffix invariant failed")
            cipher_line.append(cipher_token)

        if not cipher_line:
            raise ValueError(f"Empty source line encountered at line {line_index}")
        output.append((line_index, tuple(cipher_line)))
    return output


def make_test_units(
    test_units: Sequence[Tuple[int, CipherLine]],
    *,
    unit_count: int = 15,
) -> tuple[Dict[int, str], List[dict]]:
    """Deterministic contiguous token-balanced grouping."""
    if len(test_units) < unit_count:
        raise ValueError(
            f"Need at least {unit_count} nonempty TEST lines; found {len(test_units)}"
        )
    total_tokens = sum(len(tokens) for _, tokens in test_units)
    if total_tokens <= 0:
        raise ValueError("TEST has no tokens")

    assignment: Dict[int, str] = {}
    summaries: List[dict] = []
    current_lines: List[Tuple[int, CipherLine]] = []
    cumulative_tokens = 0
    closed_units = 0

    for position, item in enumerate(test_units):
        line_index, tokens = item
        current_lines.append(item)
        cumulative_tokens += len(tokens)

        if closed_units >= unit_count - 1:
            continue

        next_unit_number = closed_units + 1
        target = total_tokens * next_unit_number / unit_count
        remaining_lines = len(test_units) - position - 1
        remaining_units_after_close = unit_count - next_unit_number

        close = (
            cumulative_tokens >= target
            or remaining_lines == remaining_units_after_close
        )
        if close:
            unit_id = f"u{next_unit_number:02d}"
            for idx, _ in current_lines:
                assignment[int(idx)] = unit_id
            summaries.append(
                {
                    "unit_id": unit_id,
                    "first_test_line": current_lines[0][0],
                    "last_test_line": current_lines[-1][0],
                    "line_count": len(current_lines),
                    "token_count": sum(len(line) for _, line in current_lines),
                }
            )
            current_lines = []
            closed_units += 1

    if current_lines:
        unit_id = f"u{unit_count:02d}"
        for idx, _ in current_lines:
            assignment[int(idx)] = unit_id
        summaries.append(
            {
                "unit_id": unit_id,
                "first_test_line": current_lines[0][0],
                "last_test_line": current_lines[-1][0],
                "line_count": len(current_lines),
                "token_count": sum(len(line) for _, line in current_lines),
            }
        )

    if len(summaries) != unit_count:
        raise RuntimeError(
            f"Unitization produced {len(summaries)} units, expected {unit_count}"
        )
    if len(assignment) != len(test_units):
        raise RuntimeError("Not every TEST line was assigned exactly once")
    if any(int(row["line_count"]) <= 0 for row in summaries):
        raise RuntimeError("Unitization produced an empty unit")

    return assignment, summaries


def baseline_examples(
    units: Sequence[Tuple[int, CipherLine]],
    *,
    view: str,
    group_by_line: Mapping[int, str] | None = None,
    group_prefix: str = "train",
) -> List[BaselineSequence]:
    if view not in ("space_free", "token_aware"):
        raise ValueError(f"Unknown view: {view}")

    output: List[BaselineSequence] = []
    for line_index, tokens in units:
        if view == "space_free":
            symbols = tuple(glyph for token in tokens for glyph in token)
            boundaries = (False,) * len(symbols)
        else:
            symbols_list: List[str] = []
            boundary_list: List[bool] = []
            for token_index, token in enumerate(tokens):
                if token_index > 0:
                    symbols_list.append(WB)
                    boundary_list.append(True)
                symbols_list.extend(token)
                boundary_list.extend([False] * len(token))
            symbols = tuple(symbols_list)
            boundaries = tuple(boundary_list)

        if not symbols:
            continue
        leaf_group = (
            group_by_line[int(line_index)]
            if group_by_line is not None
            else group_prefix
        )
        output.append(
            BaselineSequence(
                leaf_group=leaf_group,
                folio=leaf_group,
                locus=f"{leaf_group}.line{int(line_index)}",
                symbols=symbols,
                is_boundary=boundaries,
            )
        )
    return output


def fit_score_view(
    train_units: Sequence[Tuple[int, CipherLine]],
    test_units: Sequence[Tuple[int, CipherLine]],
    *,
    assignment: Mapping[int, str],
    view: str,
    hmm_config: Mapping[str, object],
    ngram_config: Mapping[str, object],
) -> dict:
    train_examples = baseline_examples(
        train_units,
        view=view,
        group_prefix="train",
    )
    test_examples = baseline_examples(
        test_units,
        view=view,
        group_by_line=assignment,
    )
    train_sequences = [
        example.symbols
        for example in train_examples
        if example.symbols
    ]

    view_index = list(hmm_config["views"]).index(view)
    hidden_states = 16
    fitted = []

    for restart in range(int(hmm_config["restarts"])):
        model_seed = (
            int(hmm_config["seed"])
            + view_index * 1_000_000
            + hidden_states * 10_000
            + restart
        )
        model = CategoricalHMM(
            hidden_states,
            pseudocount=float(hmm_config["pseudocount"]),
            max_iterations=int(hmm_config["max_iterations"]),
            min_iterations=int(hmm_config["min_iterations"]),
            tolerance_bits_per_event=float(
                hmm_config["tolerance_bits_per_event"]
            ),
            batch_size=int(hmm_config["batch_size"]),
        )
        fit = model.fit(train_sequences, seed=model_seed)
        fitted.append((fit, restart, model_seed, model))

    best_fit, best_restart, best_seed, best_model = max(
        fitted,
        key=lambda item: (
            item[0].training_log_likelihood_nats,
            -item[1],
        ),
    )

    hmm_aggregate, hmm_rows, _ = evaluate_hmm(
        best_model,
        test_examples,
        view=view,
        hidden_states=hidden_states,
        restart=best_restart,
        seed=best_seed,
    )

    trigram = WittenBellNGram(
        max_order=max(int(order) for order in ngram_config["orders"])
    )
    trigram.fit(
        example.symbols
        for example in train_examples
        if example.symbols
    )
    trigram_aggregates, trigram_rows, _ = evaluate_model(
        trigram,
        test_examples,
        orders=(3,),
        view=view,
    )
    trigram_aggregate = trigram_aggregates[0]

    if int(hmm_aggregate["total_events"]) != int(trigram_aggregate["total_events"]):
        raise RuntimeError(f"{view}: HMM/trigram aggregate event mismatch")

    hmm_by_unit = {row["leaf_group"]: row for row in hmm_rows}
    tri_by_unit = {row["leaf_group"]: row for row in trigram_rows}
    if set(hmm_by_unit) != set(tri_by_unit):
        raise RuntimeError(f"{view}: HMM/trigram unit identities differ")

    return {
        "view": view,
        "train_selected_restart": int(best_restart),
        "train_selected_seed": int(best_seed),
        "training_log_likelihood_nats": float(
            best_fit.training_log_likelihood_nats
        ),
        "hmm_aggregate": hmm_aggregate,
        "trigram_aggregate": trigram_aggregate,
        "hmm_by_unit": hmm_by_unit,
        "trigram_by_unit": tri_by_unit,
    }


def boundary_anatomy_for_unit(
    lines: Sequence[CipherLine],
) -> dict:
    pairs: Counter = Counter()
    for tokens in lines:
        for previous, current in zip(tokens, tokens[1:]):
            if not previous or not current:
                raise ValueError("Empty ciphertext token in boundary anatomy")
            pairs[(previous[-1], current[0])] += 1

    n = sum(pairs.values())
    if n <= 0:
        raise ValueError("Prospective unit has no within-line token boundaries")

    finals: Counter = Counter()
    initials: Counter = Counter()
    for (final_glyph, initial_glyph), count in pairs.items():
        finals[final_glyph] += count
        initials[initial_glyph] += count

    h_final = entropy(finals)
    h_initial = entropy(initials)
    h_pair = entropy(pairs)
    distinct_final = len(finals)
    distinct_initial = len(initials)
    distinct_pairs = len(pairs)
    possible_pairs = distinct_final * distinct_initial

    probs = sorted(
        (count / n for count in pairs.values()),
        reverse=True,
    )
    simpson = sum(probability ** 2 for probability in probs)

    return {
        "boundary_pair_count": n,
        "boundary_pair_entropy_bits": h_pair,
        "effective_boundary_pairs": 2.0 ** h_pair,
        "effective_final_glyphs": 2.0 ** h_final,
        "effective_initial_glyphs": 2.0 ** h_initial,
        "distinct_final_glyphs": distinct_final,
        "distinct_initial_glyphs": distinct_initial,
        "distinct_boundary_pairs": distinct_pairs,
        "possible_boundary_pairs_from_observed_marginals": possible_pairs,
        "observed_pair_support_fraction": (
            distinct_pairs / possible_pairs
            if possible_pairs > 0
            else 0.0
        ),
        "top5_boundary_pair_probability_mass": sum(probs[:5]),
        "boundary_pair_simpson_concentration": simpson,
        "boundary_pair_effective_simpson_pairs": (
            1.0 / simpson if simpson > 0.0 else 0.0
        ),
    }


def build_per_unit_rows(
    *,
    source_id: str,
    language: str,
    mechanism_seed: int,
    test_units: Sequence[Tuple[int, CipherLine]],
    assignment: Mapping[int, str],
    unit_summaries: Sequence[Mapping[str, object]],
    scored: Mapping[str, Mapping[str, object]],
) -> List[dict]:
    lines_by_unit: Dict[str, List[CipherLine]] = defaultdict(list)
    for line_index, tokens in test_units:
        lines_by_unit[assignment[int(line_index)]].append(tokens)

    summary_by_unit = {
        str(row["unit_id"]): row
        for row in unit_summaries
    }

    output: List[dict] = []
    for unit_id in sorted(lines_by_unit):
        row = {
            "source_id": source_id,
            "language": language,
            "mechanism_seed": int(mechanism_seed),
            "unit_id": unit_id,
            **summary_by_unit[unit_id],
            **boundary_anatomy_for_unit(lines_by_unit[unit_id]),
        }

        deltas = {}
        for view in ("space_free", "token_aware"):
            hmm = scored[view]["hmm_by_unit"][unit_id]
            tri = scored[view]["trigram_by_unit"][unit_id]

            hmm_events = int(hmm["total_events"])
            tri_events = int(tri["total_events"])
            if hmm_events != tri_events:
                raise RuntimeError(
                    f"{source_id}/seed={mechanism_seed}/{unit_id}/{view}: "
                    "HMM/trigram event mismatch"
                )

            hmm_bpe = float(hmm["total_bits"]) / hmm_events
            tri_bpe = float(tri["total_bits"]) / tri_events
            delta = hmm_bpe - tri_bpe
            deltas[view] = delta

            row[f"{view}_events"] = hmm_events
            row[f"{view}_hmm_bits_per_event"] = hmm_bpe
            row[f"{view}_trigram_bits_per_event"] = tri_bpe
            row[f"{view}_delta_bits_per_event"] = delta

        row["boundary_boost_bits_per_event"] = (
            deltas["token_aware"] - deltas["space_free"]
        )
        output.append(row)

    if len(output) != 15:
        raise RuntimeError(f"Expected 15 per-unit rows; found {len(output)}")
    return output


def write_control_artifacts(
    output_dir: Path,
    *,
    train_units: Sequence[Tuple[int, CipherLine]],
    test_units: Sequence[Tuple[int, CipherLine]],
    key: Mapping[str, object],
    provenance: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for line_index, tokens in train_units:
            handle.write(
                json.dumps(
                    {
                        "split": "train",
                        "line_index": int(line_index),
                        "tokens": [list(token) for token in tokens],
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
        for line_index, tokens in test_units:
            handle.write(
                json.dumps(
                    {
                        "split": "test_locked",
                        "line_index": int(line_index),
                        "tokens": [list(token) for token in tokens],
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_json(output_dir / "key.json", dict(key))
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "1.0",
            "experiment_id": "phase-5a-prospective-boundary-prediction-v1",
            "status": "confirmatory_prospective_control_prediction",
            "train_lines_encoded": len(train_units),
            "test_lines_encoded": len(test_units),
            "test_continues_directly_from_final_train_state": True,
            "provenance": dict(provenance),
        },
    )

    files = (
        corpus_path,
        output_dir / "key.json",
        output_dir / "summary.json",
    )
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def execute(config_path: Path, config: Mapping[str, object], frozen: Mapping[str, object]) -> int:
    output_cfg = config["output_paths"]
    result_root = Path(output_cfg["result_root"])
    control_root = Path(output_cfg["control_root"])
    marker_path = Path(output_cfg["access_marker"])
    final_summary_path = result_root / "summary.json"

    # These paths are not TEST inputs and may be checked safely.
    if marker_path.exists():
        raise RuntimeError(
            "Phase 5A prospective TEST access marker already exists. "
            "Refusing a second prospective execution."
        )
    if final_summary_path.exists():
        raise RuntimeError(
            "Phase 5A final summary already exists. "
            "Refusing a second prospective execution."
        )

    result_root.mkdir(parents=True, exist_ok=True)

    # CRITICAL: write marker BEFORE first filesystem operation on any TEST path.
    marker = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "test_access_boundary_crossed": True,
        "written_before_first_test_touch": True,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "phase5a_config_sha256": frozen["phase5a_config_sha256"],
        "fresh_manifest_sha256": frozen["manifest_sha256"],
        "d3_config_sha256": frozen["d3_config_sha256"],
        "d3_summary_sha256": frozen["d3_summary_sha256"],
        "mechanism_generator_sha256": frozen["mechanism_generator_sha256"],
        "hmm_config_sha256": frozen["hmm_config_sha256"],
        "ngram_config_sha256": frozen["ngram_config_sha256"],
        "sources": [
            {
                "source_id": source["source_id"],
                "language": source["language"],
                "test_locked_path": source["test_locked_path"],
                "expected_test_locked_sha256": source["test_locked_sha256"],
            }
            for source in frozen["sources"]
        ],
    }
    write_json(marker_path, marker)

    all_per_unit: List[dict] = []
    primary_rows: List[dict] = []
    secondary_rows: List[dict] = []
    source_execution = []
    seeds = [int(seed) for seed in config["seeds"]]

    secondary = dict(config["secondary_predictions"]["features"])

    for source in frozen["sources"]:
        source_id = str(source["source_id"])
        language = str(source["language"])
        train_lines = source["_train_lines"]

        # First TEST touch occurs here, after marker creation.
        test_path = Path(str(source["test_locked_path"]))
        if not test_path.is_file():
            raise FileNotFoundError(f"{source_id}: locked TEST missing: {test_path}")
        observed_test_sha = sha256_file(test_path)
        if observed_test_sha != source["test_locked_sha256"]:
            raise ValueError(f"{source_id}: locked TEST SHA mismatch")
        test_lines = mechanism.read_source_lines(test_path)

        if len(test_lines) != int(source["test_locked_lines"]):
            raise ValueError(f"{source_id}: locked TEST line-count mismatch")
        if mechanism.token_count(test_lines) != int(source["test_locked_tokens"]):
            raise ValueError(f"{source_id}: locked TEST token-count mismatch")
        if len(test_lines) < 15:
            raise ValueError(f"{source_id}: locked TEST requires at least 15 lines")

        print(
            f"[{language}] {source_id}: "
            f"train={len(train_lines)} lines, "
            f"test={len(test_lines)} lines"
        )

        source_execution.append(
            {
                "source_id": source_id,
                "language": language,
                "observed_test_sha256": observed_test_sha,
                "test_lines": len(test_lines),
                "test_tokens": mechanism.token_count(test_lines),
            }
        )

        for mechanism_seed in seeds:
            print(f"  seed={mechanism_seed}: encode -> unitize -> fit/score")

            key = mechanism.build_master_key(
                train_lines,
                seed=mechanism_seed,
            )
            counters: Counter = Counter()
            train_units = encode_lines(
                train_lines,
                key=key,
                seed=mechanism_seed,
                counters=counters,
            )
            final_train_state = dict(counters)
            test_units = encode_lines(
                test_lines,
                key=key,
                seed=mechanism_seed,
                counters=counters,
            )

            state_payload = [
                [list(base), int(count)]
                for base, count in sorted(
                    final_train_state.items(),
                    key=lambda item: repr(item[0]),
                )
            ]
            train_state_sha = sha256(
                json.dumps(
                    state_payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()

            assignment, unit_summaries = make_test_units(
                test_units,
                unit_count=int(
                    config["test_unitization"]["unit_count_per_source_seed"]
                ),
            )

            scored = {}
            for view in ("space_free", "token_aware"):
                scored[view] = fit_score_view(
                    train_units,
                    test_units,
                    assignment=assignment,
                    view=view,
                    hmm_config=frozen["hmm_config"],
                    ngram_config=frozen["ngram_config"],
                )

            unit_rows = build_per_unit_rows(
                source_id=source_id,
                language=language,
                mechanism_seed=mechanism_seed,
                test_units=test_units,
                assignment=assignment,
                unit_summaries=unit_summaries,
                scored=scored,
            )
            all_per_unit.extend(unit_rows)

            primary_feature = config["primary_prediction"]["feature"]
            primary_outcome = config["primary_prediction"]["outcome"]
            p, s = correlation(
                unit_rows,
                primary_feature,
                primary_outcome,
            )
            direction_match = (
                p is not None
                and s is not None
                and p > 0.0
                and s > 0.0
            )
            primary_rows.append(
                {
                    "source_id": source_id,
                    "language": language,
                    "mechanism_seed": mechanism_seed,
                    "test_units": len(unit_rows),
                    "feature": primary_feature,
                    "outcome": primary_outcome,
                    "pearson_r": p,
                    "spearman_rho": s,
                    "direction_match": direction_match,
                    "train_state_sha256": train_state_sha,
                    "space_free_train_selected_restart": scored[
                        "space_free"
                    ]["train_selected_restart"],
                    "space_free_train_selected_seed": scored[
                        "space_free"
                    ]["train_selected_seed"],
                    "token_aware_train_selected_restart": scored[
                        "token_aware"
                    ]["train_selected_restart"],
                    "token_aware_train_selected_seed": scored[
                        "token_aware"
                    ]["train_selected_seed"],
                }
            )

            for feature, expected_direction in secondary.items():
                sp, ss = correlation(
                    unit_rows,
                    feature,
                    primary_outcome,
                )
                expected_positive = expected_direction == "positive"
                direction_ok = (
                    sp is not None
                    and ss is not None
                    and (
                        (sp > 0.0 and ss > 0.0)
                        if expected_positive
                        else (sp < 0.0 and ss < 0.0)
                    )
                )
                secondary_rows.append(
                    {
                        "source_id": source_id,
                        "language": language,
                        "mechanism_seed": mechanism_seed,
                        "feature": feature,
                        "outcome": primary_outcome,
                        "expected_direction": expected_direction,
                        "pearson_r": sp,
                        "spearman_rho": ss,
                        "direction_match": direction_ok,
                        "scored_for_primary_decision": False,
                    }
                )

            write_control_artifacts(
                control_root
                / source_id
                / f"seed_{mechanism_seed}",
                train_units=train_units,
                test_units=test_units,
                key=key,
                provenance={
                    "phase5a_config_sha256": frozen[
                        "phase5a_config_sha256"
                    ],
                    "fresh_manifest_sha256": frozen["manifest_sha256"],
                    "source_train_sha256": source["train_sha256"],
                    "source_test_locked_sha256": observed_test_sha,
                    "mechanism_generator_sha256": frozen[
                        "mechanism_generator_sha256"
                    ],
                    "mechanism_seed": mechanism_seed,
                    "frozen_train_occurrence_state_sha256": train_state_sha,
                    "test_state_policy": (
                        "direct_from_final_train_state"
                    ),
                },
            )

    expected_runs = int(config["expected_grid"]["runs"])
    if len(primary_rows) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} primary runs; found {len(primary_rows)}"
        )
    expected_unit_rows = (
        expected_runs
        * int(config["expected_grid"]["test_units_per_run"])
    )
    if len(all_per_unit) != expected_unit_rows:
        raise RuntimeError(
            f"Expected {expected_unit_rows} unit rows; found {len(all_per_unit)}"
        )

    language_rows: List[dict] = []
    replicated_languages = []
    for language in config["fresh_controls"]["required_languages"]:
        group = [
            row
            for row in primary_rows
            if row["language"] == language
        ]
        if len(group) != 5:
            raise RuntimeError(
                f"{language}: expected five seed runs; found {len(group)}"
            )
        matches = sum(bool(row["direction_match"]) for row in group)
        replicated = matches >= 4
        if replicated:
            replicated_languages.append(language)
        language_rows.append(
            {
                "language": language,
                "seeds": 5,
                "direction_match_seeds": matches,
                "language_replicated_4_of_5": replicated,
                "mean_pearson_r": statistics.mean(
                    float(row["pearson_r"])
                    for row in group
                    if row["pearson_r"] is not None
                ),
                "mean_spearman_rho": statistics.mean(
                    float(row["spearman_rho"])
                    for row in group
                    if row["spearman_rho"] is not None
                ),
            }
        )

    supported = len(replicated_languages) >= 3
    decision = (
        config["decision_labels"]["support"]
        if supported
        else config["decision_labels"]["not_supported"]
    )

    write_csv(
        result_root / "per_unit_boundary_prediction.csv",
        all_per_unit,
    )
    write_csv(
        result_root / "per_run_primary_correlations.csv",
        primary_rows,
    )
    write_csv(
        result_root / "language_replication.csv",
        language_rows,
    )
    write_csv(
        result_root / "secondary_correlations.csv",
        secondary_rows,
    )

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "decision": decision,
        "primary_prediction_supported": supported,
        "runs": len(primary_rows),
        "per_unit_rows": len(all_per_unit),
        "languages_replicated": len(replicated_languages),
        "replicated_languages": replicated_languages,
        "language_replication": language_rows,
        "primary_feature": config["primary_prediction"]["feature"],
        "primary_outcome": config["primary_prediction"]["outcome"],
        "primary_run_rule": config["primary_prediction"][
            "per_run_direction_match"
        ],
        "language_rule": config["primary_prediction"][
            "language_replication_rule"
        ],
        "phase5a_rule": config["primary_prediction"][
            "phase5a_support_rule"
        ],
        "secondary_features_scored_for_primary_decision": False,
        "phase4_decision_changed": False,
        "voynich_files_accessed": False,
        "fresh_test_access_marker_sha256": sha256_file(marker_path),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(final_summary_path, summary)

    run_manifest = {
        "phase5a_config_path": str(config_path),
        "phase5a_config_sha256": frozen["phase5a_config_sha256"],
        "fresh_control_manifest_path": str(frozen["manifest_path"]),
        "fresh_control_manifest_sha256": frozen["manifest_sha256"],
        "d3_config_sha256": frozen["d3_config_sha256"],
        "d3_summary_sha256": frozen["d3_summary_sha256"],
        "d3_run_manifest_sha256": frozen["d3_run_manifest_sha256"],
        "mechanism_config_sha256": frozen["mechanism_config_sha256"],
        "mechanism_generator_sha256": frozen["mechanism_generator_sha256"],
        "hmm_config_sha256": frozen["hmm_config_sha256"],
        "ngram_config_sha256": frozen["ngram_config_sha256"],
        "source_execution": source_execution,
        "test_unitization": config["test_unitization"],
        "primary_prediction": config["primary_prediction"],
        "fresh_test_accessed": True,
        "voynich_files_accessed": False,
        "one_shot": True,
    }
    write_json(result_root / "run_manifest.json", run_manifest)

    checksum_targets = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (result_root / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(result_root)}"
            for path in checksum_targets
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("VOYAGER PHASE 5A — PROSPECTIVE BOUNDARY-STRUCTURE PREDICTION")
    print("=" * 96)
    print(f"Decision:              {decision}")
    print(f"Runs:                  {len(primary_rows)}")
    print(f"Per-unit rows:         {len(all_per_unit)}")
    print(
        "Languages replicated: "
        f"{len(replicated_languages)}/4"
        + (
            " (" + ", ".join(replicated_languages) + ")"
            if replicated_languages
            else ""
        )
    )
    print()
    for row in language_rows:
        print(
            f"{row['language']:<10s} "
            f"direction-match seeds={row['direction_match_seeds']}/5 "
            f"replicated={'YES' if row['language_replicated_4_of_5'] else 'NO'} "
            f"mean Pearson={float(row['mean_pearson_r']):+.4f} "
            f"mean Spearman={float(row['mean_spearman_rho']):+.4f}"
        )
    print()
    print("Phase 4 decision:      RETAINED UNCHANGED")
    print("Voynich files:         NOT ACCESSED")
    print(
        "Interpretation: a Phase 5A PASS is prospective control support for "
        "the D3-derived prediction, not decipherment or mechanism identification."
    )
    print(f"Saved: {result_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VOYAGER Phase 5A prospective boundary prediction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--execute-prospective-test",
        action="store_true",
        help=(
            "After protocol/data freeze, cross the one-shot fresh-control "
            "TEST boundary."
        ),
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        frozen = preflight(args.config, config)

        print("=" * 96)
        print("VOYAGER PHASE 5A — PROSPECTIVE BOUNDARY-STRUCTURE PREDICTION")
        print("=" * 96)
        print(f"Experiment:            {config['experiment_id']}")
        print(f"Status:                {config['status'].upper()}")
        print(f"Phase 5A cfg SHA:      {frozen['phase5a_config_sha256']}")
        print(f"Fresh manifest SHA:    {frozen['manifest_sha256']}")
        print(f"D3 config SHA:         {frozen['d3_config_sha256']}")
        print(f"Mechanism cfg SHA:     {frozen['mechanism_config_sha256']}")
        print(f"Mechanism gen SHA:     {frozen['mechanism_generator_sha256']}")
        print(f"HMM config SHA:        {frozen['hmm_config_sha256']}")
        print(f"N-gram config SHA:     {frozen['ngram_config_sha256']}")
        print("Mechanism:             recurrence-aware K=16 (FROZEN)")
        print("HMM:                   K=16 / TRAIN-only restart selection")
        print("Matched comparator:    order-3 Witten-Bell")
        print("Primary feature:       boundary_pair_entropy_bits = H(F,I)")
        print("Primary outcome:       token-aware delta - space-free delta")
        print("Units/run:             15 contiguous token-balanced units")
        print("Run rule:              Pearson > 0 AND Spearman > 0")
        print("Language rule:         >=4/5 seeds")
        print("Phase 5A rule:         >=3/4 languages")
        print("Voynich files:         NOT ACCESSED")
        print("Fresh locked TEST:     NOT ACCESSED DURING PREFLIGHT")
        print()

        for source in frozen["sources"]:
            print(
                f"[{source['language']}] {source['source_id']}: "
                f"TRAIN verified "
                f"({source['train_lines']} lines / "
                f"{source['train_tokens']} tokens); "
                "TEST path NOT TOUCHED"
            )

        print()
        print(
            "PRE-FLIGHT PASS — FRESH LOCKED TEST FILES REMAIN UNTOUCHED."
        )

        if not args.execute_prospective_test:
            print(
                "Freeze/commit the protocol, fresh manifest, TRAIN, and locked "
                "TEST resources before rerunning with:"
            )
            print(
                "  python scripts/run_phase_5a_boundary_prediction.py "
                "--execute-prospective-test"
            )
            return 0

        print()
        print(
            "EXECUTION REQUESTED — access marker will be written before "
            "the first fresh TEST file is touched."
        )
        return execute(args.config, config, frozen)

    except Exception as exc:
        print(
            f"PHASE 5A ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
