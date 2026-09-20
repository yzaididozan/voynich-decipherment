#!/usr/bin/env python3
"""VOYAGER Phase 4.D1 — post-hoc space-free HMM failure localization.

This script intentionally re-accesses the already-used Phase-4 Voynich TEST
split because the original locked scorer did not persist per-leaf HMM/trigram
score rows.

It is exploratory. It cannot change the Phase-4 confirmatory decision.

The diagnostic:
1. verifies the frozen Phase-4 parent result and implementation hashes;
2. reconstructs the exact K=16 space-free and token-aware HMM TEST scores;
3. reproduces the Phase-4 paired bootstrap and requires aggregate parity;
4. writes per-leaf deltas, exact corpus contributions, leave-one-out influence,
   and descriptive metadata-group summaries.

No validation split is read. No TEST hyperparameter selection is performed.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import make_analytical_loci
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.hmm import CategoricalHMM, evaluate_hmm
from src.models.ngram import (
    WittenBellNGram,
    atomic_leaf_group,
    build_baseline_sequences,
    evaluate_model,
    parse_leaf_ids,
    select_records_by_leaves,
)


DEFAULT_CONFIG = Path(
    "configs/diagnostics/phase4_space_free_hmm_localization_v1.json"
)
RELATIONS = ("hmm_space_free", "hmm_token_aware")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile from empty values")
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def index_leaf_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> Dict[str, Mapping[str, object]]:
    output: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        leaf = str(row.get("leaf_group", "")).strip()
        if not leaf:
            raise ValueError(f"{label}: missing leaf_group")
        if leaf in output:
            raise ValueError(f"{label}: duplicate leaf_group {leaf}")
        output[leaf] = row
    if not output:
        raise ValueError(f"{label}: no leaf rows")
    return output


def paired_bootstrap(
    competitor_rows: Sequence[Mapping[str, object]],
    trigram_rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    replicates: int,
) -> Tuple[dict, List[dict]]:
    competitor = index_leaf_rows(competitor_rows, label="HMM")
    trigram = index_leaf_rows(trigram_rows, label="trigram")

    if set(competitor) != set(trigram):
        raise ValueError("HMM/trigram atomic leaf groups differ")

    leaves = sorted(competitor)
    for leaf in leaves:
        c_events = int(float(competitor[leaf]["total_events"]))
        t_events = int(float(trigram[leaf]["total_events"]))
        if c_events != t_events:
            raise ValueError(
                f"{leaf}: HMM/trigram event mismatch {c_events} != {t_events}"
            )
        if c_events <= 0:
            raise ValueError(f"{leaf}: non-positive event count")

    def ratio(index, sample):
        bits = sum(float(index[leaf]["total_bits"]) for leaf in sample)
        events = sum(int(float(index[leaf]["total_events"])) for leaf in sample)
        return bits / events

    observed_hmm = ratio(competitor, leaves)
    observed_tri = ratio(trigram, leaves)
    observed_delta = observed_hmm - observed_tri

    rng = random.Random(int(seed))
    deltas: List[float] = []
    bootstrap_rows: List[dict] = []

    for replicate in range(int(replicates)):
        sample = [leaves[rng.randrange(len(leaves))] for _ in leaves]
        hmm_bpe = ratio(competitor, sample)
        tri_bpe = ratio(trigram, sample)
        delta = hmm_bpe - tri_bpe
        deltas.append(delta)
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "competitor_bits_per_event": hmm_bpe,
                "trigram_bits_per_event": tri_bpe,
                "delta_competitor_minus_trigram_bits_per_event": delta,
            }
        )

    return (
        {
            "bootstrap_units": len(leaves),
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "competitor_bits_per_event": observed_hmm,
            "trigram_bits_per_event": observed_tri,
            "delta_competitor_minus_trigram_bits_per_event": observed_delta,
            "ci_95_lower": percentile(deltas, 0.025),
            "ci_95_upper": percentile(deltas, 0.975),
        },
        bootstrap_rows,
    )


def require_parent_phase4(config: Mapping[str, object]) -> dict:
    parent = config["parent_phase4"]

    require_hash(
        Path(parent["config_path"]),
        parent["expected_config_sha256"],
        "Phase 4 config",
    )
    require_hash(
        Path(parent["scorer_path"]),
        parent["expected_scorer_sha256"],
        "Phase 4 scorer",
    )

    summary = read_json(Path(parent["summary_path"]))
    if summary.get("decision") != parent["required_decision"]:
        raise ValueError(
            "Unexpected Phase 4 decision: "
            f"{summary.get('decision')!r}"
        )
    if summary.get("locked_test_accessed") is not True:
        raise ValueError("Phase 4 summary does not state locked_test_accessed=true")
    if summary.get("all_five_relations_strong") is not False:
        raise ValueError(
            "Phase 4 summary does not state all_five_relations_strong=false"
        )

    observed_status = {
        relation: bool(
            summary["relations"][relation]["strong_direction_match_voynich"]
        )
        for relation in parent["required_relation_status"]
    }
    expected_status = {
        relation: bool(value)
        for relation, value in parent["required_relation_status"].items()
    }
    if observed_status != expected_status:
        raise ValueError(
            "Phase 4 relation-status pattern mismatch\n"
            f"  expected={expected_status}\n"
            f"  observed={observed_status}"
        )

    relation_rows = read_csv(Path(parent["relation_results_path"]))
    relation_index = {
        row["relation"]: row
        for row in relation_rows
    }
    if set(relation_index) != set(expected_status):
        raise ValueError(
            "Phase 4 relation_results.csv identity mismatch: "
            f"{sorted(relation_index)}"
        )

    return {
        "summary": summary,
        "relation_rows": relation_rows,
        "relation_index": relation_index,
        "summary_sha256": sha256_file(Path(parent["summary_path"])),
        "relation_results_sha256": sha256_file(
            Path(parent["relation_results_path"])
        ),
    }


def load_frozen_inputs(config: Mapping[str, object], parent: Mapping[str, object]) -> dict:
    inputs = config["voynich_inputs"]
    models = config["frozen_models"]

    source_hash = require_hash(
        Path(inputs["source_path"]),
        inputs["expected_source_sha256"],
        "ZL3b source",
    )
    rules_hash = require_hash(
        Path(inputs["sta1_rules_path"]),
        inputs["expected_sta1_rules_sha256"],
        "STA1 rules",
    )
    train_hash = require_hash(
        Path(inputs["train_split_path"]),
        inputs["expected_train_sha256"],
        "TRAIN split",
    )
    hmm_hash = require_hash(
        Path(models["hmm_config_path"]),
        models["expected_hmm_config_sha256"],
        "HMM config",
    )
    ngram_hash = require_hash(
        Path(models["ngram_config_path"]),
        models["expected_ngram_config_sha256"],
        "n-gram config",
    )

    expected_test_hash = str(parent["summary"].get("locked_test_sha256", ""))
    if len(expected_test_hash) != 64:
        raise ValueError(
            "Phase 4 summary lacks a valid locked_test_sha256"
        )
    test_hash = require_hash(
        Path(inputs["locked_test_path"]),
        expected_test_hash,
        "already-used TEST split",
    )

    hmm_cfg = read_json(Path(models["hmm_config_path"]))
    ngram_cfg = read_json(Path(models["ngram_config_path"]))

    required_hmm = {
        "hidden_states": [2, 4, 8, 16],
        "views": ["space_free", "token_aware"],
        "restarts": 5,
        "restart_selection": "highest_training_log_likelihood",
        "seed": 40814041438,
        "pseudocount": 0.5,
    }
    for key, value in required_hmm.items():
        if hmm_cfg.get(key) != value:
            raise ValueError(
                f"Frozen HMM config mismatch for {key}: "
                f"{hmm_cfg.get(key)!r} != {value!r}"
            )

    if ngram_cfg.get("orders") != [1, 2, 3, 4, 5]:
        raise ValueError("Frozen n-gram order grid changed")
    if ngram_cfg.get("views") != ["space_free", "token_aware"]:
        raise ValueError("Frozen n-gram views changed")

    train_leaves = parse_leaf_ids(
        Path(inputs["train_split_path"]).read_text(encoding="utf-8"),
        label="train",
    )
    test_leaves = parse_leaf_ids(
        Path(inputs["locked_test_path"]).read_text(encoding="utf-8"),
        label="phase4_test_locked",
    )

    if len(train_leaves) != int(inputs["expected_train_leaves"]):
        raise ValueError(
            f"Expected {inputs['expected_train_leaves']} TRAIN leaves; "
            f"found {len(train_leaves)}"
        )
    if len(test_leaves) != int(inputs["expected_test_leaves"]):
        raise ValueError(
            f"Expected {inputs['expected_test_leaves']} TEST leaves; "
            f"found {len(test_leaves)}"
        )
    if set(train_leaves) & set(test_leaves):
        raise ValueError("TRAIN/TEST overlap")

    return {
        "source_hash": source_hash,
        "rules_hash": rules_hash,
        "train_hash": train_hash,
        "test_hash": test_hash,
        "hmm_config_hash": hmm_hash,
        "ngram_config_hash": ngram_hash,
        "hmm_config": hmm_cfg,
        "ngram_config": ngram_cfg,
        "train_leaves": train_leaves,
        "test_leaves": test_leaves,
    }


def fit_and_score_view(
    train_loci,
    test_loci,
    *,
    view: str,
    hidden_states: int,
    hmm_cfg: Mapping[str, object],
    ngram_cfg: Mapping[str, object],
) -> dict:
    view_index = list(hmm_cfg["views"]).index(view)

    train_examples = build_baseline_sequences(train_loci, view=view)
    test_examples = build_baseline_sequences(test_loci, view=view)
    train_sequences = [
        example.symbols
        for example in train_examples
        if example.symbols
    ]

    fitted = []
    for restart in range(int(hmm_cfg["restarts"])):
        seed = (
            int(hmm_cfg["seed"])
            + view_index * 1_000_000
            + int(hidden_states) * 10_000
            + restart
        )
        model = CategoricalHMM(
            int(hidden_states),
            pseudocount=float(hmm_cfg["pseudocount"]),
            max_iterations=int(hmm_cfg["max_iterations"]),
            min_iterations=int(hmm_cfg["min_iterations"]),
            tolerance_bits_per_event=float(
                hmm_cfg["tolerance_bits_per_event"]
            ),
            batch_size=int(hmm_cfg["batch_size"]),
        )
        fit = model.fit(train_sequences, seed=seed)
        fitted.append((fit, restart, seed, model))

    best_fit, best_restart, best_seed, best_model = max(
        fitted,
        key=lambda item: (
            item[0].training_log_likelihood_nats,
            -item[1],
        ),
    )

    hmm_aggregate, hmm_leaves, hmm_loci = evaluate_hmm(
        best_model,
        test_examples,
        view=view,
        hidden_states=int(hidden_states),
        restart=best_restart,
        seed=best_seed,
    )

    trigram = WittenBellNGram(
        max_order=max(int(x) for x in ngram_cfg["orders"])
    )
    trigram.fit(
        example.symbols
        for example in train_examples
        if example.symbols
    )
    tri_aggregates, tri_leaves, tri_loci = evaluate_model(
        trigram,
        test_examples,
        orders=(3,),
        view=view,
    )
    tri_aggregate = tri_aggregates[0]

    if int(hmm_aggregate["total_events"]) != int(
        tri_aggregate["total_events"]
    ):
        raise ValueError(
            f"{view}: aggregate HMM/trigram event mismatch"
        )

    return {
        "view": view,
        "hidden_states": int(hidden_states),
        "train_selected_restart": int(best_restart),
        "train_selected_seed": int(best_seed),
        "training_log_likelihood_nats": float(
            best_fit.training_log_likelihood_nats
        ),
        "hmm_aggregate": hmm_aggregate,
        "hmm_leaves": hmm_leaves,
        "hmm_loci": hmm_loci,
        "trigram_aggregate": tri_aggregate,
        "trigram_leaves": tri_leaves,
        "trigram_loci": tri_loci,
    }


def parity_rows(
    *,
    reconstructed: Mapping[str, Mapping[str, object]],
    parent_index: Mapping[str, Mapping[str, object]],
    tolerance: float,
) -> List[dict]:
    fields = (
        "competitor_bits_per_event",
        "trigram_bits_per_event",
        "delta_competitor_minus_trigram_bits_per_event",
        "ci_95_lower",
        "ci_95_upper",
    )
    integer_fields = (
        "bootstrap_units",
        "bootstrap_replicates",
        "bootstrap_seed",
    )
    rows: List[dict] = []

    for relation in RELATIONS:
        observed = reconstructed[relation]
        expected = parent_index[relation]

        for field in fields:
            left = float(observed[field])
            right = float(expected[field])
            difference = left - right
            rows.append(
                {
                    "relation": relation,
                    "field": field,
                    "phase4_value": right,
                    "reconstructed_value": left,
                    "difference": difference,
                    "tolerance": tolerance,
                    "pass": abs(difference) <= tolerance,
                }
            )

        for field in integer_fields:
            left = int(observed[field])
            right = int(float(expected[field]))
            rows.append(
                {
                    "relation": relation,
                    "field": field,
                    "phase4_value": right,
                    "reconstructed_value": left,
                    "difference": left - right,
                    "tolerance": 0,
                    "pass": left == right,
                }
            )

    return rows


def metadata_label(values: Iterable[object]) -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if value not in (None, "")
            and str(value).strip()
    }
    )
    if not cleaned:
        return "UNKNOWN"
    if len(cleaned) == 1:
        return cleaned[0]
    return "MIXED[" + "|".join(cleaned) + "]"


def leaf_metadata(test_loci) -> Dict[str, dict]:
    fields = ("section", "currier", "scribe", "quire", "locus_type")
    buckets: Dict[str, MutableMapping[str, list]] = {}

    for locus in test_loci:
        leaf = atomic_leaf_group(locus)
        bucket = buckets.setdefault(
            leaf,
            {
                "folios": [],
                "physical_leaves": [],
                **{field: [] for field in fields},
            },
        )
        bucket["folios"].append(locus.folio)
        if locus.physical_leaf is not None:
            bucket["physical_leaves"].append(locus.physical_leaf)
        for field in fields:
            bucket[field].append(getattr(locus, field, None))

    output: Dict[str, dict] = {}
    for leaf, bucket in buckets.items():
        output[leaf] = {
            "leaf_group": leaf,
            "folios": "|".join(sorted(set(map(str, bucket["folios"])))),
            "physical_leaves": "|".join(
                str(x) for x in sorted(set(bucket["physical_leaves"]))
            ),
            **{
                f"{field}_label": metadata_label(bucket[field])
                for field in fields
            },
        }
    return output


def build_per_leaf_rows(
    scored_by_view: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, object]],
) -> List[dict]:
    view_data = {}
    for view, scored in scored_by_view.items():
        hmm = index_leaf_rows(scored["hmm_leaves"], label=f"{view} HMM")
        tri = index_leaf_rows(
            scored["trigram_leaves"],
            label=f"{view} trigram",
        )
        if set(hmm) != set(tri):
            raise ValueError(f"{view}: leaf identities differ")
        total_events = sum(
            int(float(row["total_events"]))
            for row in hmm.values()
        )
        view_data[view] = {
            "hmm": hmm,
            "tri": tri,
            "total_events": total_events,
        }

    leaf_sets = [set(item["hmm"]) for item in view_data.values()]
    if len(leaf_sets) != 2 or leaf_sets[0] != leaf_sets[1]:
        raise ValueError("space-free/token-aware leaf identities differ")

    rows: List[dict] = []
    for leaf in sorted(leaf_sets[0]):
        row = dict(metadata.get(leaf, {"leaf_group": leaf}))
        row["leaf_group"] = leaf

        for view in ("space_free", "token_aware"):
            hmm = view_data[view]["hmm"][leaf]
            tri = view_data[view]["tri"][leaf]
            events = int(float(hmm["total_events"]))
            tri_events = int(float(tri["total_events"]))
            if events != tri_events:
                raise ValueError(
                    f"{view} {leaf}: HMM/trigram event mismatch"
                )

            hmm_bits = float(hmm["total_bits"])
            tri_bits = float(tri["total_bits"])
            delta_bits = hmm_bits - tri_bits
            leaf_delta = delta_bits / events
            contribution = (
                delta_bits / view_data[view]["total_events"]
            )

            prefix = view
            row[f"{prefix}_events"] = events
            row[f"{prefix}_hmm_total_bits"] = hmm_bits
            row[f"{prefix}_trigram_total_bits"] = tri_bits
            row[f"{prefix}_hmm_bits_per_event"] = hmm_bits / events
            row[f"{prefix}_trigram_bits_per_event"] = tri_bits / events
            row[f"{prefix}_delta_bits"] = delta_bits
            row[f"{prefix}_delta_bits_per_event"] = leaf_delta
            row[
                f"{prefix}_contribution_to_corpus_delta_bits_per_event"
            ] = contribution
            row[f"{prefix}_direction"] = (
                "positive"
                if leaf_delta > 0
                else "negative"
                if leaf_delta < 0
                else "zero"
            )

        row["token_aware_minus_space_free_delta_bits_per_event"] = (
            float(row["token_aware_delta_bits_per_event"])
            - float(row["space_free_delta_bits_per_event"])
        )
        row["sign_concordant"] = (
            row["space_free_direction"] == row["token_aware_direction"]
        )
        rows.append(row)

    return rows


def corpus_delta_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    view: str,
    excluded_leaf: str | None = None,
) -> float:
    selected = [
        row for row in rows
        if excluded_leaf is None or row["leaf_group"] != excluded_leaf
    ]
    hmm_bits = sum(float(row[f"{view}_hmm_total_bits"]) for row in selected)
    tri_bits = sum(
        float(row[f"{view}_trigram_total_bits"]) for row in selected
    )
    events = sum(int(row[f"{view}_events"]) for row in selected)
    return (hmm_bits - tri_bits) / events


def leave_one_out_rows(per_leaf: Sequence[Mapping[str, object]]) -> List[dict]:
    output: List[dict] = []
    full = {
        view: corpus_delta_from_rows(per_leaf, view=view)
        for view in ("space_free", "token_aware")
    }

    for row in per_leaf:
        leaf = str(row["leaf_group"])
        out = {
            "leaf_group": leaf,
            "space_free_full_delta_bits_per_event": full["space_free"],
            "token_aware_full_delta_bits_per_event": full["token_aware"],
        }
        for view in ("space_free", "token_aware"):
            loo = corpus_delta_from_rows(
                per_leaf,
                view=view,
                excluded_leaf=leaf,
            )
            out[f"{view}_leave_one_out_delta_bits_per_event"] = loo
            out[f"{view}_loo_minus_full_delta_bits_per_event"] = (
                loo - full[view]
            )
            out[f"{view}_influence_interpretation"] = (
                "leaf_pulls_delta_down"
                if loo - full[view] > 0
                else "leaf_pulls_delta_up"
                if loo - full[view] < 0
                else "no_change"
            )
        output.append(out)

    return sorted(
        output,
        key=lambda item: float(
            item["space_free_loo_minus_full_delta_bits_per_event"]
        ),
        reverse=True,
    )


def group_summary(
    per_leaf: Sequence[Mapping[str, object]],
    *,
    field: str,
) -> List[dict]:
    groups: Dict[str, List[Mapping[str, object]]] = {}
    label_field = f"{field}_label"
    for row in per_leaf:
        groups.setdefault(str(row[label_field]), []).append(row)

    output: List[dict] = []
    for label, rows in sorted(groups.items()):
        result = {
            "metadata_field": field,
            "metadata_value": label,
            "atomic_leaf_groups": len(rows),
            "leaf_groups": "|".join(sorted(str(x["leaf_group"]) for x in rows)),
        }

        for view in ("space_free", "token_aware"):
            events = sum(int(row[f"{view}_events"]) for row in rows)
            hmm_bits = sum(
                float(row[f"{view}_hmm_total_bits"])
                for row in rows
            )
            tri_bits = sum(
                float(row[f"{view}_trigram_total_bits"])
                for row in rows
            )
            result[f"{view}_events"] = events
            result[f"{view}_hmm_bits_per_event"] = hmm_bits / events
            result[f"{view}_trigram_bits_per_event"] = tri_bits / events
            result[f"{view}_delta_bits_per_event"] = (
                (hmm_bits - tri_bits) / events
            )
            result[f"{view}_positive_leaf_groups"] = sum(
                float(row[f"{view}_delta_bits_per_event"]) > 0
                for row in rows
            )
            result[f"{view}_negative_leaf_groups"] = sum(
                float(row[f"{view}_delta_bits_per_event"]) < 0
                for row in rows
            )

        output.append(result)

    return output


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0.0 or dy == 0.0:
        return None
    return numerator / (dx * dy)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc localization of the Phase-4 space-free HMM mismatch."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a prior deterministic D1 output directory.",
    )
    args = parser.parse_args()

    try:
        config = read_json(args.config)
        if (
            config.get("experiment_id")
            != "phase4-space-free-hmm-localization-v1"
        ):
            raise ValueError("Unexpected Phase 4.D1 experiment_id")
        if config.get("status") != "exploratory_posthoc_locked_test_diagnostic":
            raise ValueError("Phase 4.D1 must remain explicitly exploratory")

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(config["output_root"])
        )
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite only to reproduce the same frozen D1."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        parent = require_parent_phase4(config)
        frozen = load_frozen_inputs(config, parent)

        records = load_ivtff(
            Path(config["voynich_inputs"]["source_path"]),
            transcription_id=config["voynich_inputs"]["transcription_id"],
            strict=True,
        )
        train_records = select_records_by_leaves(
            records,
            frozen["train_leaves"],
            split_name="phase4d1_train",
        )
        test_records = select_records_by_leaves(
            records,
            frozen["test_leaves"],
            split_name="phase4d1_test_already_used",
        )
        rules = load_bitrans_rules(
            Path(config["voynich_inputs"]["sta1_rules_path"]),
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != frozen["rules_hash"]:
            raise ValueError("Loaded STA1 rule hash changed")

        train_loci = make_analytical_loci(train_records, rules)
        test_loci = make_analytical_loci(test_records, rules)

        scored_by_view = {}
        reconstructed = {}
        bootstrap_rows = {}

        for view, relation in (
            ("space_free", "hmm_space_free"),
            ("token_aware", "hmm_token_aware"),
        ):
            print(f"[Phase 4.D1] reconstructing {relation} at frozen K=16...")
            scored = fit_and_score_view(
                train_loci,
                test_loci,
                view=view,
                hidden_states=16,
                hmm_cfg=frozen["hmm_config"],
                ngram_cfg=frozen["ngram_config"],
            )
            comparison, boot = paired_bootstrap(
                scored["hmm_leaves"],
                scored["trigram_leaves"],
                seed=int(config["bootstrap"]["seed"]),
                replicates=int(config["bootstrap"]["replicates"]),
            )
            comparison["relation"] = relation
            comparison["selected_hyperparameter"] = 16
            comparison["train_selected_restart"] = scored[
                "train_selected_restart"
            ]
            comparison["train_selected_seed"] = scored[
                "train_selected_seed"
            ]

            scored_by_view[view] = scored
            reconstructed[relation] = comparison
            bootstrap_rows[relation] = boot

        parity = parity_rows(
            reconstructed=reconstructed,
            parent_index=parent["relation_index"],
            tolerance=float(config["phase4_parity"]["absolute_tolerance"]),
        )
        if not all(bool(row["pass"]) for row in parity):
            failures = [
                row for row in parity
                if not bool(row["pass"])
            ]
            raise ValueError(
                "Phase 4 aggregate/bootstrap parity failed; refusing "
                f"localization. First failures: {failures[:5]}"
            )

        metadata = leaf_metadata(test_loci)
        per_leaf = build_per_leaf_rows(scored_by_view, metadata)
        loo = leave_one_out_rows(per_leaf)

        group_rows: List[dict] = []
        for field in config["diagnostics"]["metadata"]["fields"]:
            group_rows.extend(group_summary(per_leaf, field=field))

        write_csv(output_dir / "aggregate_parity.csv", parity)
        write_csv(output_dir / "per_leaf_hmm_deltas.csv", per_leaf)
        write_csv(output_dir / "leave_one_leaf_out.csv", loo)
        write_csv(output_dir / "metadata_group_summary.csv", group_rows)
        for relation in RELATIONS:
            write_csv(
                output_dir / "bootstrap" / f"{relation}.csv",
                bootstrap_rows[relation],
            )

        space = [
            float(row["space_free_delta_bits_per_event"])
            for row in per_leaf
        ]
        token = [
            float(row["token_aware_delta_bits_per_event"])
            for row in per_leaf
        ]
        sign_concordant = sum(
            bool(row["sign_concordant"])
            for row in per_leaf
        )

        most_downward = sorted(
            loo,
            key=lambda row: float(
                row["space_free_loo_minus_full_delta_bits_per_event"]
            ),
            reverse=True,
        )[:5]
        most_upward = sorted(
            loo,
            key=lambda row: float(
                row["space_free_loo_minus_full_delta_bits_per_event"]
            ),
        )[:5]

        summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": config["status"],
            "parent_phase4_decision_retained": parent["summary"]["decision"],
            "phase4_reclassification_permitted": False,
            "phase4_parity_all_passed": True,
            "atomic_leaf_groups": len(per_leaf),
            "reconstructed_relations": {
                relation: reconstructed[relation]
                for relation in RELATIONS
            },
            "space_free_leaf_direction_counts": {
                "positive": sum(x > 0 for x in space),
                "negative": sum(x < 0 for x in space),
                "zero": sum(x == 0 for x in space),
            },
            "token_aware_leaf_direction_counts": {
                "positive": sum(x > 0 for x in token),
                "negative": sum(x < 0 for x in token),
                "zero": sum(x == 0 for x in token),
            },
            "cross_view": {
                "pearson_leaf_delta_correlation": pearson(space, token),
                "sign_concordant_leaf_groups": sign_concordant,
                "sign_discordant_leaf_groups": len(per_leaf) - sign_concordant,
                "mean_token_aware_minus_space_free_leaf_delta": (
                    sum(
                        float(
                            row[
                                "token_aware_minus_space_free_delta_bits_per_event"
                            ]
                        )
                        for row in per_leaf
                    )
                    / len(per_leaf)
                ),
            },
            "most_space_free_downward_influential_leaf_groups": [
                {
                    "leaf_group": row["leaf_group"],
                    "loo_minus_full_delta_bits_per_event": row[
                        "space_free_loo_minus_full_delta_bits_per_event"
                    ],
                }
                for row in most_downward
            ],
            "most_space_free_upward_influential_leaf_groups": [
                {
                    "leaf_group": row["leaf_group"],
                    "loo_minus_full_delta_bits_per_event": row[
                        "space_free_loo_minus_full_delta_bits_per_event"
                    ],
                }
                for row in most_upward
            ],
            "interpretation_boundary": (
                "Post-hoc localization only. No leaf, section, subgroup, "
                "alternative K, or alternative threshold may be used to "
                "retroactively convert Phase 4 into a pass."
            ),
        }
        write_json(output_dir / "summary.json", summary)

        manifest = {
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "phase4_summary_sha256": parent["summary_sha256"],
            "phase4_relation_results_sha256": parent[
                "relation_results_sha256"
            ],
            "phase4_scorer_sha256": config["parent_phase4"][
                "expected_scorer_sha256"
            ],
            "source_sha256": frozen["source_hash"],
            "sta1_rules_sha256": frozen["rules_hash"],
            "train_split_sha256": frozen["train_hash"],
            "test_split_sha256": frozen["test_hash"],
            "hmm_config_sha256": frozen["hmm_config_hash"],
            "ngram_config_sha256": frozen["ngram_config_hash"],
            "test_was_already_accessed_before_diagnostic": True,
            "validation_split_read_by_diagnostic": False,
            "test_hyperparameter_selection": False,
            "hmm_hidden_states": 16,
            "matched_trigram_order": 3,
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "bootstrap_seed": int(config["bootstrap"]["seed"]),
        }
        write_json(output_dir / "run_manifest.json", manifest)

        checksum_targets = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS"
        )
        (output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
                for path in checksum_targets
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 96)
        print("PHASE 4.D1 — SPACE-FREE HMM FAILURE LOCALIZATION")
        print("=" * 96)
        print("Status:                  EXPLORATORY / POST-HOC")
        print("Phase 4 decision:        RETAINED UNCHANGED")
        print("Aggregate parity:        PASS")
        print(f"Atomic leaf groups:      {len(per_leaf)}")
        print()
        print(
            "space-free leaf signs:   "
            f"+{sum(x > 0 for x in space)} / "
            f"-{sum(x < 0 for x in space)} / "
            f"0={sum(x == 0 for x in space)}"
        )
        print(
            "token-aware leaf signs:  "
            f"+{sum(x > 0 for x in token)} / "
            f"-{sum(x < 0 for x in token)} / "
            f"0={sum(x == 0 for x in token)}"
        )
        corr = pearson(space, token)
        print(
            "cross-view leaf Pearson: "
            + ("NA" if corr is None else f"{corr:+.4f}")
        )
        print()
        print("Largest downward space-free influences:")
        for row in most_downward[:5]:
            print(
                f"  {row['leaf_group']:10s} "
                f"LOO-full="
                f"{float(row['space_free_loo_minus_full_delta_bits_per_event']):+.6f}"
            )
        print()
        print(
            "Interpretation boundary: this diagnostic cannot reclassify "
            "the Phase-4 locked result."
        )
        print(f"Saved: {output_dir}")
        return 0

    except Exception as exc:
        print(
            f"PHASE 4.D1 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
