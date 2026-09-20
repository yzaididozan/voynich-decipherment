#!/usr/bin/env python3
"""VOYAGER Phase 4 locked Voynich evaluation.

Default behavior is PRE-FLIGHT ONLY and deliberately does not touch
data/splits/v1/test_LOCKED.txt.

With --execute-locked-test, this orchestrator:
1. creates an irreversible access marker before reading the locked split;
2. verifies the split and frozen provenance;
3. requires a repository-provided locked-test scorer implementing the exact
   already-selected five models with no test-set hyperparameter selection;
4. validates and aggregates its five relation rows under the preregistered rule.

Why an adapter?
---------------
The repository's existing Level-2 control evaluator performs held-out
hyperparameter selection. That behavior is appropriate for development/control
validation but is not admissible on the Voynich locked test. This Phase 4
orchestrator therefore refuses to reuse it directly.

The scorer contract is intentionally narrow:
    src.evaluation.phase4_voynich_locked.evaluate_locked_relations(...)

It must return exactly five relation rows, each containing:
    relation
    selected_hyperparameter
    competitor_bits_per_event
    trigram_bits_per_event
    delta_competitor_minus_trigram_bits_per_event
    bootstrap_units
    bootstrap_replicates
    bootstrap_seed
    ci_95_lower
    ci_95_upper

and may optionally return:
    bootstrap_rows: mapping relation -> rows

The adapter must only score the already-frozen settings supplied here.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Phase 4 config is YAML. Install PyYAML in the repository environment."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = Path("configs/phase_4_voynich_locked.yaml")
RELATIONS = (
    "hmm_space_free",
    "hmm_token_aware",
    "slot_grammar",
    "token_markov",
    "copy_edit",
)


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


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}"
        )
    return observed


def parse_leaf_ids(text: str, *, label: str) -> List[int]:
    leaves: List[int] = []
    seen = set()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        value = raw.strip()
        if not value:
            continue
        if not value.startswith("f") or not value[1:].isdigit():
            raise ValueError(f"{label}: invalid leaf on line {lineno}: {value!r}")
        leaf = int(value[1:])
        if leaf in seen:
            raise ValueError(f"{label}: duplicate leaf f{leaf}")
        seen.add(leaf)
        leaves.append(leaf)
    return sorted(leaves)


def require_phase3b(config: Mapping[str, object]) -> dict:
    spec = config["phase3b"]
    require_hash(
        Path(spec["config_path"]),
        spec["expected_config_sha256"],
        "Phase 3B config",
    )
    summary = read_json(Path(spec["result_summary_path"]))

    decision = (
        summary.get("decision")
        or summary.get("confirmatory_decision")
        or summary.get("phase_3b_decision")
        or summary.get("result")
    )
    if isinstance(decision, Mapping):
        decision = decision.get("decision") or decision.get("status")

    if decision != spec["required_decision"]:
        raise ValueError(
            f"Phase 3B result is not {spec['required_decision']!r}: {decision!r}"
        )

    runs = summary.get("runs", summary.get("total_runs"))
    rows = summary.get("relation_rows", summary.get("relations"))
    if int(runs) != int(spec["required_runs"]):
        raise ValueError(f"Phase 3B runs mismatch: {runs}")
    if int(rows) != int(spec["required_relation_rows"]):
        raise ValueError(f"Phase 3B relation-row mismatch: {rows}")

    hierarchy = (
        summary.get("full_hierarchy")
        if "full_hierarchy" in summary
        else summary.get("full_inherited_level2_hierarchy")
    )
    if hierarchy is not True:
        raise ValueError("Phase 3B summary does not report full_hierarchy=true")

    touched = summary.get(
        "voynich_locked_test_accessed",
        summary.get("voynich_test_accessed", False),
    )
    if touched is not False:
        raise ValueError("Phase 3B summary reports prior Voynich locked-test access")

    return summary


def require_static_frozen_inputs(config: Mapping[str, object]) -> dict:
    mech = config["confirmed_mechanism"]
    level2 = config["frozen_level2"]
    corpus = config["voynich_corpus"]
    splits = config["splits"]

    observed = {}
    observed["phase3a6_config"] = require_hash(
        Path(mech["config_path"]),
        mech["expected_config_sha256"],
        "Phase 3A.6 config",
    )
    observed["phase3a6_generator"] = require_hash(
        Path(mech["generator_path"]),
        mech["expected_generator_sha256"],
        "Phase 3A.6 generator",
    )
    observed["level2_config"] = require_hash(
        Path(level2["config_path"]),
        level2["expected_config_sha256"],
        "Level-2 config",
    )
    level2_data = read_json(Path(level2["config_path"]))
    if level2_data.get("freeze_id") != level2["required_freeze_id"]:
        raise ValueError("Unexpected frozen Level-2 freeze_id")
    if list(level2_data.get("operational_relations", [])) != list(
        level2["operational_relations"]
    ):
        raise ValueError("Level-2 operational relation list mismatch")

    observed["zl3b_source"] = require_hash(
        Path(corpus["source_path"]),
        corpus["expected_source_sha256"],
        "ZL3b source",
    )
    observed["sta1_rules"] = require_hash(
        Path(corpus["sta1_rules_path"]),
        corpus["expected_sta1_rules_sha256"],
        "STA1 rules",
    )
    observed["train_split"] = require_hash(
        Path(splits["train_path"]),
        splits["expected_train_sha256"],
        "TRAIN split",
    )
    observed["validation_split"] = require_hash(
        Path(splits["validation_path"]),
        splits["expected_validation_sha256"],
        "VALIDATION split",
    )

    # Reading TRAIN/VALIDATION is allowed. Do not even Path.exists/stat TEST here.
    train = parse_leaf_ids(
        Path(splits["train_path"]).read_text(encoding="utf-8"),
        label="TRAIN",
    )
    validation = parse_leaf_ids(
        Path(splits["validation_path"]).read_text(encoding="utf-8"),
        label="VALIDATION",
    )
    if len(train) != int(splits["expected_train_leaves"]):
        raise ValueError(f"Expected 72 TRAIN leaves; found {len(train)}")
    if len(validation) != int(splits["expected_validation_leaves"]):
        raise ValueError(f"Expected 15 VALIDATION leaves; found {len(validation)}")
    if set(train) & set(validation):
        raise ValueError("TRAIN/VALIDATION overlap")

    return {
        "hashes": observed,
        "train_leaves": train,
        "validation_leaves": validation,
    }


def frozen_validation_models(config: Mapping[str, object]) -> dict:
    models = config["validation_frozen_models"]
    summaries = config["validation_summaries"]

    slot = read_json(Path(summaries["slot_grammar"]))
    token = read_json(Path(summaries["token_markov"]))
    copy = read_json(Path(summaries["copy_edit"]))

    for name, payload in (("slot", slot), ("token", token), ("copy", copy)):
        if payload.get("locked_test_accessed") is not False:
            raise ValueError(
                f"{name} validation summary does not explicitly report "
                "locked_test_accessed=false"
            )

    slot_k = int(slot["selected_template_count"])
    if slot_k != int(models["slot_grammar"]["selected_template_count"]):
        raise ValueError(f"Frozen slot-grammar K mismatch: {slot_k}")

    token_context = int(token["selected_context_order"])
    allowed = [int(x) for x in models["token_markov"]["allowed_values"]]
    if token_context not in allowed:
        raise ValueError(
            f"Frozen Token-Markov context {token_context} outside {allowed}"
        )

    copy_window = int(copy["selected_source_window"])
    if copy_window != int(models["copy_edit"]["selected_source_window"]):
        raise ValueError(f"Frozen copy-edit window mismatch: {copy_window}")

    return {
        "hmm_space_free": int(
            models["hmm_space_free"]["selected_hidden_states"]
        ),
        "hmm_token_aware": int(
            models["hmm_token_aware"]["selected_hidden_states"]
        ),
        "slot_grammar": slot_k,
        "token_markov": token_context,
        "copy_edit": copy_window,
        "validation_summary_sha256": {
            "slot_grammar": sha256_file(Path(summaries["slot_grammar"])),
            "token_markov": sha256_file(Path(summaries["token_markov"])),
            "copy_edit": sha256_file(Path(summaries["copy_edit"])),
        },
    }


def preflight(config_path: Path, config: Mapping[str, object]) -> dict:
    phase3b = require_phase3b(config)
    static = require_static_frozen_inputs(config)
    selections = frozen_validation_models(config)

    config_sha = sha256_file(config_path)

    print("=" * 96)
    print("VOYAGER PHASE 4 — LOCKED VOYNICH EVALUATION")
    print("=" * 96)
    print(f"Experiment:         {config['experiment_id']}")
    print(f"Status:             {config['status'].upper()}")
    print(f"Phase 4 cfg SHA:    {config_sha}")
    print(f"Phase 3B decision:  {config['phase3b']['required_decision']}")
    print(f"Phase 3A.6 cfg:     {static['hashes']['phase3a6_config']}")
    print(f"Phase 3A.6 gen:     {static['hashes']['phase3a6_generator']}")
    print(f"Level-2 config:     {static['hashes']['level2_config']}")
    print(f"ZL3b source:        {static['hashes']['zl3b_source']}")
    print(f"STA1 rules:         {static['hashes']['sta1_rules']}")
    print(f"TRAIN split:        {static['hashes']['train_split']}")
    print(f"VALIDATION split:   {static['hashes']['validation_split']}")
    print()
    print("Frozen validation-selected models:")
    for relation in RELATIONS:
        print(f"  {relation:18s} {selections[relation]}")
    print()
    print("Voynich test:       NOT ACCESSED")
    print("test_LOCKED path:   NOT OPENED / NOT HASHED / NOT STATTED")
    print()
    print("PRE-FLIGHT PASS — VOYNICH LOCKED TEST REMAINS UNTOUCHED.")
    print(
        "Freeze/commit this protocol and record its hashes before rerunning with:\n"
        "  python scripts/run_phase_4_voynich_locked.py --execute-locked-test"
    )

    return {
        "phase4_config_sha256": config_sha,
        "static": static,
        "selections": selections,
        "phase3b_summary_sha256": sha256_file(
            Path(config["phase3b"]["result_summary_path"])
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_test_split(
    config: Mapping[str, object],
    *,
    train: Sequence[int],
    validation: Sequence[int],
) -> tuple[List[int], str]:
    # This function is called ONLY after the access marker is written.
    splits = config["splits"]
    path = Path(splits["locked_test_path"])
    text = path.read_text(encoding="utf-8")
    test_hash = sha256_file(path)
    test = parse_leaf_ids(text, label="TEST_LOCKED")

    if len(test) != int(splits["expected_test_leaves"]):
        raise ValueError(
            f"Expected {splits['expected_test_leaves']} TEST leaves; found {len(test)}"
        )

    train_set, val_set, test_set = set(train), set(validation), set(test)
    if train_set & test_set:
        raise ValueError("TRAIN/TEST overlap")
    if val_set & test_set:
        raise ValueError("VALIDATION/TEST overlap")

    union = train_set | val_set | test_set
    if len(union) != int(splits["expected_total_physical_leaves"]):
        raise ValueError(
            "TRAIN+VALIDATION+TEST does not contain the expected number of "
            f"physical leaves: {len(union)}"
        )

    linked = [int(x) for x in splits["rosettes_linked_leaves"]]
    memberships = [
        ("train" if x in train_set else "validation" if x in val_set else "test")
        for x in linked
    ]
    if len(set(memberships)) != 1:
        raise ValueError("Rosettes linked leaves f85/f86 are split apart")

    return test, test_hash


def load_locked_scorer():
    try:
        module = importlib.import_module("src.evaluation.phase4_voynich_locked")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Locked-test scorer adapter is not present: "
            "src/evaluation/phase4_voynich_locked.py\n\n"
            "This orchestrator intentionally refuses to pass test_LOCKED into the "
            "development Level-2 evaluator because that evaluator performs "
            "held-out hyperparameter selection. Implement/freeze the Phase 4 "
            "adapter before first locked-test access, then rerun PRE-FLIGHT and "
            "commit it with the protocol. DO NOT execute the locked test until "
            "that adapter exists and is frozen."
        ) from exc

    fn = getattr(module, "evaluate_locked_relations", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            "Phase 4 scorer adapter must export evaluate_locked_relations(...)"
        )
    return fn, module


def execute_locked_test(
    config_path: Path,
    config: Mapping[str, object],
    pf: Mapping[str, object],
) -> int:
    out_root = Path(config["execution"]["result_output_root"])
    marker = Path(config["execution"]["execution_marker"])
    final_summary = out_root / "summary.json"

    if marker.exists() or final_summary.exists():
        raise RuntimeError(
            "Phase 4 locked test has already been accessed or completed. "
            "Refusing a second confirmatory execution."
        )

    # Crucial: require/import the scorer BEFORE touching the test split.
    scorer, scorer_module = load_locked_scorer()
    scorer_path = Path(scorer_module.__file__)
    scorer_sha = sha256_file(scorer_path)

    # Write marker BEFORE the first read/stat/hash of test_LOCKED.
    marker_payload = {
        "experiment_id": config["experiment_id"],
        "event": "LOCKED_TEST_ACCESS_BEGUN",
        "phase4_config_sha256": pf["phase4_config_sha256"],
        "scorer_path": str(scorer_path),
        "scorer_sha256": scorer_sha,
        "phase3b_summary_sha256": pf["phase3b_summary_sha256"],
        "validation_frozen_models": pf["selections"],
        "note": (
            "Presence of this file means test_LOCKED access was authorized. "
            "Do not treat the locked split as untouched after this point."
        ),
    }
    write_json(marker, marker_payload)

    train = pf["static"]["train_leaves"]
    validation = pf["static"]["validation_leaves"]
    test, test_hash = validate_test_split(
        config,
        train=train,
        validation=validation,
    )

    evaluation = scorer(
        source_path=Path(config["voynich_corpus"]["source_path"]),
        sta1_rules_path=Path(config["voynich_corpus"]["sta1_rules_path"]),
        train_leaf_ids=tuple(train),
        test_leaf_ids=tuple(test),
        frozen_hyperparameters={
            relation: pf["selections"][relation] for relation in RELATIONS
        },
        bootstrap_replicates=int(
            config["test_evaluation"]["bootstrap_replicates"]
        ),
        bootstrap_seed=int(config["test_evaluation"]["bootstrap_seed"]),
    )

    if isinstance(evaluation, Mapping):
        relation_rows = list(evaluation.get("relations", []))
        bootstrap_rows = evaluation.get("bootstrap_rows", {})
    else:
        relation_rows = list(evaluation)
        bootstrap_rows = {}

    if len(relation_rows) != 5:
        raise ValueError(
            f"Locked scorer must return exactly 5 relation rows; got "
            f"{len(relation_rows)}"
        )

    by_relation = {}
    for raw in relation_rows:
        row = dict(raw)
        relation = str(row.get("relation", ""))
        if relation not in RELATIONS:
            raise ValueError(f"Unexpected relation from scorer: {relation!r}")
        if relation in by_relation:
            raise ValueError(f"Duplicate relation from scorer: {relation}")

        required = (
            "selected_hyperparameter",
            "competitor_bits_per_event",
            "trigram_bits_per_event",
            "delta_competitor_minus_trigram_bits_per_event",
            "bootstrap_units",
            "bootstrap_replicates",
            "bootstrap_seed",
            "ci_95_lower",
            "ci_95_upper",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{relation}: scorer row missing {missing}")

        if int(row["selected_hyperparameter"]) != int(pf["selections"][relation]):
            raise ValueError(
                f"{relation}: test scorer used hyperparameter "
                f"{row['selected_hyperparameter']}, frozen value is "
                f"{pf['selections'][relation]}"
            )
        if int(row["bootstrap_replicates"]) != int(
            config["test_evaluation"]["bootstrap_replicates"]
        ):
            raise ValueError(f"{relation}: bootstrap replicate count changed")
        if int(row["bootstrap_seed"]) != int(
            config["test_evaluation"]["bootstrap_seed"]
        ):
            raise ValueError(f"{relation}: bootstrap seed changed")

        delta = float(row["delta_competitor_minus_trigram_bits_per_event"])
        lower = float(row["ci_95_lower"])
        strong = delta > 0.0 and lower > 0.0
        row["direction_match_voynich"] = delta > 0.0
        row["strong_direction_match_voynich"] = strong
        row["locked_test_accessed"] = True
        by_relation[relation] = row

    if set(by_relation) != set(RELATIONS):
        raise ValueError("Scorer relation identity mismatch")

    ordered = [by_relation[x] for x in RELATIONS]
    all_five = all(bool(x["strong_direction_match_voynich"]) for x in ordered)
    decision = (
        config["decision_rule"]["if_all_five_match"]
        if all_five
        else config["decision_rule"]["if_any_relation_does_not_match"]
    )

    write_csv(out_root / "relation_results.csv", ordered)
    for relation, rows in dict(bootstrap_rows).items():
        if relation not in RELATIONS:
            raise ValueError(f"Unexpected bootstrap relation {relation!r}")
        if rows:
            write_csv(out_root / "bootstrap" / f"{relation}.csv", rows)

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "decision": decision,
        "all_five_relations_strong": all_five,
        "locked_test_accessed": True,
        "locked_test_sha256": test_hash,
        "test_physical_leaves": len(test),
        "relations": {
            row["relation"]: {
                "selected_hyperparameter": int(row["selected_hyperparameter"]),
                "delta_competitor_minus_trigram_bits_per_event": float(
                    row["delta_competitor_minus_trigram_bits_per_event"]
                ),
                "ci_95_lower": float(row["ci_95_lower"]),
                "ci_95_upper": float(row["ci_95_upper"]),
                "strong_direction_match_voynich": bool(
                    row["strong_direction_match_voynich"]
                ),
            }
            for row in ordered
        },
        "interpretation": (
            "A five-relation match is structural compatibility only and is "
            "non-discriminating among mechanisms that can reproduce the same "
            "fingerprint. It is not decipherment or mechanism identification."
        ),
    }
    write_json(final_summary, summary)

    manifest = {
        "phase4_config_sha256": pf["phase4_config_sha256"],
        "scorer_path": str(scorer_path),
        "scorer_sha256": scorer_sha,
        "source_sha256": pf["static"]["hashes"]["zl3b_source"],
        "sta1_rules_sha256": pf["static"]["hashes"]["sta1_rules"],
        "train_split_sha256": pf["static"]["hashes"]["train_split"],
        "validation_split_sha256": pf["static"]["hashes"]["validation_split"],
        "locked_test_sha256": test_hash,
        "phase3b_summary_sha256": pf["phase3b_summary_sha256"],
        "frozen_hyperparameters": {
            relation: pf["selections"][relation] for relation in RELATIONS
        },
        "bootstrap_replicates": int(
            config["test_evaluation"]["bootstrap_replicates"]
        ),
        "bootstrap_seed": int(config["test_evaluation"]["bootstrap_seed"]),
    }
    write_json(out_root / "run_manifest.json", manifest)

    files = sorted(
        p for p in out_root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    )
    (out_root / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(p)}  {p.relative_to(out_root)}" for p in files
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print(f"PHASE 4: {decision}")
    print("=" * 96)
    for row in ordered:
        print(
            f"{row['relation']:18s} "
            f"delta={float(row['delta_competitor_minus_trigram_bits_per_event']):+.6f} "
            f"CI=[{float(row['ci_95_lower']):+.6f},"
            f"{float(row['ci_95_upper']):+.6f}] "
            f"strong={'YES' if row['strong_direction_match_voynich'] else 'NO'}"
        )
    print()
    print(
        "Interpretation boundary: compatibility is not unique mechanism "
        "identification or decipherment."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VOYAGER Phase 4 locked Voynich evaluation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--execute-locked-test",
        action="store_true",
        help="Authorize the first access to Voynich test_LOCKED after freeze.",
    )
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        if config.get("experiment_id") != "phase-4-voynich-locked-evaluation-v1":
            raise ValueError("Unexpected Phase 4 experiment_id")
        if tuple(config["frozen_level2"]["operational_relations"]) != RELATIONS:
            raise ValueError("Phase 4 relation order differs from frozen relations")

        pf = preflight(args.config, config)

        if not args.execute_locked_test:
            return 0

        return execute_locked_test(args.config, config, pf)

    except Exception as exc:
        print(
            f"PHASE 4 SOFTWARE/PROTOCOL ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
