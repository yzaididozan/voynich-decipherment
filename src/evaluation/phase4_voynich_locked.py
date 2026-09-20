"""Locked-test scorer for VOYAGER Phase 4.

This module is intentionally *not* a model-selection pipeline.

It fits the already-frozen Voynich baseline/competitor models on the frozen
TRAIN leaves and evaluates exactly one preselected setting per relation on the
locked TEST leaves. No TEST statistic is allowed to select a hyperparameter,
restart count, model family, preprocessing rule, bootstrap setting, or
interpretation threshold.

Public API
----------
evaluate_locked_relations(
    source_path,
    sta1_rules_path,
    train_leaf_ids,
    test_leaf_ids,
    frozen_hyperparameters,
    bootstrap_replicates,
    bootstrap_seed,
) -> {
    "relations": [... exactly five rows ...],
    "bootstrap_rows": {relation: [...]},
}

Relation delta convention
-------------------------
    delta = competitor bits/event - matched trigram bits/event

Positive delta therefore means the matched character trigram has lower
held-out negative log likelihood.

Leakage safeguards
------------------
* TRAIN is the only fitting data.
* TEST is scored once with already-frozen hyperparameters.
* HMM/slot random restarts are chosen using TRAIN likelihood only.
* No validation file is read by this module.
* No TEST-driven grid search or fallback is implemented.
* Paired uncertainty resamples atomic physical-leaf groups with replacement.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.analysis.tier0 import make_analytical_loci
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.copy_edit import (
    LocalCopyEditModel,
    build_matched_trigram_examples as build_copy_matched_trigram_examples,
    evaluate_copy_edit,
    extract_copy_sequences,
)
from src.models.hmm import CategoricalHMM, evaluate_hmm
from src.models.ngram import (
    WittenBellNGram,
    build_baseline_sequences,
    evaluate_model,
    select_records_by_leaves,
)
from src.models.slot_grammar import (
    FiniteTemplateSlotGrammar,
    build_matched_trigram_examples as build_slot_matched_trigram_examples,
    evaluate_slot_grammar,
    extract_slot_tokens,
)
from src.models.token_markov import (
    TokenMarkovModel,
    build_matched_trigram_examples as build_token_matched_trigram_examples,
    evaluate_token_markov,
    extract_token_sequences,
)


RELATIONS: Tuple[str, ...] = (
    "hmm_space_free",
    "hmm_token_aware",
    "slot_grammar",
    "token_markov",
    "copy_edit",
)

# Phase-4 preflight established these exact validation-selected settings.
EXPECTED_FROZEN_HYPERPARAMETERS = {
    "hmm_space_free": 16,
    "hmm_token_aware": 16,
    "slot_grammar": 16,
    "token_markov": 1,
    "copy_edit": 16,
}

# Repository model configs. These are development-time specifications only;
# this scorer validates them but never uses TEST to select among their grids.
HMM_CONFIG_PATH = Path("configs/baselines/hmm_v1.json")
NGRAM_CONFIG_PATH = Path("configs/baselines/ngram_v1.json")
SLOT_CONFIG_PATH = Path("configs/baselines/slot_grammar_v1.json")
TOKEN_CONFIG_PATH = Path("configs/baselines/token_markov_v1.json")
COPY_CONFIG_PATH = Path("configs/baselines/copy_edit_v1.json")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen model config missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _require_exact(mapping: Mapping[str, object], expected: Mapping[str, object], *, label: str) -> None:
    for key, value in expected.items():
        observed = mapping.get(key)
        if observed != value:
            raise ValueError(
                f"{label}: unexpected frozen value for {key!r}: "
                f"observed={observed!r}, expected={value!r}"
            )


def _load_and_validate_configs() -> dict:
    hmm = _read_json(HMM_CONFIG_PATH)
    ngram = _read_json(NGRAM_CONFIG_PATH)
    slot = _read_json(SLOT_CONFIG_PATH)
    token = _read_json(TOKEN_CONFIG_PATH)
    copy = _read_json(COPY_CONFIG_PATH)

    _require_exact(
        hmm,
        {
            "schema_version": "1.0",
            "hidden_states": [2, 4, 8, 16],
            "views": ["space_free", "token_aware"],
            "restarts": 5,
            "restart_selection": "highest_training_log_likelihood",
            "validation_selection": "lowest_bits_per_event",
            "pseudocount": 0.5,
            "seed": 40814041438,
            "unknown_policy": "reset_and_do_not_score",
        },
        label="HMM config",
    )
    _require_exact(
        ngram,
        {
            "schema_version": "1.0",
            "orders": [1, 2, 3, 4, 5],
            "views": ["space_free", "token_aware"],
            "smoothing": "interpolated_witten_bell",
            "unknown_policy": "reset_and_do_not_score",
        },
        label="n-gram config",
    )
    _require_exact(
        slot,
        {
            "schema_version": "1.0",
            "templates": [1, 2, 4, 8, 16],
            "slot_count": 8,
            "max_token_length": 64,
            "restarts": 5,
            "restart_selection": "highest_training_log_likelihood",
            "validation_selection": "lowest_bits_per_event",
            "matched_trigram_order": 3,
            "pseudocount": 0.5,
            "seed": 40814041438,
            "unknown_policy": "exclude_token_if_contains_Z1",
        },
        label="slot-grammar config",
    )
    _require_exact(
        token,
        {
            "schema_version": "1.0",
            "context_orders": [1, 2],
            "max_context": 2,
            "char_base_order": 3,
            "smoothing": "interpolated_witten_bell_to_char_trigram",
            "sequence_boundary": "locus_or_Z1_reset",
            "bos_policy": "left_pad_with_TOKEN_BOS",
            "unknown_policy": "exclude_token_if_contains_Z1_and_reset_context",
            "matched_comparator": "token_reset_char_trigram_with_explicit_EOT",
        },
        label="Token-Markov config",
    )
    _require_exact(
        copy,
        {
            "schema_version": "1.0",
            "source_windows": [1, 4, 16],
            "max_token_length": 64,
            "pseudocount": 0.5,
            "rho_max_iterations": 100,
            "rho_tolerance": 1e-10,
            "validation_selection": "lowest_bits_per_event",
            "matched_trigram_order": 3,
            "source_scope": "previous_readable_tokens_within_same_locus_segment",
            "training_source_inference": (
                "minimum_normalized_levenshtein_then_raw_distance_then_recency"
            ),
            "unknown_policy": "exclude_token_if_contains_Z1_and_reset_context",
        },
        label="copy-edit config",
    )

    return {
        "hmm": hmm,
        "ngram": ngram,
        "slot": slot,
        "token": token,
        "copy": copy,
        "sha256": {
            "hmm": sha256_file(HMM_CONFIG_PATH),
            "ngram": sha256_file(NGRAM_CONFIG_PATH),
            "slot": sha256_file(SLOT_CONFIG_PATH),
            "token": sha256_file(TOKEN_CONFIG_PATH),
            "copy": sha256_file(COPY_CONFIG_PATH),
        },
    }


def _validate_frozen_hyperparameters(values: Mapping[str, object]) -> Dict[str, int]:
    if set(values) != set(RELATIONS):
        missing = sorted(set(RELATIONS) - set(values))
        extra = sorted(set(values) - set(RELATIONS))
        raise ValueError(
            "Frozen hyperparameter identity mismatch: "
            f"missing={missing}, extra={extra}"
        )

    normalized = {name: int(values[name]) for name in RELATIONS}
    if normalized != EXPECTED_FROZEN_HYPERPARAMETERS:
        raise ValueError(
            "Phase 4 frozen hyperparameters differ from the preflight freeze.\n"
            f"  observed: {normalized}\n"
            f"  expected: {EXPECTED_FROZEN_HYPERPARAMETERS}"
        )
    return normalized


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")

    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _index_leaf_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> Dict[str, Mapping[str, object]]:
    indexed: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        leaf = str(row.get("leaf_group", "")).strip()
        if not leaf:
            raise ValueError(f"{label}: row missing leaf_group")
        if leaf in indexed:
            raise ValueError(f"{label}: duplicate row for {leaf}")
        indexed[leaf] = row

    if not indexed:
        raise ValueError(f"{label}: no per-leaf rows")
    return indexed


def _paired_bootstrap(
    competitor_leaf_rows: Sequence[Mapping[str, object]],
    trigram_leaf_rows: Sequence[Mapping[str, object]],
    *,
    competitor_event_field: str,
    trigram_event_field: str,
    seed: int,
    replicates: int,
) -> Tuple[dict, List[dict]]:
    """Paired atomic-leaf bootstrap using corpus ratio-of-sums.

    Duplicated sampled leaves contribute their bits/events repeatedly, exactly
    as required by a nonparametric bootstrap over leaf groups.
    """
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be >= 1")

    competitor = _index_leaf_rows(
        competitor_leaf_rows,
        label="competitor",
    )
    trigram = _index_leaf_rows(
        trigram_leaf_rows,
        label="matched trigram",
    )

    if set(competitor) != set(trigram):
        raise ValueError(
            "Competitor and matched trigram do not contain identical "
            "atomic leaf groups"
        )

    leaves = sorted(competitor)
    if not leaves:
        raise ValueError("No paired atomic leaf groups")

    for leaf in leaves:
        c_events = int(float(competitor[leaf][competitor_event_field]))
        t_events = int(float(trigram[leaf][trigram_event_field]))
        if c_events <= 0 or t_events <= 0:
            raise ValueError(f"{leaf}: non-positive event count")
        if c_events != t_events:
            raise ValueError(
                f"{leaf}: competitor/trigram event denominators differ: "
                f"{c_events} != {t_events}"
            )

    def corpus_bits_per_event(
        index: Mapping[str, Mapping[str, object]],
        sample: Sequence[str],
        *,
        event_field: str,
    ) -> float:
        bits = sum(float(index[leaf]["total_bits"]) for leaf in sample)
        events = sum(int(float(index[leaf][event_field])) for leaf in sample)
        if events <= 0:
            raise ValueError("Bootstrap sample has non-positive event count")
        return bits / events

    observed_competitor = corpus_bits_per_event(
        competitor,
        leaves,
        event_field=competitor_event_field,
    )
    observed_trigram = corpus_bits_per_event(
        trigram,
        leaves,
        event_field=trigram_event_field,
    )
    observed_delta = observed_competitor - observed_trigram

    rng = random.Random(int(seed))
    deltas: List[float] = []
    bootstrap_rows: List[dict] = []

    for replicate in range(int(replicates)):
        sample = [leaves[rng.randrange(len(leaves))] for _ in leaves]
        c_bpe = corpus_bits_per_event(
            competitor,
            sample,
            event_field=competitor_event_field,
        )
        t_bpe = corpus_bits_per_event(
            trigram,
            sample,
            event_field=trigram_event_field,
        )
        delta = c_bpe - t_bpe
        deltas.append(delta)
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "competitor_bits_per_event": c_bpe,
                "trigram_bits_per_event": t_bpe,
                "delta_competitor_minus_trigram_bits_per_event": delta,
            }
        )

    lower = _percentile(deltas, 0.025)
    upper = _percentile(deltas, 0.975)

    positive = sum(delta > 0.0 for delta in deltas)
    negative = sum(delta < 0.0 for delta in deltas)
    ties = len(deltas) - positive - negative

    return (
        {
            "bootstrap_units": len(leaves),
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "competitor_bits_per_event": observed_competitor,
            "trigram_bits_per_event": observed_trigram,
            "delta_competitor_minus_trigram_bits_per_event": observed_delta,
            "relative_nll_reduction_trigram_vs_competitor": (
                (observed_competitor - observed_trigram) / observed_competitor
                if observed_competitor > 0.0
                else None
            ),
            "ci_95_lower": lower,
            "ci_95_upper": upper,
            "bootstrap_probability_trigram_better": positive / len(deltas),
            "bootstrap_probability_competitor_better": negative / len(deltas),
            "bootstrap_probability_tie": ties / len(deltas),
        },
        bootstrap_rows,
    )


def _matched_ngram_for_baseline_view(
    train_loci,
    test_loci,
    *,
    view: str,
    ngram_config: Mapping[str, object],
) -> Tuple[dict, List[dict]]:
    """Fit the original Tier-1 n-gram family and return frozen order-3 TEST rows."""
    train_examples = build_baseline_sequences(train_loci, view=view)
    test_examples = build_baseline_sequences(test_loci, view=view)

    model = WittenBellNGram(max_order=max(int(x) for x in ngram_config["orders"]))
    model.fit(example.symbols for example in train_examples if example.symbols)
    aggregates, leaves, _ = evaluate_model(
        model,
        test_examples,
        orders=(3,),
        view=view,
    )
    if len(aggregates) != 1:
        raise RuntimeError(f"{view}: expected one trigram aggregate row")
    return aggregates[0], leaves


def _score_hmm_relation(
    train_loci,
    test_loci,
    *,
    view: str,
    hidden_states: int,
    hmm_config: Mapping[str, object],
    ngram_config: Mapping[str, object],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Tuple[dict, List[dict]]:
    view_index = list(hmm_config["views"]).index(view)
    train_examples = build_baseline_sequences(train_loci, view=view)
    test_examples = build_baseline_sequences(test_loci, view=view)
    train_sequences = [
        example.symbols for example in train_examples if example.symbols
    ]

    fitted = []
    for restart in range(int(hmm_config["restarts"])):
        seed = (
            int(hmm_config["seed"])
            + view_index * 1_000_000
            + int(hidden_states) * 10_000
            + restart
        )
        model = CategoricalHMM(
            int(hidden_states),
            pseudocount=float(hmm_config["pseudocount"]),
            max_iterations=int(hmm_config["max_iterations"]),
            min_iterations=int(hmm_config["min_iterations"]),
            tolerance_bits_per_event=float(
                hmm_config["tolerance_bits_per_event"]
            ),
            batch_size=int(hmm_config["batch_size"]),
        )
        fit = model.fit(train_sequences, seed=seed)
        fitted.append((fit, restart, seed, model))

    # Restart selection is legal because it uses TRAIN likelihood only.
    best_fit, best_restart, best_seed, best_model = max(
        fitted,
        key=lambda item: (
            item[0].training_log_likelihood_nats,
            -item[1],
        ),
    )

    hmm_aggregate, hmm_leaves, _ = evaluate_hmm(
        best_model,
        test_examples,
        view=view,
        hidden_states=int(hidden_states),
        restart=best_restart,
        seed=best_seed,
    )

    trigram_aggregate, trigram_leaves = _matched_ngram_for_baseline_view(
        train_loci,
        test_loci,
        view=view,
        ngram_config=ngram_config,
    )

    if int(hmm_aggregate["total_events"]) != int(
        trigram_aggregate["total_events"]
    ):
        raise ValueError(
            f"HMM {view}: aggregate event denominator differs from trigram"
        )

    comparison, boot = _paired_bootstrap(
        hmm_leaves,
        trigram_leaves,
        competitor_event_field="total_events",
        trigram_event_field="total_events",
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    relation = (
        "hmm_space_free" if view == "space_free" else "hmm_token_aware"
    )
    comparison.update(
        {
            "relation": relation,
            "competitor": "categorical_hmm",
            "selected_hyperparameter": int(hidden_states),
            "train_selected_restart": int(best_restart),
            "train_selected_seed": int(best_seed),
            "training_log_likelihood_nats": float(
                best_fit.training_log_likelihood_nats
            ),
        }
    )
    return comparison, boot


def _score_slot_relation(
    train_loci,
    test_loci,
    *,
    templates: int,
    config: Mapping[str, object],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Tuple[dict, List[dict]]:
    train_tokens, _ = extract_slot_tokens(train_loci)
    test_tokens, _ = extract_slot_tokens(test_loci)

    max_train = max(len(example.glyphs) for example in train_tokens)
    max_test = max(len(example.glyphs) for example in test_tokens)
    if max(max_train, max_test) > int(config["max_token_length"]):
        raise ValueError(
            "Slot grammar encountered a token exceeding frozen max length"
        )

    train_sequences = [example.glyphs for example in train_tokens]
    fitted = []

    for restart in range(int(config["restarts"])):
        seed = int(config["seed"]) + int(templates) * 10_000 + restart
        model = FiniteTemplateSlotGrammar(
            int(templates),
            slot_count=int(config["slot_count"]),
            max_token_length=int(config["max_token_length"]),
            pseudocount=float(config["pseudocount"]),
            max_iterations=int(config["max_iterations"]),
            min_iterations=int(config["min_iterations"]),
            tolerance_bits_per_token=float(
                config["tolerance_bits_per_token"]
            ),
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

    slot_aggregate, slot_leaves, _ = evaluate_slot_grammar(
        best_model,
        test_tokens,
        templates=int(templates),
        restart=best_restart,
        seed=best_seed,
    )

    train_tri = build_slot_matched_trigram_examples(train_tokens)
    test_tri = build_slot_matched_trigram_examples(test_tokens)
    trigram = WittenBellNGram(max_order=int(config["matched_trigram_order"]))
    trigram.fit(example.symbols for example in train_tri)
    tri_agg_rows, tri_leaves, _ = evaluate_model(
        trigram,
        test_tri,
        orders=(int(config["matched_trigram_order"]),),
        view="token_reset_eot",
    )
    tri_agg = tri_agg_rows[0]

    if int(slot_aggregate["events"]) != int(tri_agg["total_events"]):
        raise ValueError("Slot grammar/trigram aggregate event mismatch")

    comparison, boot = _paired_bootstrap(
        slot_leaves,
        tri_leaves,
        competitor_event_field="events",
        trigram_event_field="total_events",
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    comparison.update(
        {
            "relation": "slot_grammar",
            "competitor": "finite_template_slot_grammar",
            "selected_hyperparameter": int(templates),
            "train_selected_restart": int(best_restart),
            "train_selected_seed": int(best_seed),
            "training_log_likelihood_nats": float(
                best_fit.training_log_likelihood_nats
            ),
        }
    )
    return comparison, boot


def _score_token_markov_relation(
    train_loci,
    test_loci,
    *,
    context_order: int,
    config: Mapping[str, object],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Tuple[dict, List[dict]]:
    train_sequences, _ = extract_token_sequences(train_loci)
    test_sequences, _ = extract_token_sequences(test_loci)

    train_tokens = [
        [example.glyphs for example in sequence]
        for sequence in train_sequences
    ]
    model = TokenMarkovModel(
        max_context=int(config["max_context"]),
        char_base_order=int(config["char_base_order"]),
    )
    model.fit(train_tokens)

    token_aggregate, token_leaves, _ = evaluate_token_markov(
        model,
        test_sequences,
        context_order=int(context_order),
    )

    test_tri = build_token_matched_trigram_examples(test_sequences)
    tri_agg_rows, tri_leaves, _ = evaluate_model(
        model.char_base_model,
        test_tri,
        orders=(int(config["char_base_order"]),),
        view="token_reset_eot",
    )
    tri_agg = tri_agg_rows[0]

    if int(token_aggregate["events"]) != int(tri_agg["total_events"]):
        raise ValueError("Token-Markov/trigram aggregate event mismatch")

    # Exact matched-comparator parity: the external comparator must be the same
    # fitted character model used as TokenMarkovModel's open-vocabulary base.
    direct_base_bits = 0.0
    for sequence in test_sequences:
        for example in sequence:
            probability = model.character_token_probability(example.glyphs)
            if probability <= 0.0:
                raise RuntimeError("Non-positive Token-Markov character-base P")
            direct_base_bits += -math.log2(probability)

    if abs(direct_base_bits - float(tri_agg["total_bits"])) > 1e-8:
        raise ValueError(
            "Token-Markov internal character base and matched trigram differ"
        )

    comparison, boot = _paired_bootstrap(
        token_leaves,
        tri_leaves,
        competitor_event_field="events",
        trigram_event_field="total_events",
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    comparison.update(
        {
            "relation": "token_markov",
            "competitor": "exact_token_markov",
            "selected_hyperparameter": int(context_order),
        }
    )
    return comparison, boot


def _score_copy_edit_relation(
    train_loci,
    test_loci,
    *,
    source_window: int,
    config: Mapping[str, object],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> Tuple[dict, List[dict]]:
    train_sequences, _ = extract_copy_sequences(train_loci)
    test_sequences, _ = extract_copy_sequences(test_loci)

    max_train = max(
        len(example.glyphs)
        for sequence in train_sequences
        for example in sequence
    )
    max_test = max(
        len(example.glyphs)
        for sequence in test_sequences
        for example in sequence
    )
    if max(max_train, max_test) > int(config["max_token_length"]):
        raise ValueError(
            "Copy-edit encountered a token exceeding frozen max length"
        )

    train_tokens = [
        [example.glyphs for example in sequence]
        for sequence in train_sequences
    ]
    model = LocalCopyEditModel(
        int(source_window),
        max_token_length=int(config["max_token_length"]),
        pseudocount=float(config["pseudocount"]),
        rho_max_iterations=int(config["rho_max_iterations"]),
        rho_tolerance=float(config["rho_tolerance"]),
    )
    fit = model.fit(train_tokens)

    copy_aggregate, copy_leaves, _ = evaluate_copy_edit(
        model,
        test_sequences,
    )

    train_tri = build_copy_matched_trigram_examples(train_sequences)
    test_tri = build_copy_matched_trigram_examples(test_sequences)
    trigram = WittenBellNGram(max_order=int(config["matched_trigram_order"]))
    trigram.fit(example.symbols for example in train_tri)
    tri_agg_rows, tri_leaves, _ = evaluate_model(
        trigram,
        test_tri,
        orders=(int(config["matched_trigram_order"]),),
        view="token_reset_eot",
    )
    tri_agg = tri_agg_rows[0]

    if int(copy_aggregate["events"]) != int(tri_agg["total_events"]):
        raise ValueError("Copy-edit/trigram aggregate event mismatch")

    comparison, boot = _paired_bootstrap(
        copy_leaves,
        tri_leaves,
        competitor_event_field="events",
        trigram_event_field="total_events",
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    comparison.update(
        {
            "relation": "copy_edit",
            "competitor": "local_copy_edit",
            "selected_hyperparameter": int(source_window),
            "training_rho_copy": float(fit.rho_copy),
        }
    )
    return comparison, boot


def evaluate_locked_relations(
    *,
    source_path: Path,
    sta1_rules_path: Path,
    train_leaf_ids: Sequence[int],
    test_leaf_ids: Sequence[int],
    frozen_hyperparameters: Mapping[str, object],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    """Fit on TRAIN and score the five frozen relations on locked TEST.

    The function deliberately has no validation argument.
    """
    frozen = _validate_frozen_hyperparameters(frozen_hyperparameters)

    train_leaf_ids = tuple(sorted(int(x) for x in train_leaf_ids))
    test_leaf_ids = tuple(sorted(int(x) for x in test_leaf_ids))

    if not train_leaf_ids or not test_leaf_ids:
        raise ValueError("TRAIN and TEST leaf lists must both be non-empty")
    if set(train_leaf_ids) & set(test_leaf_ids):
        raise ValueError("TRAIN/TEST leaf overlap")
    if int(bootstrap_replicates) != 10_000:
        raise ValueError(
            "Phase 4 bootstrap_replicates must remain exactly 10000"
        )
    if int(bootstrap_seed) != 40814041438:
        raise ValueError(
            "Phase 4 bootstrap_seed must remain exactly 40814041438"
        )

    configs = _load_and_validate_configs()

    records = load_ivtff(
        Path(source_path),
        transcription_id="ZL3b",
        strict=True,
    )
    train_records = select_records_by_leaves(
        records,
        train_leaf_ids,
        split_name="phase4_train",
    )
    test_records = select_records_by_leaves(
        records,
        test_leaf_ids,
        split_name="phase4_test_locked",
    )

    rules = load_bitrans_rules(
        Path(sta1_rules_path),
        expected_native_alphabet="Eva-",
    )
    train_loci = make_analytical_loci(train_records, rules)
    test_loci = make_analytical_loci(test_records, rules)

    relation_rows: List[dict] = []
    bootstrap_rows: Dict[str, List[dict]] = {}

    for view, relation in (
        ("space_free", "hmm_space_free"),
        ("token_aware", "hmm_token_aware"),
    ):
        row, boot = _score_hmm_relation(
            train_loci,
            test_loci,
            view=view,
            hidden_states=frozen[relation],
            hmm_config=configs["hmm"],
            ngram_config=configs["ngram"],
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_replicates=int(bootstrap_replicates),
        )
        relation_rows.append(row)
        bootstrap_rows[relation] = boot

    row, boot = _score_slot_relation(
        train_loci,
        test_loci,
        templates=frozen["slot_grammar"],
        config=configs["slot"],
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_replicates=int(bootstrap_replicates),
    )
    relation_rows.append(row)
    bootstrap_rows["slot_grammar"] = boot

    row, boot = _score_token_markov_relation(
        train_loci,
        test_loci,
        context_order=frozen["token_markov"],
        config=configs["token"],
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_replicates=int(bootstrap_replicates),
    )
    relation_rows.append(row)
    bootstrap_rows["token_markov"] = boot

    row, boot = _score_copy_edit_relation(
        train_loci,
        test_loci,
        source_window=frozen["copy_edit"],
        config=configs["copy"],
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_replicates=int(bootstrap_replicates),
    )
    relation_rows.append(row)
    bootstrap_rows["copy_edit"] = boot

    if [row["relation"] for row in relation_rows] != list(RELATIONS):
        raise AssertionError("Internal Phase 4 relation order mismatch")

    # Add provenance that the orchestrator will retain in relation_results.csv.
    for row in relation_rows:
        row["fit_scope"] = "VOYNICH_TRAIN_ONLY"
        row["evaluation_scope"] = "VOYNICH_TEST_LOCKED_ONLY"
        row["test_hyperparameter_selection"] = False
        row["bootstrap_method"] = (
            "paired atomic-physical-leaf-group resampling with replacement; "
            "ratio-of-sums bits/event"
        )
        row["model_config_sha256"] = configs["sha256"][
            {
                "hmm_space_free": "hmm",
                "hmm_token_aware": "hmm",
                "slot_grammar": "slot",
                "token_markov": "token",
                "copy_edit": "copy",
            }[row["relation"]]
        ]
        if row["relation"].startswith("hmm_"):
            row["matched_ngram_config_sha256"] = configs["sha256"]["ngram"]

    return {
        "relations": relation_rows,
        "bootstrap_rows": bootstrap_rows,
        "provenance": {
            "source_path": str(source_path),
            "source_sha256": sha256_file(Path(source_path)),
            "sta1_rules_path": str(sta1_rules_path),
            "sta1_rules_sha256": sha256_file(Path(sta1_rules_path)),
            "train_leaf_count": len(train_leaf_ids),
            "test_leaf_count": len(test_leaf_ids),
            "frozen_hyperparameters": dict(frozen),
            "model_config_sha256": configs["sha256"],
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "validation_read_by_scorer": False,
        },
    }
