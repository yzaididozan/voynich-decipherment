"""Generic Level-2 predictive-hierarchy evaluation for VOYAGER controls.

This module adapts the already-frozen VOYAGER model implementations to the
synthetic historical cipher-control corpora. It does not modify the model
classes or their hyperparameter grids.

Each JSONL control row is treated as one source-line unit. The same frozen
train/validation/reserved-test unit membership is used for every family and
key derived from a given source language.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.models.ngram import (
    BaselineSequence,
    WB,
    WittenBellNGram,
    evaluate_model,
)
from src.models.hmm import (
    CategoricalHMM,
    evaluate_hmm,
)
from src.models.slot_grammar import (
    FiniteTemplateSlotGrammar,
    SlotTokenExample,
    evaluate_slot_grammar,
)
from src.models.token_markov import (
    TokenMarkovExample,
    TokenMarkovModel,
    evaluate_token_markov,
)
from src.models.copy_edit import (
    CopyTokenExample,
    LocalCopyEditModel,
    evaluate_copy_edit,
)


EOT = "<EOT>"

Token = Tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(x) for x in parts)])
    value = int.from_bytes(
        sha256(payload.encode("utf-8")).digest()[:8],
        "big",
    )
    return value % (2**63 - 1)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def largest_remainder_counts(
    total: int,
    weights: Mapping[str, int],
) -> Dict[str, int]:
    if total < 1:
        raise ValueError("total must be >= 1")

    names = list(weights)
    weight_sum = sum(int(weights[name]) for name in names)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    quotas = {
        name: total * int(weights[name]) / weight_sum
        for name in names
    }
    counts = {
        name: math.floor(quotas[name])
        for name in names
    }

    remaining = total - sum(counts.values())
    ranking = sorted(
        names,
        key=lambda name: (
            -(quotas[name] - counts[name]),
            names.index(name),
        ),
    )

    for name in ranking[:remaining]:
        counts[name] += 1

    if sum(counts.values()) != total:
        raise RuntimeError("largest-remainder allocation failed")

    return counts


def deterministic_unit_split(
    source_id: str,
    line_count: int,
    *,
    seed: int,
    weights: Mapping[str, int],
) -> dict:
    counts = largest_remainder_counts(
        line_count,
        weights,
    )

    ranked = sorted(
        range(1, line_count + 1),
        key=lambda index: (
            sha256(
                f"{seed}|{source_id}|{index}".encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )

    cursor = 0
    split = {}

    for name in weights:
        count = counts[name]
        split[name] = sorted(
            ranked[cursor : cursor + count]
        )
        cursor += count

    if cursor != line_count:
        raise RuntimeError("split did not consume all source lines")

    memberships = [
        set(split[name])
        for name in weights
    ]
    for i, left in enumerate(memberships):
        for right in memberships[i + 1 :]:
            if left & right:
                raise RuntimeError("split overlap detected")

    return {
        "source_id": source_id,
        "source_line_count": line_count,
        "split_seed": seed,
        "assignment": "sha256_rank_without_replacement",
        "count_allocation": "largest_remainder",
        "weights": dict(weights),
        "counts": counts,
        "train": split["train"],
        "validation": split["validation"],
        "reserved_test": split["reserved_test"],
    }


def read_control_units(
    corpus_path: Path,
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    expected_line_count: int,
) -> Tuple[
    List[Tuple[int, Tuple[Token, ...]]],
    List[Tuple[int, Tuple[Token, ...]]],
]:
    """Read only train/validation JSON payloads from one monolithic control.

    Reserved-test rows are streamed past but are not JSON-parsed, tokenized,
    modeled, or scored.
    """
    train_set = set(int(x) for x in train_indices)
    validation_set = set(
        int(x) for x in validation_indices
    )

    if train_set & validation_set:
        raise ValueError("train/validation line overlap")

    train: List[Tuple[int, Tuple[Token, ...]]] = []
    validation: List[Tuple[int, Tuple[Token, ...]]] = []

    observed_lines = 0

    with corpus_path.open(encoding="utf-8") as handle:
        for observed_lines, raw in enumerate(handle, start=1):
            if (
                observed_lines not in train_set
                and observed_lines not in validation_set
            ):
                continue

            payload = json.loads(raw)
            declared = int(
                payload.get("line_index", observed_lines)
            )
            if declared != observed_lines:
                raise ValueError(
                    f"{corpus_path}: JSON line_index={declared} "
                    f"but physical row={observed_lines}"
                )

            raw_tokens = payload.get("tokens")
            if not isinstance(raw_tokens, list):
                raise ValueError(
                    f"{corpus_path}:{observed_lines}: missing tokens list"
                )

            tokens: List[Token] = []
            for raw_token in raw_tokens:
                if (
                    not isinstance(raw_token, list)
                    or not raw_token
                    or not all(
                        isinstance(glyph, str) and glyph
                        for glyph in raw_token
                    )
                ):
                    raise ValueError(
                        f"{corpus_path}:{observed_lines}: invalid token"
                    )
                tokens.append(tuple(raw_token))

            if not tokens:
                raise ValueError(
                    f"{corpus_path}:{observed_lines}: empty unit"
                )

            item = (
                observed_lines,
                tuple(tokens),
            )

            if observed_lines in train_set:
                train.append(item)
            else:
                validation.append(item)

    if observed_lines != expected_line_count:
        raise ValueError(
            f"{corpus_path}: expected {expected_line_count} rows, "
            f"found {observed_lines}"
        )

    if len(train) != len(train_set):
        raise ValueError(
            f"{corpus_path}: missing train rows "
            f"({len(train)} != {len(train_set)})"
        )
    if len(validation) != len(validation_set):
        raise ValueError(
            f"{corpus_path}: missing validation rows "
            f"({len(validation)} != {len(validation_set)})"
        )

    return train, validation


def _unit_id(index: int) -> str:
    return f"u{index:06d}"


def build_hmm_examples(
    source_id: str,
    units: Sequence[Tuple[int, Tuple[Token, ...]]],
    *,
    view: str,
) -> List[BaselineSequence]:
    if view not in {"space_free", "token_aware"}:
        raise ValueError(f"Unknown HMM view: {view!r}")

    output = []

    for line_index, tokens in units:
        symbols: List[str] = []
        boundaries: List[bool] = []

        for token_index, token in enumerate(tokens):
            if view == "token_aware" and token_index > 0:
                symbols.append(WB)
                boundaries.append(True)

            symbols.extend(token)
            boundaries.extend([False] * len(token))

        output.append(
            BaselineSequence(
                leaf_group=_unit_id(line_index),
                folio=source_id,
                locus=_unit_id(line_index),
                symbols=tuple(symbols),
                is_boundary=tuple(boundaries),
            )
        )

    return output


def build_token_reset_examples(
    source_id: str,
    units: Sequence[Tuple[int, Tuple[Token, ...]]],
) -> List[BaselineSequence]:
    output = []

    for line_index, tokens in units:
        for token in tokens:
            output.append(
                BaselineSequence(
                    leaf_group=_unit_id(line_index),
                    folio=source_id,
                    locus=_unit_id(line_index),
                    symbols=tuple(token) + (EOT,),
                    is_boundary=(
                        (False,) * len(token)
                        + (True,)
                    ),
                )
            )

    return output


def build_slot_examples(
    source_id: str,
    units: Sequence[Tuple[int, Tuple[Token, ...]]],
) -> List[SlotTokenExample]:
    output = []

    for line_index, tokens in units:
        for token_index, token in enumerate(tokens):
            output.append(
                SlotTokenExample(
                    leaf_group=_unit_id(line_index),
                    folio=source_id,
                    locus=_unit_id(line_index),
                    token_index=token_index,
                    glyphs=tuple(token),
                )
            )

    return output


def build_token_markov_sequences(
    source_id: str,
    units: Sequence[Tuple[int, Tuple[Token, ...]]],
) -> List[List[TokenMarkovExample]]:
    output = []

    for line_index, tokens in units:
        sequence = [
            TokenMarkovExample(
                leaf_group=_unit_id(line_index),
                folio=source_id,
                locus=_unit_id(line_index),
                token_index=token_index,
                segment_index=0,
                glyphs=tuple(token),
            )
            for token_index, token in enumerate(tokens)
        ]
        if sequence:
            output.append(sequence)

    return output


def build_copy_sequences(
    source_id: str,
    units: Sequence[Tuple[int, Tuple[Token, ...]]],
) -> List[List[CopyTokenExample]]:
    output = []

    for line_index, tokens in units:
        sequence = [
            CopyTokenExample(
                leaf_group=_unit_id(line_index),
                folio=source_id,
                locus=_unit_id(line_index),
                token_index=token_index,
                segment_index=0,
                glyphs=tuple(token),
            )
            for token_index, token in enumerate(tokens)
        ]
        if sequence:
            output.append(sequence)

    return output


def fit_and_score_trigram(
    train_examples: Sequence[BaselineSequence],
    validation_examples: Sequence[BaselineSequence],
    *,
    view: str,
) -> Tuple[dict, List[dict]]:
    model = WittenBellNGram(max_order=3)
    model.fit(
        example.symbols
        for example in train_examples
        if example.symbols
    )

    aggregate, leaf_rows, _ = evaluate_model(
        model,
        validation_examples,
        orders=(3,),
        view=view,
    )

    if len(aggregate) != 1:
        raise RuntimeError("Expected exactly one trigram aggregate row")

    return aggregate[0], leaf_rows


def _standardize_unit_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, Tuple[float, int]]:
    result = {}

    for row in rows:
        unit = str(row["leaf_group"])
        bits = float(row["total_bits"])

        if "events" in row:
            events = int(row["events"])
        else:
            events = int(row["total_events"])

        if unit in result:
            raise ValueError(
                f"Duplicate per-unit row for {unit}"
            )
        if events <= 0:
            raise ValueError(
                f"Non-positive event count for {unit}"
            )

        result[unit] = (bits, events)

    return result


def paired_bootstrap(
    competitor_rows: Sequence[Mapping[str, object]],
    trigram_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict:
    competitor = _standardize_unit_rows(
        competitor_rows
    )
    trigram = _standardize_unit_rows(
        trigram_rows
    )

    if set(competitor) != set(trigram):
        raise ValueError(
            "Competitor/trigram validation units differ"
        )

    units = sorted(competitor)

    comp_bits = np.asarray(
        [competitor[u][0] for u in units],
        dtype=np.float64,
    )
    comp_events = np.asarray(
        [competitor[u][1] for u in units],
        dtype=np.float64,
    )
    tri_bits = np.asarray(
        [trigram[u][0] for u in units],
        dtype=np.float64,
    )
    tri_events = np.asarray(
        [trigram[u][1] for u in units],
        dtype=np.float64,
    )

    if not np.array_equal(
        comp_events,
        tri_events,
    ):
        raise ValueError(
            "Competitor and matched trigram event counts "
            "differ within at least one validation unit"
        )

    observed_comp = float(
        comp_bits.sum() / comp_events.sum()
    )
    observed_tri = float(
        tri_bits.sum() / tri_events.sum()
    )
    observed_delta = observed_comp - observed_tri

    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)

    # Chunked vectorization keeps memory bounded even for the German source.
    chunk = 250
    n = len(units)
    position = 0

    while position < replicates:
        take = min(chunk, replicates - position)
        indices = rng.integers(
            0,
            n,
            size=(take, n),
            endpoint=False,
        )

        comp_b = comp_bits[indices].sum(axis=1)
        events = comp_events[indices].sum(axis=1)
        tri_b = tri_bits[indices].sum(axis=1)

        deltas[position : position + take] = (
            comp_b / events
            - tri_b / events
        )
        position += take

    lower, upper = np.quantile(
        deltas,
        [0.025, 0.975],
    )

    return {
        "bootstrap_units": len(units),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "competitor_bits_per_event": observed_comp,
        "trigram_bits_per_event": observed_tri,
        "delta_competitor_minus_trigram_bits_per_event": (
            observed_delta
        ),
        "relative_nll_reduction_trigram_vs_competitor": (
            1.0 - tri_bits.sum() / comp_bits.sum()
        ),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_probability_trigram_better": float(
            np.mean(deltas > 0.0)
        ),
        "bootstrap_probability_competitor_better": float(
            np.mean(deltas < 0.0)
        ),
        "direction_match_voynich": bool(
            observed_delta > 0.0
        ),
        "strong_direction_match_voynich": bool(
            observed_delta > 0.0
            and lower > 0.0
        ),
    }


def evaluate_one_control(
    *,
    source_id: str,
    language: str,
    family: str,
    control_seed: int,
    train_units: Sequence[Tuple[int, Tuple[Token, ...]]],
    validation_units: Sequence[Tuple[int, Tuple[Token, ...]]],
    config: Mapping[str, object],
) -> Tuple[List[dict], dict]:
    """Fit/select all frozen model families and return five relations."""

    # ------------------------------------------------------------------
    # HMM relations: one space-free and one token-aware comparison.
    # ------------------------------------------------------------------
    relation_rows: List[dict] = []
    detailed: dict = {
        "hmm": {},
        "slot_grammar": {},
        "token_markov": {},
        "copy_edit": {},
    }

    hmm_cfg = config["hmm"]
    bootstrap_cfg = config["bootstrap"]

    for view_index, view in enumerate(
        hmm_cfg["views"]
    ):
        train_examples = build_hmm_examples(
            source_id,
            train_units,
            view=view,
        )
        validation_examples = build_hmm_examples(
            source_id,
            validation_units,
            view=view,
        )

        trigram_aggregate, trigram_leaves = (
            fit_and_score_trigram(
                train_examples,
                validation_examples,
                view=f"level2_{view}_trigram",
            )
        )

        candidate_rows = []
        candidate_leaves = {}

        for hidden_states in hmm_cfg[
            "hidden_states"
        ]:
            fitted = []

            for restart in range(
                int(hmm_cfg["restarts"])
            ):
                fit_seed = (
                    int(hmm_cfg["seed"])
                    + view_index * 1_000_000
                    + int(hidden_states) * 10_000
                    + restart
                )

                model = CategoricalHMM(
                    int(hidden_states),
                    pseudocount=float(
                        hmm_cfg["pseudocount"]
                    ),
                    max_iterations=int(
                        hmm_cfg["max_iterations"]
                    ),
                    min_iterations=int(
                        hmm_cfg["min_iterations"]
                    ),
                    tolerance_bits_per_event=float(
                        hmm_cfg[
                            "tolerance_bits_per_event"
                        ]
                    ),
                    batch_size=int(
                        hmm_cfg["batch_size"]
                    ),
                )
                fit = model.fit(
                    [
                        example.symbols
                        for example in train_examples
                    ],
                    seed=fit_seed,
                )
                fitted.append(
                    (fit, restart, fit_seed, model)
                )

            (
                best_fit,
                best_restart,
                best_seed,
                best_model,
            ) = max(
                fitted,
                key=lambda item: (
                    item[0].training_log_likelihood_nats,
                    -item[1],
                ),
            )

            aggregate, leaves, _ = evaluate_hmm(
                best_model,
                validation_examples,
                view=view,
                hidden_states=int(hidden_states),
                restart=best_restart,
                seed=best_seed,
            )
            candidate_rows.append(aggregate)
            candidate_leaves[int(hidden_states)] = leaves

        selected = min(
            candidate_rows,
            key=lambda row: (
                row["bits_per_event"],
                row["hidden_states"],
            ),
        )
        selected_k = int(
            selected["hidden_states"]
        )
        leaves = candidate_leaves[selected_k]

        relation = (
            "hmm_space_free"
            if view == "space_free"
            else "hmm_token_aware"
        )
        boot = paired_bootstrap(
            leaves,
            trigram_leaves,
            replicates=int(
                bootstrap_cfg["replicates"]
            ),
            seed=stable_seed(
                int(bootstrap_cfg["seed_base"]),
                source_id,
                family,
                control_seed,
                relation,
            ),
        )
        row = {
            "source_id": source_id,
            "language": language,
            "family": family,
            "control_seed": control_seed,
            "relation": relation,
            "competitor": "categorical_hmm",
            "selected_hyperparameter": selected_k,
            **boot,
        }
        relation_rows.append(row)
        detailed["hmm"][view] = {
            "trigram": trigram_aggregate,
            "candidates": candidate_rows,
            "selected_hidden_states": selected_k,
            "relation": row,
        }

    # ------------------------------------------------------------------
    # Shared matched token-reset trigram + EOT.
    # ------------------------------------------------------------------
    train_reset = build_token_reset_examples(
        source_id,
        train_units,
    )
    validation_reset = (
        build_token_reset_examples(
            source_id,
            validation_units,
        )
    )
    matched_trigram_aggregate, matched_trigram_leaves = (
        fit_and_score_trigram(
            train_reset,
            validation_reset,
            view="token_reset_eot",
        )
    )

    # ------------------------------------------------------------------
    # Slot grammar.
    # ------------------------------------------------------------------
    slot_cfg = config["slot_grammar"]
    train_slot = build_slot_examples(
        source_id,
        train_units,
    )
    validation_slot = build_slot_examples(
        source_id,
        validation_units,
    )

    max_train = max(
        len(example.glyphs)
        for example in train_slot
    )
    max_validation = max(
        len(example.glyphs)
        for example in validation_slot
    )
    max_support = int(
        slot_cfg["max_token_length"]
    )
    if max(max_train, max_validation) > max_support:
        raise ValueError(
            "Slot-grammar token length exceeds frozen "
            f"support {max_support}: train={max_train}, "
            f"validation={max_validation}"
        )

    slot_candidates = []
    slot_leaves = {}

    train_token_sequences = [
        example.glyphs
        for example in train_slot
    ]

    for templates in slot_cfg["templates"]:
        fitted = []

        for restart in range(
            int(slot_cfg["restarts"])
        ):
            fit_seed = (
                int(slot_cfg["seed"])
                + int(templates) * 10_000
                + restart
            )
            model = FiniteTemplateSlotGrammar(
                int(templates),
                slot_count=int(
                    slot_cfg["slot_count"]
                ),
                max_token_length=max_support,
                pseudocount=float(
                    slot_cfg["pseudocount"]
                ),
                max_iterations=int(
                    slot_cfg["max_iterations"]
                ),
                min_iterations=int(
                    slot_cfg["min_iterations"]
                ),
                tolerance_bits_per_token=float(
                    slot_cfg[
                        "tolerance_bits_per_token"
                    ]
                ),
            )
            fit = model.fit(
                train_token_sequences,
                seed=fit_seed,
            )
            fitted.append(
                (fit, restart, fit_seed, model)
            )

        (
            _best_fit,
            best_restart,
            best_seed,
            best_model,
        ) = max(
            fitted,
            key=lambda item: (
                item[0].training_log_likelihood_nats,
                -item[1],
            ),
        )

        aggregate, leaves, _ = (
            evaluate_slot_grammar(
                best_model,
                validation_slot,
                templates=int(templates),
                restart=best_restart,
                seed=best_seed,
            )
        )
        slot_candidates.append(aggregate)
        slot_leaves[int(templates)] = leaves

    selected_slot = min(
        slot_candidates,
        key=lambda row: (
            row["bits_per_event"],
            row["templates"],
        ),
    )
    selected_templates = int(
        selected_slot["templates"]
    )

    relation = "slot_grammar"
    boot = paired_bootstrap(
        slot_leaves[selected_templates],
        matched_trigram_leaves,
        replicates=int(
            bootstrap_cfg["replicates"]
        ),
        seed=stable_seed(
            int(bootstrap_cfg["seed_base"]),
            source_id,
            family,
            control_seed,
            relation,
        ),
    )
    slot_relation = {
        "source_id": source_id,
        "language": language,
        "family": family,
        "control_seed": control_seed,
        "relation": relation,
        "competitor": "finite_template_slot_grammar",
        "selected_hyperparameter": selected_templates,
        **boot,
    }
    relation_rows.append(slot_relation)
    detailed["slot_grammar"] = {
        "candidates": slot_candidates,
        "selected_templates": selected_templates,
        "relation": slot_relation,
    }

    # ------------------------------------------------------------------
    # Exact token Markov.
    # ------------------------------------------------------------------
    token_cfg = config["token_markov"]
    train_token_examples = (
        build_token_markov_sequences(
            source_id,
            train_units,
        )
    )
    validation_token_examples = (
        build_token_markov_sequences(
            source_id,
            validation_units,
        )
    )
    train_token_sequences = [
        [
            example.glyphs
            for example in sequence
        ]
        for sequence in train_token_examples
    ]

    token_model = TokenMarkovModel(
        max_context=int(
            token_cfg["max_context"]
        ),
        char_base_order=int(
            token_cfg["char_base_order"]
        ),
    ).fit(train_token_sequences)

    token_candidates = []
    token_leaves = {}

    for context_order in token_cfg[
        "context_orders"
    ]:
        aggregate, leaves, _ = (
            evaluate_token_markov(
                token_model,
                validation_token_examples,
                context_order=int(
                    context_order
                ),
            )
        )
        token_candidates.append(aggregate)
        token_leaves[int(context_order)] = leaves

    selected_token = min(
        token_candidates,
        key=lambda row: (
            row["bits_per_event"],
            row["context_order"],
        ),
    )
    selected_context = int(
        selected_token["context_order"]
    )

    # Verify the token model's internal character base is the same
    # distribution as the external matched comparator.
    internal_aggregate, _, _ = evaluate_model(
        token_model.char_base_model,
        validation_reset,
        orders=(3,),
        view="token_markov_internal_char_base",
    )
    internal_bits = float(
        internal_aggregate[0]["total_bits"]
    )
    external_bits = float(
        matched_trigram_aggregate["total_bits"]
    )
    if abs(internal_bits - external_bits) > 1e-8:
        raise ValueError(
            "Token-Markov internal character base does not "
            "match the shared token-reset trigram: "
            f"{internal_bits:.12f} vs {external_bits:.12f}"
        )

    relation = "token_markov"
    boot = paired_bootstrap(
        token_leaves[selected_context],
        matched_trigram_leaves,
        replicates=int(
            bootstrap_cfg["replicates"]
        ),
        seed=stable_seed(
            int(bootstrap_cfg["seed_base"]),
            source_id,
            family,
            control_seed,
            relation,
        ),
    )
    token_relation = {
        "source_id": source_id,
        "language": language,
        "family": family,
        "control_seed": control_seed,
        "relation": relation,
        "competitor": "exact_token_markov",
        "selected_hyperparameter": selected_context,
        **boot,
    }
    relation_rows.append(token_relation)
    detailed["token_markov"] = {
        "candidates": token_candidates,
        "selected_context_order": selected_context,
        "relation": token_relation,
    }

    # ------------------------------------------------------------------
    # Local copy-edit.
    # ------------------------------------------------------------------
    copy_cfg = config["copy_edit"]
    train_copy = build_copy_sequences(
        source_id,
        train_units,
    )
    validation_copy = build_copy_sequences(
        source_id,
        validation_units,
    )

    max_train_copy = max(
        len(example.glyphs)
        for sequence in train_copy
        for example in sequence
    )
    max_validation_copy = max(
        len(example.glyphs)
        for sequence in validation_copy
        for example in sequence
    )
    copy_support = int(
        copy_cfg["max_token_length"]
    )
    if max(
        max_train_copy,
        max_validation_copy,
    ) > copy_support:
        raise ValueError(
            "Copy-edit token length exceeds frozen "
            f"support {copy_support}: "
            f"train={max_train_copy}, "
            f"validation={max_validation_copy}"
        )

    copy_candidates = []
    copy_leaves = {}
    train_copy_tokens = [
        [
            example.glyphs
            for example in sequence
        ]
        for sequence in train_copy
    ]

    for source_window in copy_cfg[
        "source_windows"
    ]:
        model = LocalCopyEditModel(
            int(source_window),
            max_token_length=copy_support,
            pseudocount=float(
                copy_cfg["pseudocount"]
            ),
            rho_max_iterations=int(
                copy_cfg["rho_max_iterations"]
            ),
            rho_tolerance=float(
                copy_cfg["rho_tolerance"]
            ),
        )
        fit = model.fit(
            train_copy_tokens
        )
        aggregate, leaves, _ = evaluate_copy_edit(
            model,
            validation_copy,
        )
        aggregate = dict(aggregate)
        aggregate.update(
            {
                "p_insert": fit.p_insert,
                "p_delete": fit.p_delete,
                "p_copy": fit.p_copy,
                "rho_copy": fit.rho_copy,
                "rho_iterations": fit.rho_iterations,
                "rho_converged": fit.rho_converged,
            }
        )
        copy_candidates.append(aggregate)
        copy_leaves[int(source_window)] = leaves

    selected_copy = min(
        copy_candidates,
        key=lambda row: (
            row["bits_per_event"],
            row["source_window"],
        ),
    )
    selected_window = int(
        selected_copy["source_window"]
    )

    relation = "copy_edit"
    boot = paired_bootstrap(
        copy_leaves[selected_window],
        matched_trigram_leaves,
        replicates=int(
            bootstrap_cfg["replicates"]
        ),
        seed=stable_seed(
            int(bootstrap_cfg["seed_base"]),
            source_id,
            family,
            control_seed,
            relation,
        ),
    )
    copy_relation = {
        "source_id": source_id,
        "language": language,
        "family": family,
        "control_seed": control_seed,
        "relation": relation,
        "competitor": "local_copy_edit",
        "selected_hyperparameter": selected_window,
        **boot,
    }
    relation_rows.append(copy_relation)
    detailed["copy_edit"] = {
        "candidates": copy_candidates,
        "selected_source_window": selected_window,
        "relation": copy_relation,
    }

    expected_relations = set(
        config["operational_relations"]
    )
    observed_relations = {
        row["relation"]
        for row in relation_rows
    }
    if observed_relations != expected_relations:
        raise RuntimeError(
            "Operational relation set mismatch: "
            f"{observed_relations} != {expected_relations}"
        )

    detailed["matched_token_reset_trigram"] = (
        matched_trigram_aggregate
    )
    detailed["full_run_direction_match"] = all(
        row["direction_match_voynich"]
        for row in relation_rows
    )
    detailed["full_run_strong_direction_match"] = all(
        row["strong_direction_match_voynich"]
        for row in relation_rows
    )

    return relation_rows, detailed


def summarize_level2(
    relation_rows: Sequence[Mapping[str, object]],
    *,
    families: Sequence[str],
    relations: Sequence[str],
    languages: Sequence[str],
) -> Tuple[List[dict], List[dict]]:
    """Apply the frozen 4/5-key and 3/4-language replication rules."""

    by_group: Dict[
        Tuple[str, str, str],
        List[Mapping[str, object]],
    ] = defaultdict(list)

    for row in relation_rows:
        by_group[
            (
                str(row["family"]),
                str(row["relation"]),
                str(row["language"]),
            )
        ].append(row)

    language_rows = []

    for family in families:
        for relation in relations:
            for language in languages:
                group = by_group[
                    (family, relation, language)
                ]
                if len(group) != 5:
                    raise ValueError(
                        f"{family}/{relation}/{language}: "
                        f"expected 5 keys; found {len(group)}"
                    )

                positive = sum(
                    bool(
                        row[
                            "direction_match_voynich"
                        ]
                    )
                    for row in group
                )
                strong = sum(
                    bool(
                        row[
                            "strong_direction_match_voynich"
                        ]
                    )
                    for row in group
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
                        "family": family,
                        "relation": relation,
                        "language": language,
                        "keys": 5,
                        "positive_direction_keys": positive,
                        "strong_direction_keys": strong,
                        "mean_delta_bits_per_event": mean(
                            deltas
                        ),
                        "language_relation_replicated": (
                            strong >= 4
                        ),
                    }
                )

    family_rows = []

    for family in families:
        robust_relations = 0

        for relation in relations:
            group = [
                row
                for row in language_rows
                if row["family"] == family
                and row["relation"] == relation
            ]
            languages_replicated = sum(
                bool(
                    row[
                        "language_relation_replicated"
                    ]
                )
                for row in group
            )
            robust = languages_replicated >= 3
            robust_relations += int(robust)

            family_rows.append(
                {
                    "family": family,
                    "relation": relation,
                    "languages_replicated_4": (
                        languages_replicated
                    ),
                    "family_relation_robust": robust,
                }
            )

        family_rows.append(
            {
                "family": family,
                "relation": "__FULL_HIERARCHY__",
                "languages_replicated_4": "",
                "family_relation_robust": "",
                "robust_relations_5": robust_relations,
                "family_full_hierarchy_reproduction": (
                    robust_relations == len(relations)
                ),
            }
        )

    return language_rows, family_rows
