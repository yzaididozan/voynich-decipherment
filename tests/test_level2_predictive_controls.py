from __future__ import annotations

import math

from src.evaluation.level2_predictive_controls import (
    build_hmm_examples,
    build_token_reset_examples,
    deterministic_unit_split,
    evaluate_one_control,
    largest_remainder_counts,
    paired_bootstrap,
    summarize_level2,
)


def test_largest_remainder_matches_frozen_72_15_15_ratio():
    assert largest_remainder_counts(
        725,
        {
            "train": 72,
            "validation": 15,
            "reserved_test": 15,
        },
    ) == {
        "train": 512,
        "validation": 107,
        "reserved_test": 106,
    }


def test_deterministic_split_is_disjoint_complete_and_reproducible():
    kwargs = dict(
        source_id="latin_udante_ud218",
        line_count=725,
        seed=40814041438,
        weights={
            "train": 72,
            "validation": 15,
            "reserved_test": 15,
        },
    )
    first = deterministic_unit_split(**kwargs)
    second = deterministic_unit_split(**kwargs)

    assert first == second

    train = set(first["train"])
    val = set(first["validation"])
    test = set(first["reserved_test"])

    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(range(1, 726))


def test_generic_hmm_views_and_token_reset_eventization():
    units = [
        (
            7,
            (
                ("A", "B"),
                ("C",),
            ),
        )
    ]

    space = build_hmm_examples(
        "source",
        units,
        view="space_free",
    )
    aware = build_hmm_examples(
        "source",
        units,
        view="token_aware",
    )
    reset = build_token_reset_examples(
        "source",
        units,
    )

    assert space[0].symbols == ("A", "B", "C")
    assert aware[0].symbols == ("A", "B", "<WB>", "C")
    assert sum(aware[0].is_boundary) == 1

    assert len(reset) == 2
    assert reset[0].symbols == ("A", "B", "<EOT>")
    assert reset[1].symbols == ("C", "<EOT>")


def test_paired_bootstrap_positive_when_competitor_is_uniformly_worse():
    trigram = [
        {
            "leaf_group": f"u{i}",
            "total_bits": 10.0 + i,
            "total_events": 10,
        }
        for i in range(1, 8)
    ]
    competitor = [
        {
            "leaf_group": f"u{i}",
            "total_bits": 15.0 + i,
            "total_events": 10,
        }
        for i in range(1, 8)
    ]

    result = paired_bootstrap(
        competitor,
        trigram,
        replicates=500,
        seed=123,
    )

    assert result[
        "delta_competitor_minus_trigram_bits_per_event"
    ] > 0
    assert result["ci_95_lower"] > 0
    assert result["strong_direction_match_voynich"] is True


def test_summary_applies_4_of_5_keys_and_3_of_4_languages():
    families = ["nomenclator"]
    relations = [
        "hmm_space_free",
        "hmm_token_aware",
        "slot_grammar",
        "token_markov",
        "copy_edit",
    ]
    languages = ["Latin", "Italian", "German", "French"]

    rows = []
    for relation in relations:
        for language in languages:
            for seed in range(5):
                # Four strong keys in three languages, only three in French.
                strong = (
                    seed < 4
                    if language != "French"
                    else seed < 3
                )
                rows.append(
                    {
                        "family": "nomenclator",
                        "relation": relation,
                        "language": language,
                        "control_seed": seed,
                        "direction_match_voynich": strong,
                        "strong_direction_match_voynich": strong,
                        "delta_competitor_minus_trigram_bits_per_event": (
                            0.1 if strong else -0.1
                        ),
                    }
                )

    language_rows, family_rows = summarize_level2(
        rows,
        families=families,
        relations=relations,
        languages=languages,
    )

    assert len(language_rows) == 20

    full = [
        row
        for row in family_rows
        if row["relation"] == "__FULL_HIERARCHY__"
    ][0]
    assert full["robust_relations_5"] == 5
    assert full[
        "family_full_hierarchy_reproduction"
    ] is True


def test_tiny_end_to_end_uses_all_five_operational_relations():
    train_units = []
    validation_units = []

    patterns = [
        (("A", "B"), ("A", "C"), ("A", "B")),
        (("B", "A"), ("B", "C"), ("B", "A")),
    ]

    for i in range(1, 11):
        train_units.append(
            (i, patterns[i % 2])
        )

    for i in range(20, 25):
        validation_units.append(
            (i, patterns[i % 2])
        )

    cfg = {
        "operational_relations": [
            "hmm_space_free",
            "hmm_token_aware",
            "slot_grammar",
            "token_markov",
            "copy_edit",
        ],
        "hmm": {
            "hidden_states": [2],
            "views": ["space_free", "token_aware"],
            "restarts": 1,
            "seed": 7,
            "pseudocount": 0.5,
            "max_iterations": 3,
            "min_iterations": 2,
            "tolerance_bits_per_event": 1e-4,
            "batch_size": 64,
        },
        "slot_grammar": {
            "templates": [1],
            "slot_count": 8,
            "max_token_length": 64,
            "restarts": 1,
            "seed": 7,
            "pseudocount": 0.5,
            "max_iterations": 3,
            "min_iterations": 2,
            "tolerance_bits_per_token": 1e-4,
        },
        "token_markov": {
            "context_orders": [1],
            "max_context": 2,
            "char_base_order": 3,
        },
        "copy_edit": {
            "source_windows": [1],
            "max_token_length": 64,
            "pseudocount": 0.5,
            "rho_max_iterations": 10,
            "rho_tolerance": 1e-8,
        },
        "bootstrap": {
            "replicates": 50,
            "seed_base": 7,
        },
    }

    rows, detail = evaluate_one_control(
        source_id="synthetic",
        language="Synthetic",
        family="monoalphabetic",
        control_seed=1,
        train_units=train_units,
        validation_units=validation_units,
        config=cfg,
    )

    assert {
        row["relation"]
        for row in rows
    } == set(cfg["operational_relations"])

    assert all(
        math.isfinite(
            row[
                "delta_competitor_minus_trigram_bits_per_event"
            ]
        )
        for row in rows
    )
    assert isinstance(
        detail["full_run_direction_match"],
        bool,
    )

    # Per-run summaries are written directly to JSON by the production runner.
    import json
    json.dumps(detail, sort_keys=True)
