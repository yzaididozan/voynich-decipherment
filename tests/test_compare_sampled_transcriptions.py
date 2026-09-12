from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.compare_sampled_transcriptions import (
    ComparableText,
    classify_comparison,
    disagreement_features,
    levenshtein_distance,
    normalized_similarity,
    select_recommended_review,
)


def ct(glyphs, token_lengths=None, uncertainty=0):
    glyphs = tuple(glyphs)
    if token_lengths is None:
        tokens = (glyphs,)
    else:
        tokens = []
        pos = 0
        for length in token_lengths:
            tokens.append(tuple(glyphs[pos:pos + length]))
            pos += length
        assert pos == len(glyphs)
        tokens = tuple(tokens)
    return ComparableText(
        glyphs=glyphs,
        tokens=tokens,
        glyph_count=len(glyphs),
        token_count=len(tokens),
        unknown_glyphs=uncertainty,
        uncertain_spaces=0,
        alternatives=0,
        canonical_glyph_text=" ".join(glyphs),
        canonical_token_text=".".join(
            " ".join(token) for token in tokens
        ),
    )


def test_levenshtein_and_similarity():
    assert levenshtein_distance(("A", "B"), ("A", "B")) == 0
    assert levenshtein_distance(("A", "B"), ("A", "C")) == 1
    assert normalized_similarity(("A", "B"), ("A", "C")) == pytest.approx(0.5)


def test_all_three_exact():
    comparable = {
        "ZL3b": ct(("A1", "B1"), [1, 1]),
        "GC2a": ct(("A1", "B1"), [1, 1]),
        "IT2a": ct(("A1", "B1"), [1, 1]),
    }
    features = disagreement_features(comparable)
    assert features["all_glyph_exact"] is True
    assert features["all_token_exact"] is True
    assert features["primary_exact_supporters"] == ["GC2a", "IT2a"]


def test_boundary_only_disagreement_is_mandatory():
    comparable = {
        "ZL3b": ct(("A1", "B1", "C1"), [1, 2]),
        "GC2a": ct(("A1", "B1", "C1"), [2, 1]),
        "IT2a": ct(("A1", "B1", "C1"), [1, 1, 1]),
    }
    features = disagreement_features(comparable)
    category, mandatory, _, _ = classify_comparison(
        features,
        comparable,
        low_threshold=0.85,
        high_threshold=0.95,
    )
    assert category == "BOUNDARY_ONLY_DISAGREEMENT"
    assert mandatory is True


def test_gc_it_consensus_against_zl_is_mandatory():
    comparable = {
        "ZL3b": ct(("A1", "X1"), [2]),
        "GC2a": ct(("A1", "B1"), [2]),
        "IT2a": ct(("A1", "B1"), [2]),
    }
    features = disagreement_features(comparable)
    assert features["alternate_token_consensus_against_primary"] is True
    category, mandatory, score, _ = classify_comparison(
        features,
        comparable,
        low_threshold=0.40,
        high_threshold=0.95,
    )
    assert category == "GC_IT_CONSENSUS_AGAINST_ZL"
    assert mandatory is True
    assert score >= 1000


def test_primary_supported_by_one():
    comparable = {
        "ZL3b": ct(("A1", "B1", "C1"), [3]),
        "GC2a": ct(("A1", "B1", "C1"), [3]),
        "IT2a": ct(("A1", "B1", "D1"), [3]),
    }
    features = disagreement_features(comparable)
    category, mandatory, _, _ = classify_comparison(
        features,
        comparable,
        low_threshold=0.50,
        high_threshold=0.60,
    )
    assert category == "PRIMARY_SUPPORTED_BY_ONE_TRANSCRIPTION"
    assert mandatory is False


def row(
    order,
    folio,
    locus,
    *,
    score,
    mandatory=False,
    exact=False,
):
    return {
        "sample_order": order,
        "sample_index": str(order),
        "locus_check_index": "1",
        "folio": folio,
        "locus": locus,
        "comparison_class": (
            "ALL_THREE_EXACT" if exact else "MODERATE_GLYPH_AGREEMENT"
        ),
        "mandatory_manual_review": mandatory,
        "priority_score": score,
        "review_reason": "x",
        "primary_support": "x",
        "min_pairwise_glyph_similarity": 1.0 if exact else 0.9,
        "mean_pairwise_glyph_similarity": 1.0 if exact else 0.9,
        "all_three_glyph_exact": exact,
        "all_three_token_exact": exact,
        "gc_it_consensus_against_zl": False,
        "ZL3b_native": "z",
        "GC2a_native": "g",
        "IT2a_native": "i",
    }


def test_review_selection_keeps_mandatory_and_adds_controls():
    rows = [
        row(1, "f1r", "L1", score=900, mandatory=True),
        row(2, "f1r", "L2", score=800),
        row(3, "f2r", "L3", score=700),
        row(4, "f3r", "L4", score=600),
        row(5, "f4r", "C1", score=0, exact=True),
        row(6, "f5r", "C2", score=0, exact=True),
    ]
    config = {
        "target_disagreement_reviews": 3,
        "consensus_control_reviews": 2,
        "max_initial_reviews_per_folio": 1,
    }
    selected = select_recommended_review(rows, config=config)

    roles = [item["review_role"] for item in selected]
    assert roles.count("MANDATORY_DISAGREEMENT") == 1
    assert roles.count("TARGETED_DISAGREEMENT") == 2
    assert roles.count("CONSENSUS_CONTROL") == 2
    assert selected[0]["locus"] == "L1"


def test_mandatory_cases_can_exceed_target():
    rows = [
        row(i, f"f{i}r", f"L{i}", score=1000 - i, mandatory=True)
        for i in range(1, 5)
    ]
    config = {
        "target_disagreement_reviews": 2,
        "consensus_control_reviews": 0,
        "max_initial_reviews_per_folio": 1,
    }
    selected = select_recommended_review(rows, config=config)
    assert len(selected) == 4
    assert all(
        item["review_role"] == "MANDATORY_DISAGREEMENT"
        for item in selected
    )
