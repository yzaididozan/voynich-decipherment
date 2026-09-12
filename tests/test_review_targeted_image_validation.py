from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.review_targeted_image_validation import (
    TargetedReviewStore,
)


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "v1"

    shortlist_fields = [
        "review_rank",
        "review_role",
        "folio",
        "locus",
        "comparison_class",
        "review_reason",
        "primary_support",
        "min_pairwise_glyph_similarity",
        "ZL3b_native",
        "GC2a_native",
        "IT2a_native",
    ]

    rows = []
    for i in range(1, 11):
        if i <= 5:
            role = "MANDATORY_DISAGREEMENT"
            cls = "BOUNDARY_ONLY_DISAGREEMENT"
        elif i <= 8:
            role = "TARGETED_DISAGREEMENT"
            cls = "LOW_GLYPH_AGREEMENT"
        else:
            role = "CONSENSUS_CONTROL"
            cls = "ALL_THREE_EXACT"

        rows.append(
            {
                "review_rank": str(i),
                "review_role": role,
                "folio": f"f{i}r",
                "locus": f"f{i}r.1,+P0",
                "comparison_class": cls,
                "review_reason": "test",
                "primary_support": "x",
                "min_pairwise_glyph_similarity": "0.9",
                "ZL3b_native": "o.a",
                "GC2a_native": "o.a",
                "IT2a_native": "o.a",
            }
        )

    write_csv(
        root / "cross_transcription" / "recommended_manual_review.csv",
        shortlist_fields,
        rows,
    )

    map_fields = [
        "folio",
        "variant_index",
        "fetch_status",
        "canvas_label",
        "canvas_id",
        "image_url",
        "local_image",
    ]
    image_rows = []
    for i in range(1, 11):
        image_rows.append(
            {
                "folio": f"f{i}r",
                "variant_index": "1",
                "fetch_status": "DOWNLOADED",
                "canvas_label": f"{i}r",
                "canvas_id": "https://example.test/canvas",
                "image_url": "https://example.test/image.jpg",
                "local_image": f"images/f{i}r.jpg",
            }
        )
        image = root / "yale_images" / "images" / f"f{i}r.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\xff\xd8" + b"x" * 20 + b"\xff\xd9")

    write_csv(
        root / "yale_images" / "yale_image_map.csv",
        map_fields,
        image_rows,
    )
    return root


def test_initializes_exactly_ten_cases(root: Path):
    store = TargetedReviewStore(root)
    assert len(store.rows) == 10
    assert store.progress() == {"completed": 0, "total": 10}
    assert store.output_path.is_file()


def test_boundary_case_defaults(root: Path):
    store = TargetedReviewStore(root)
    row = store.rows[0]
    assert row["boundary_assessment"] == ""
    assert row["glyph_disagreement_assessment"] == "NOT_APPLICABLE"


def test_targeted_glyph_case_defaults(root: Path):
    store = TargetedReviewStore(root)
    row = store.rows[5]
    assert row["boundary_assessment"] == "NOT_APPLICABLE"
    assert row["glyph_disagreement_assessment"] == ""


def test_control_defaults(root: Path):
    store = TargetedReviewStore(root)
    row = store.rows[8]
    assert row["boundary_assessment"] == "NOT_APPLICABLE"
    assert row["glyph_disagreement_assessment"] == "NOT_APPLICABLE"


def test_save_boundary_case(root: Path):
    store = TargetedReviewStore(root)
    store.save(
        0,
        {
            "review_status": "NO_OBVIOUS_QC_PROBLEM",
            "locus_alignment": "YES",
            "boundary_assessment": "AMBIGUOUS",
            "glyph_disagreement_assessment": "NOT_APPLICABLE",
            "reviewer_notes": "spacing is not visually decisive",
        },
    )
    assert store.progress()["completed"] == 1
    assert store.rows[0]["boundary_assessment"] == "AMBIGUOUS"
    assert store.output_path.with_suffix(".csv.bak").is_file()


def test_save_targeted_disagreement(root: Path):
    store = TargetedReviewStore(root)
    store.save(
        5,
        {
            "review_status": "NO_OBVIOUS_QC_PROBLEM",
            "locus_alignment": "YES",
            "boundary_assessment": "NOT_APPLICABLE",
            "glyph_disagreement_assessment": (
                "PLAUSIBLE_TRANSCRIPTION_VARIATION"
            ),
            "reviewer_notes": "",
        },
    )
    assert (
        store.rows[5]["glyph_disagreement_assessment"]
        == "PLAUSIBLE_TRANSCRIPTION_VARIATION"
    )


def test_rejects_wrong_field_for_boundary_case(root: Path):
    store = TargetedReviewStore(root)
    with pytest.raises(ValueError):
        store.save(
            0,
            {
                "review_status": "NO_OBVIOUS_QC_PROBLEM",
                "locus_alignment": "YES",
                "boundary_assessment": "AMBIGUOUS",
                "glyph_disagreement_assessment": (
                    "PLAUSIBLE_TRANSCRIPTION_VARIATION"
                ),
                "reviewer_notes": "",
            },
        )


def test_existing_output_must_match_frozen_shortlist(root: Path):
    store = TargetedReviewStore(root)
    store.rows[0]["locus"] = "wrong"
    fields = store.fields
    write_csv(store.output_path, fields, store.rows)
    with pytest.raises(ValueError):
        TargetedReviewStore(root)


def test_images_for_folio(root: Path):
    store = TargetedReviewStore(root)
    rows = store.images_for("f1r")
    assert len(rows) == 1
    assert rows[0]["local_image"] == "images/f1r.jpg"
