from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.review_image_validation import ReviewStore


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "v1"

    page_fields = [
        "sample_index",
        "folio",
        "review_status",
        "page_alignment",
        "overall_transcription_agreement",
        "reviewer_notes",
    ]
    pages = [
        {
            "sample_index": "1",
            "folio": "f41r",
            "review_status": "",
            "page_alignment": "",
            "overall_transcription_agreement": "",
            "reviewer_notes": "",
        }
    ]
    write_csv(root / "pages.csv", page_fields, pages)

    locus_fields = [
        "sample_index",
        "folio",
        "locus_check_index",
        "locus",
        "source_line_number",
        "text_normalized",
        "uncertain_spaces",
        "alternative_readings",
        "unknown_question_marks",
        "drawing_interruptions",
        "check_status",
        "glyph_alignment",
        "spacing_alignment",
        "locus_alignment",
        "notes",
    ]
    loci = [
        {
            "sample_index": "1",
            "folio": "f41r",
            "locus_check_index": "1",
            "locus": "f41r.1,+P0",
            "source_line_number": "10",
            "text_normalized": "o.a",
            "uncertain_spaces": "0",
            "alternative_readings": "0",
            "unknown_question_marks": "0",
            "drawing_interruptions": "0",
            "check_status": "",
            "glyph_alignment": "",
            "spacing_alignment": "",
            "locus_alignment": "",
            "notes": "",
        },
        {
            "sample_index": "1",
            "folio": "f41r",
            "locus_check_index": "2",
            "locus": "f41r.2,+P0",
            "source_line_number": "11",
            "text_normalized": "a.o",
            "uncertain_spaces": "0",
            "alternative_readings": "0",
            "unknown_question_marks": "0",
            "drawing_interruptions": "0",
            "check_status": "",
            "glyph_alignment": "",
            "spacing_alignment": "",
            "locus_alignment": "",
            "notes": "",
        },
    ]
    write_csv(root / "loci.csv", locus_fields, loci)

    map_fields = [
        "folio",
        "variant_index",
        "fetch_status",
        "canvas_label",
        "canvas_id",
        "image_url",
        "local_image",
    ]
    image_rows = [
        {
            "folio": "f41r",
            "variant_index": "1",
            "fetch_status": "DOWNLOADED",
            "canvas_label": "41r",
            "canvas_id": "https://example.test/canvas",
            "image_url": "https://example.test/image.jpg",
            "local_image": "images/f41r.jpg",
        }
    ]
    write_csv(
        root / "yale_images" / "yale_image_map.csv",
        map_fields,
        image_rows,
    )
    (root / "yale_images" / "images").mkdir(parents=True, exist_ok=True)
    (root / "yale_images" / "images" / "f41r.jpg").write_bytes(
        b"\xff\xd8" + b"x" * 100 + b"\xff\xd9"
    )

    return root


def test_save_locus_and_backup(root: Path):
    store = ReviewStore(root)
    store.save_locus(
        0,
        {
            "check_status": "PASS",
            "glyph_alignment": "YES",
            "spacing_alignment": "YES",
            "locus_alignment": "YES",
            "notes": "looks aligned",
        },
    )

    assert store.loci[0]["check_status"] == "PASS"
    assert (root / "loci.csv.bak").is_file()
    assert (root / "pages.csv.bak").is_file()

    _, rows = __import__(
        "scripts.review_image_validation",
        fromlist=["read_csv"],
    ).read_csv(root / "loci.csv")
    assert rows[0]["notes"] == "looks aligned"


def test_progress(root: Path):
    store = ReviewStore(root)
    assert store.progress() == {
        "completed_loci": 0,
        "total_loci": 2,
        "completed_pages": 0,
        "total_pages": 1,
    }
    store.save_locus(
        0,
        {
            "check_status": "PASS",
            "glyph_alignment": "YES",
            "spacing_alignment": "YES",
            "locus_alignment": "YES",
            "notes": "",
        },
    )
    assert store.progress()["completed_loci"] == 1


def test_suggested_page_status(root: Path):
    store = ReviewStore(root)

    for i in (0, 1):
        store.save_locus(
            i,
            {
                "check_status": "PASS",
                "glyph_alignment": "YES",
                "spacing_alignment": "YES",
                "locus_alignment": "YES",
                "notes": "",
            },
        )
    assert store.suggested_page_status("f41r") == "PASS"

    store.save_locus(
        1,
        {
            "check_status": "TRANSCRIPTION_DISAGREEMENT",
            "glyph_alignment": "UNCERTAIN",
            "spacing_alignment": "NO",
            "locus_alignment": "YES",
            "notes": "",
        },
    )
    assert (
        store.suggested_page_status("f41r")
        == "TRANSCRIPTION_DISAGREEMENT"
    )


def test_save_page(root: Path):
    store = ReviewStore(root)
    store.save_page(
        "f41r",
        {
            "review_status": "PASS",
            "page_alignment": "YES",
            "overall_transcription_agreement": "YES",
            "reviewer_notes": "reviewed",
        },
    )
    page = store.page_for("f41r")
    assert page is not None
    assert page["review_status"] == "PASS"
    assert page["reviewer_notes"] == "reviewed"


def test_rejects_invalid_status(root: Path):
    store = ReviewStore(root)
    with pytest.raises(ValueError):
        store.save_locus(
            0,
            {
                "check_status": "MAYBE",
                "glyph_alignment": "",
                "spacing_alignment": "",
                "locus_alignment": "",
                "notes": "",
            },
        )
