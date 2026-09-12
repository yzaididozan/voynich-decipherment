from __future__ import annotations

import json
from pathlib import Path

from scripts.fetch_freeze_language_controls import (
    LinkCollector,
    PageRecord,
    TextExtractor,
    append_data_manifest,
    crop_lines,
    normalize_line,
    ordered_child_links,
)


def test_normalize_line_keeps_unicode_letters_and_drops_digits():
    assert normalize_line(
        "Ārma, VIRUMQUE! 123 naïve."
    ) == (
        "ārma",
        "virumque",
        "naïve",
    )


def test_text_extractor_keeps_superscript_letters_but_skips_references():
    parser = TextExtractor()
    parser.feed(
        '<table><tr><td>HEADER SHOULD GO</td></tr></table>'
        '<p>gu<sup>o</sup>ten '
        '<sup class="reference">[1]</sup> rat</p>'
    )
    text = parser.text()

    assert "HEADER" not in text
    assert "guoten" in text.replace(" ", "")
    assert "[1]" not in text
    assert "rat" in text


def test_crop_lines_is_exact_and_preserves_line_order():
    lines = [
        "one two three",
        "four five six",
        "seven eight",
    ]
    cropped, count = crop_lines(
        lines,
        target_tokens=5,
    )

    assert count == 5
    assert cropped == [
        "one two three",
        "four five",
    ]


def test_ordered_child_links_filters_and_preserves_order():
    html = """
    <p>
      <a href="/wiki/Root/First">First</a>
      <a href="/wiki/Other/Page">Other</a>
      <a href="/wiki/Root/Second">Second</a>
      <a href="/wiki/Root/First">Duplicate</a>
      <a href="/wiki/Root/Second/Deep">Deep</a>
    </p>
    """
    page = PageRecord(
        title="Root",
        pageid=1,
        revid=2,
        html=html,
    )

    links = ordered_child_links(
        page,
        root_page="Root",
        direct_only=True,
        exclude_pattern=None,
    )

    assert links == [
        "Root/First",
        "Root/Second",
    ]


def test_append_data_manifest_adds_versioned_control_section(tmp_path):
    data_manifest = tmp_path / "manifest.yaml"
    data_manifest.write_text(
        'manifest_version: "1.1"\n',
        encoding="utf-8",
    )

    manifests = [
        {
            "source_id": "latin_x",
            "language": "Latin",
            "source_path": "data/controls/languages/latin_x/source.txt",
            "landing_url": "https://example.test/latin",
            "retrieved_utc": "2026-09-12T12:00:00+00:00",
            "source_pages_path": "data/controls/languages/latin_x/source_pages.json",
            "source_sha256": "a" * 64,
            "rights_note": "public domain test",
        }
    ]

    append_data_manifest(
        data_manifest,
        manifests,
        master_manifest_path=Path(
            "data/controls/languages/language_controls_v1_manifest.json"
        ),
    )

    text = data_manifest.read_text(encoding="utf-8")
    assert "historical_language_controls:" in text
    assert 'freeze_id: "language-controls-v1"' in text
    assert "latin_x" in text

    # Re-running must not duplicate the frozen section.
    append_data_manifest(
        data_manifest,
        manifests,
        master_manifest_path=Path(
            "data/controls/languages/language_controls_v1_manifest.json"
        ),
    )
    assert (
        data_manifest.read_text(encoding="utf-8").count(
            "historical_language_controls:"
        )
        == 1
    )
