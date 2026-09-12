from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path("scripts/fetch_open_language_controls.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_open_language_controls",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_form():
    mod = load_module()
    assert mod.normalize_form("D'Amore") == ("d", "amore")
    assert mod.normalize_form("Über-mære") == ("über", "mære")
    assert mod.normalize_form("123...") == ()


def test_conllu_parser_skips_multiword_rows_and_punctuation(tmp_path):
    mod = load_module()
    path = tmp_path / "sample.conllu"
    path.write_text(
        "# sent_id = 1\n"
        "1\tArma\tarma\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "2-3\tdel\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "2\tde\tde\tADP\t_\t_\t1\tcase\t_\t_\n"
        "3\tlo\tlo\tDET\t_\t_\t1\tdet\t_\t_\n"
        "4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n"
        "\n",
        encoding="utf-8",
    )

    assert list(mod.iter_conllu_sentences(path)) == [
        ("arma", "de", "lo")
    ]


def test_rem_tei_parser_uses_word_and_line_elements():
    mod = load_module()
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><body><p>
        <l><w>Ich</w><w>bin</w><w>hie</w></l>
        <l><w>guot</w><w>maere</w></l>
      </p></body></text>
    </TEI>
    """

    assert list(mod.iter_rem_tei_lines(xml)) == [
        ("ich", "bin", "hie"),
        ("guot", "maere"),
    ]


def test_crop_lines_is_exact():
    mod = load_module()
    lines, count = mod.crop_lines(
        [
            ("one", "two", "three"),
            ("four", "five", "six"),
        ],
        5,
    )
    assert count == 5
    assert lines == [
        ("one", "two", "three"),
        ("four", "five"),
    ]


def test_registry_is_frozen_to_expected_sources():
    mod = load_module()
    assert mod.TARGET_TOKENS == 20_000
    assert mod.UD_RELEASE == "2.18"
    assert mod.UD_TAG == "r2.18"
    assert [s["source_id"] for s in mod.SOURCES] == [
        "latin_udante_ud218",
        "italian_old_ud218",
        "german_rem_v21",
        "french_profiterole_ud218",
    ]
    assert len(mod.registry_sha256()) == 64
