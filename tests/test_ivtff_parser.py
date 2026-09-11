from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ivtff import (
    IVTFFParseError,
    load_ivtff,
    normalize_text_lossless,
    parse_file_header,
    parse_ivtff_lines,
)


ZL_SAMPLE = """#=IVTFF Eva- 2.0 M 5
# ZL transliteration file, updated from EVMT project
# Version 3b of 13/05/2025
<f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
<f1r.1,@P0>       <%>fachys.ykal.ar.ataiin
<f1r.2,+P0>       sory.ckhar.or,y<$>
<f1r.3,@L0>       daiin
"""

GC_SAMPLE = """#=IVTFF v101 2.0 M 6
# Original file: voyn101.txt
# IVTFF version 2a of 05/02/2023 modified 25/06/2025
<f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
<f1r.1,@P0>       <%>fa19s.9,hae.ay.Akam
"""

IT_SAMPLE = """#=IVTFF EvaT 2.0 M 3
# Extracted from LSI_ivtff_0d.txt
# Version 2a of 02/02/2023 modified 25/06/2025
<f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
<f1r.1,@P0>       <%>fachys.ykal.ar.ataiin
"""

RF_SAMPLE = """#=IVTFF Eva- 2.0 D 9
<f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
<f1r.1,@P0>       fachys.ykal.ar.@221;taiin.shol.shory.{cto}ses
"""


def parse_text(sample: str, transcription_id: str):
    return list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id=transcription_id,
        )
    )


def test_parse_file_header_zl():
    header = parse_file_header("#=IVTFF Eva- 2.0 M 5\n")
    assert header.alphabet == "Eva-"
    assert header.version == "2.0"
    assert header.options == "M 5"
    assert header.original == "#=IVTFF Eva- 2.0 M 5"


def test_parse_file_header_gc():
    header = parse_file_header("#=IVTFF v101 2.0 M 6\n")
    assert header.alphabet == "v101"
    assert header.version == "2.0"
    assert header.options == "M 6"


def test_zl_record_preserves_provenance():
    records = parse_text(ZL_SAMPLE, "ZL3b")
    first = records[0]

    assert first.transcription_id == "ZL3b"
    assert first.ivtff_alphabet == "Eva-"
    assert first.ivtff_version == "2.0"
    assert first.ivtff_options == "M 5"
    assert first.file_header_original == "#=IVTFF Eva- 2.0 M 5"


def test_gc_record_preserves_native_alphabet():
    record = parse_text(GC_SAMPLE, "GC2a")[0]

    assert record.transcription_id == "GC2a"
    assert record.ivtff_alphabet == "v101"
    assert record.text_normalized == "<%>fa19s.9,hae.ay.Akam"


def test_it_record_preserves_native_alphabet():
    record = parse_text(IT_SAMPLE, "IT2a")[0]

    assert record.transcription_id == "IT2a"
    assert record.ivtff_alphabet == "EvaT"
    assert record.ivtff_options == "M 3"


def test_rf_extended_eva_is_not_normalized_away():
    record = parse_text(RF_SAMPLE, "RF1b-full")[0]

    assert record.transcription_id == "RF1b-full"
    assert record.ivtff_alphabet == "Eva-"

    # Critical: parser must not reinterpret extended EVA or ligature notation.
    assert "@221;" in record.text_raw
    assert "@221;" in record.text_normalized
    assert "{cto}" in record.text_raw
    assert "{cto}" in record.text_normalized


def test_requested_core_fields_are_present():
    record = parse_text(ZL_SAMPLE, "ZL3b")[0]
    data = record.to_dict()

    expected = {
        "folio",
        "quire",
        "currier",
        "scribe",
        "section",
        "locus",
        "locus_type",
        "paragraph",
        "line",
        "text_raw",
        "text_normalized",
    }

    assert expected.issubset(data.keys())


def test_page_metadata_is_parsed():
    record = parse_text(ZL_SAMPLE, "ZL3b")[0]

    assert record.folio == "f1r"
    assert record.physical_leaf == 1
    assert record.recto_verso == "r"
    assert record.foldout_panel is None
    assert record.quire == 1
    assert record.quire_code == "A"
    assert record.currier == "A"
    assert record.scribe == "1"
    assert record.section_code == "T"
    assert record.section == "text_only"


def test_locus_metadata_is_parsed():
    record = parse_text(ZL_SAMPLE, "ZL3b")[0]

    assert record.locus == "f1r.1,@P0"
    assert record.locus_raw == "<f1r.1,@P0>"
    assert record.locus_type == "P"
    assert record.locus_subtype == "P0"
    assert record.locator == "@"
    assert record.line == 1


def test_paragraph_markers_are_preserved_and_tracked():
    records = parse_text(ZL_SAMPLE, "ZL3b")

    assert records[0].paragraph == 1
    assert records[0].paragraph_start is True
    assert records[0].paragraph_end is False

    assert records[1].paragraph == 1
    assert records[1].paragraph_start is False
    assert records[1].paragraph_end is True

    assert records[2].paragraph is None


def test_original_source_line_is_preserved():
    record = parse_text(ZL_SAMPLE, "ZL3b")[0]

    assert record.original == "<f1r.1,@P0>       <%>fachys.ykal.ar.ataiin"
    assert record.original_lines == (
        "<f1r.1,@P0>       <%>fachys.ykal.ar.ataiin",
    )
    assert record.text_raw == "<%>fachys.ykal.ar.ataiin"


def test_normalization_only_removes_continuation_wrapping():
    raw = "fachys.ykal./\n/  ar.ataiin"
    normalized = normalize_text_lossless(raw)

    assert normalized == "fachys.ykal.ar.ataiin"


def test_normalization_preserves_semantic_markup():
    raw = "<%>qo,keedy.[a:b].{cto}.@221;.<->.???<$>"
    assert normalize_text_lossless(raw) == raw


def test_text_tag_updates_effective_page_variables():
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=T $L=A $H=1>
<f1r.1,@P0> <%>daiin
<f1r.2,+P0> <@L=B>ol<$>
<f1r.3,@L0> chedy
"""
    records = parse_text(sample, "ZL3b")

    assert records[0].currier == "A"
    assert records[1].currier == "B"
    assert records[2].currier == "B"


def test_foldout_panel_is_parsed():
    sample = """#=IVTFF Eva- 2.0 M 5
<f85r2> <! $Q=O $I=C $L=B $H=3>
<f85r2.1,@P0> <%>daiin<$>
"""
    record = parse_text(sample, "ZL3b")[0]

    assert record.physical_leaf == 85
    assert record.recto_verso == "r"
    assert record.foldout_panel == 2


def test_transcriber_suffix_is_preserved():
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=T $L=A $H=1>
<f1r.1,@P0;Z> <%>daiin<$>
"""
    record = parse_text(sample, "ZL3b")[0]
    assert record.transcriber == "Z"


def test_strict_mode_rejects_locus_before_page_header():
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r.1,@P0> <%>daiin<$>
"""
    with pytest.raises(IVTFFParseError):
        list(
            parse_ivtff_lines(
                sample.splitlines(True),
                transcription_id="ZL3b",
                strict=True,
            )
        )


def test_strict_mode_rejects_mismatched_active_page():
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=T $L=A $H=1>
<f2r.1,@P0> <%>daiin<$>
"""
    with pytest.raises(IVTFFParseError):
        list(
            parse_ivtff_lines(
                sample.splitlines(True),
                transcription_id="ZL3b",
                strict=True,
            )
        )


def test_load_ivtff_from_file(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(ZL_SAMPLE, encoding="ascii")

    records = load_ivtff(path, transcription_id="ZL3b")

    assert len(records) == 3
    assert all(r.transcription_id == "ZL3b" for r in records)


@pytest.mark.parametrize(
    ("sample", "transcription_id", "alphabet"),
    [
        (ZL_SAMPLE, "ZL3b", "Eva-"),
        (GC_SAMPLE, "GC2a", "v101"),
        (IT_SAMPLE, "IT2a", "EvaT"),
        (RF_SAMPLE, "RF1b-full", "Eva-"),
    ],
)
def test_supported_robustness_headers(sample, transcription_id, alphabet):
    record = parse_text(sample, transcription_id)[0]
    assert record.transcription_id == transcription_id
    assert record.ivtff_alphabet == alphabet
