from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FETCHER = load_module(
    Path(__file__).resolve().parents[1]
    / "scripts/fetch_yale_image_validation.py",
    "fetch_yale_image_validation_v3",
)


def canvas(label: str, suffix: str):
    return {
        "@id": f"https://example.test/canvas/{suffix}",
        "label": label,
        "images": [
            {
                "resource": {
                    "@id": f"https://example.test/{suffix}.jpg",
                    "service": {
                        "@id": f"https://iiif.test/{suffix}"
                    },
                }
            }
        ],
    }


def test_normal_exact_folio():
    result, kind, _ = FETCHER.resolve_canvases(
        "f41r",
        [canvas("41r", "x")],
    )
    assert len(result) == 1
    assert kind == "exact_generic_folio"


def test_f68r2_maps_to_yale_coarse_68r_scan():
    result, kind, _ = FETCHER.resolve_canvases(
        "f68r2",
        [canvas("68r", "1006196")],
    )
    assert len(result) == 1
    assert FETCHER.label_text(result[0]) == "68r"
    assert kind == "documented_coarse_foldout_label"


def test_f72r3_maps_to_yale_71v_and_72r_scan():
    result, kind, _ = FETCHER.resolve_canvases(
        "f72r3",
        [canvas("71v and 72r", "1006203")],
    )
    assert len(result) == 1
    assert FETCHER.label_text(result[0]) == "71v and 72r"
    assert kind == "documented_coarse_foldout_label"


def test_f86v3_maps_to_specific_yale_86v_part_scan():
    canvases = [
        canvas(
            "85r (part) 86v (part) (part of 85-86 foldout)",
            "1006229",
        ),
        canvas(
            "86v (part) (part of 85-86 foldout)",
            "1006230",
        ),
        canvas(
            "85v and 86r (foldout)",
            "1006231",
        ),
    ]
    result, kind, _ = FETCHER.resolve_canvases(
        "f86v3",
        canvases,
    )
    assert len(result) == 1
    assert FETCHER.canvas_identifier(result[0]).endswith("1006230")
    assert kind == "documented_coarse_foldout_label"


def test_generic_f101v_accepts_both_yale_part_canvases():
    canvases = [
        canvas("101v (part)", "1006250"),
        canvas("101v (part) and 102r", "1006251"),
    ]
    result, kind, _ = FETCHER.resolve_canvases(
        "f101v",
        canvases,
    )
    assert len(result) == 2
    assert kind == "multi_part_generic_folio"


def test_non_part_duplicate_generic_remains_ambiguous():
    canvases = [
        canvas("79r", "a"),
        canvas("79r", "b"),
    ]
    result, kind, _ = FETCHER.resolve_canvases(
        "f79r",
        canvases,
    )
    assert result == []
    assert kind == "ambiguous_generic_match"


def test_v2_image_service_url():
    item = canvas("41r", "image")
    url, kind = FETCHER.extract_image_source(
        item,
        max_width=1800,
    )
    assert url == (
        "https://iiif.test/image/full/1800,/0/default.jpg"
    )
    assert kind == "iiif_image_service"
