#!/usr/bin/env python3
"""Fetch Yale MS 408 images for the frozen TRAIN-only image-validation sample.

Version 3 resolves Yale's coarse foldout labels as well as compound labels:

    68r1, r2, r3
    71v, 72r1, r2, r3
    86v5, v3
    101v1, 102r1, r2

It also supports one-to-many mappings. Thus a generic analytical folio such
as ``f101v`` may correctly require both the ``f101v1`` and ``f101v2`` Yale
canvases instead of being rejected as ambiguous.

This script automates retrieval and presentation only. Human review remains
the authority for glyph identity, spacing, and locus alignment.

It never reads validation.txt or test_LOCKED.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PAGES = Path("results/qc/image_validation/v1/pages.csv")
DEFAULT_LOCI = Path("results/qc/image_validation/v1/loci.csv")
DEFAULT_OUTPUT = Path("results/qc/image_validation/v1/yale_images")
DEFAULT_MANIFEST_URLS = (
    "https://collections.library.yale.edu/manifests/2002046",
    "https://collections.library.yale.edu/manifests/oid/2002046",
)

USER_AGENT = (
    "VOYAGER-Voynich-QC/3.0 "
    "(research image-validation; Beinecke MS 408)"
)

# Full folio token, e.g. 68r2 / f101v1.
_FULL_FOLIO_RE = re.compile(
    r"(?<![A-Za-z0-9])f?(\d{1,3})([rv])(\d*)",
    re.IGNORECASE,
)

# Panel shorthand following an already established folio number,
# e.g. ", r2" in "72r1, r2, r3" or ", v3" in "86v5, v3".
_SHORT_PANEL_RE = re.compile(
    r"(?<![A-Za-z0-9])([rv])(\d+)",
    re.IGNORECASE,
)

# Token stream preserving order between full and shorthand references.
_FOLIO_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"f?(\d{1,3})([rv])(\d*)"
    r"|"
    r"([rv])(\d+)"
    r")",
    re.IGNORECASE,
)


# Yale's current IIIF canvas labels intentionally collapse some foldout panels.
# These rules map analytical panel identifiers to the *coarse Yale label*, not
# to an inferred crop. The full Yale scan is downloaded for human review.
#
# Verified against the current MS 408 digital-collection image list:
#   f68r1/r2/r3 -> "68r"
#   f72r1/r2/r3 -> "71v and 72r"
#   f86v3       -> "86v (part) (part of 85-86 foldout)"
#
# The f86v3 assignment additionally follows the established external foldout
# panel convention in which v5/v3 occupy that Yale scan. The mapping is
# explicitly recorded in output provenance instead of being silently guessed.
PANEL_COARSE_LABEL_RULES = {
    "f68r1": re.compile(r"^68r$", re.IGNORECASE),
    "f68r2": re.compile(r"^68r$", re.IGNORECASE),
    "f68r3": re.compile(r"^68r$", re.IGNORECASE),
    "f72r1": re.compile(r"^71v\s+and\s+72r$", re.IGNORECASE),
    "f72r2": re.compile(r"^71v\s+and\s+72r$", re.IGNORECASE),
    "f72r3": re.compile(r"^71v\s+and\s+72r$", re.IGNORECASE),
    "f86v3": re.compile(
        r"^86v\s+\(part\)\s+\(part of 85-86 foldout\)$",
        re.IGNORECASE,
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def http_get_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "application/ld+json, application/json, "
                "image/jpeg, */*"
            ),
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(urls: Sequence[str]) -> Tuple[dict, str]:
    errors = []
    for url in urls:
        try:
            payload = http_get_bytes(url)
            return json.loads(payload.decode("utf-8")), url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Unable to fetch Yale IIIF manifest from any configured endpoint:\n"
        + "\n".join(f"  - {error}" for error in errors)
    )


def flatten_text(value: Any) -> List[str]:
    out: List[str] = []

    if value is None:
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_text(item))
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "@id", "id", "@type", "type",
                "service", "items", "images",
                "thumbnail", "height", "width",
            }:
                continue
            out.extend(flatten_text(item))
        return out

    return out


def canonical_folio(
    number: str,
    side: str,
    panel: str = "",
) -> str:
    return f"f{int(number)}{side.lower()}{panel}"


def split_canonical_folio(folio: str) -> Tuple[str, str]:
    """Return (base, panel), e.g. f101v1 -> (f101v, '1')."""
    match = re.fullmatch(
        r"f?(\d{1,3})([rv])(\d*)",
        folio.strip(),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unsupported folio identifier: {folio!r}")
    number, side, panel = match.groups()
    return f"f{int(number)}{side.lower()}", panel


def extract_canonical_folios(text: str) -> List[str]:
    """Expand compound Yale labels into canonical panel folios.

    Examples
    --------
    "68r1, r2, r3" -> ["f68r1", "f68r2", "f68r3"]
    "71v, 72r1, r2, r3" ->
        ["f71v", "f72r1", "f72r2", "f72r3"]
    "86v5, v3" -> ["f86v5", "f86v3"]
    """
    results: List[str] = []
    current_number: Optional[str] = None

    for match in _FOLIO_TOKEN_RE.finditer(text):
        number, side, panel, short_side, short_panel = match.groups()

        if number is not None:
            current_number = number
            folio = canonical_folio(number, side, panel)
        else:
            if current_number is None:
                continue
            folio = canonical_folio(
                current_number,
                short_side,
                short_panel,
            )

        if folio not in results:
            results.append(folio)

    return results


def label_text(canvas: Mapping[str, Any]) -> str:
    return " | ".join(flatten_text(canvas.get("label")))


def metadata_text(canvas: Mapping[str, Any]) -> str:
    pieces: List[str] = []
    for entry in canvas.get("metadata", []) or []:
        if isinstance(entry, dict):
            pieces.extend(flatten_text(entry.get("label")))
            pieces.extend(flatten_text(entry.get("value")))
    return " | ".join(pieces)


def canvas_identifier(canvas: Mapping[str, Any]) -> str:
    return str(canvas.get("id") or canvas.get("@id") or "")


def canvas_folios(canvas: Mapping[str, Any]) -> List[str]:
    # Labels are the primary source. Metadata is included only as a fallback
    # for Yale manifests whose canvas label is generic.
    label = label_text(canvas)
    extracted = extract_canonical_folios(label)
    if extracted:
        return extracted
    return extract_canonical_folios(metadata_text(canvas))


def iter_canvases(manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    # IIIF Presentation 3.
    if isinstance(manifest.get("items"), list):
        items = [
            item for item in manifest["items"]
            if isinstance(item, dict)
        ]
        if items:
            return items

    # IIIF Presentation 2.
    canvases: List[Mapping[str, Any]] = []
    for sequence in manifest.get("sequences", []) or []:
        if not isinstance(sequence, dict):
            continue
        for canvas in sequence.get("canvases", []) or []:
            if isinstance(canvas, dict):
                canvases.append(canvas)

    if not canvases:
        raise ValueError(
            "No canvases found in Yale IIIF manifest; "
            "unsupported or changed manifest structure."
        )

    return canvases


def resolve_canvases(
    target_folio: str,
    canvases: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], str, List[dict]]:
    """Resolve one analytical folio to one or more Yale canvases.

    Resolution order:
      1. exact panel/generic folio extracted from Yale label;
      2. one-to-many Yale "(part)" canvases for a generic folio;
      3. explicit coarse-label rule for foldout panel identifiers.

    The explicit foldout rule returns the whole Yale scan. It never invents a
    crop or claims that Yale itself supplied the analytical panel identifier.
    """
    target_folio = target_folio.strip().lower()
    target_base, target_panel = split_canonical_folio(target_folio)

    candidates: List[dict] = []

    for index, canvas in enumerate(canvases):
        label = label_text(canvas).strip()
        folios = canvas_folios(canvas)

        exact = target_folio in {
            folio.lower() for folio in folios
        }
        same_base = [
            folio
            for folio in folios
            if split_canonical_folio(folio)[0] == target_base
        ]

        if exact:
            candidates.append(
                {
                    "score": 120,
                    "reason": "exact_folio_label",
                    "index": index,
                    "label": label,
                    "canvas_id": canvas_identifier(canvas),
                    "expanded_folios": folios,
                    "matched_folios": [target_folio],
                    "canvas": canvas,
                }
            )
            continue

        if not target_panel and same_base:
            candidates.append(
                {
                    "score": 90,
                    "reason": "generic_folio_part",
                    "index": index,
                    "label": label,
                    "canvas_id": canvas_identifier(canvas),
                    "expanded_folios": folios,
                    "matched_folios": same_base,
                    "canvas": canvas,
                }
            )

    exact = [
        row for row in candidates
        if row["reason"] == "exact_folio_label"
    ]

    if target_panel and exact:
        if len(exact) == 1:
            return (
                [exact[0]["canvas"]],
                "exact_panel_folio",
                [
                    {k: v for k, v in row.items() if k != "canvas"}
                    for row in candidates
                ],
            )
        return (
            [],
            "ambiguous_exact_panel",
            [
                {k: v for k, v in row.items() if k != "canvas"}
                for row in exact
            ],
        )

    if not target_panel and exact:
        # Yale intentionally splits some generic folios across multiple
        # "(part)" canvases (e.g. 101v). In that case *all* parts are needed.
        unique = []
        seen = set()
        for row in sorted(exact, key=lambda item: item["index"]):
            cid = row["canvas_id"]
            if cid in seen:
                continue
            seen.add(cid)
            unique.append(row["canvas"])

        if len(unique) == 1:
            kind = "exact_generic_folio"
        else:
            labels = [
                label_text(canvas).lower()
                for canvas in unique
            ]
            if all("(part)" in label for label in labels):
                kind = "multi_part_generic_folio"
            else:
                return (
                    [],
                    "ambiguous_generic_match",
                    [
                        {k: v for k, v in row.items() if k != "canvas"}
                        for row in exact
                    ],
                )

        return (
            unique,
            kind,
            [
                {k: v for k, v in row.items() if k != "canvas"}
                for row in candidates
            ],
        )

    if not target_panel:
        generic_parts = [
            row for row in candidates
            if row["reason"] == "generic_folio_part"
        ]
        if generic_parts:
            unique = []
            seen = set()
            for row in sorted(
                generic_parts,
                key=lambda item: item["index"],
            ):
                cid = row["canvas_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                unique.append(row["canvas"])
            return (
                unique,
                (
                    "panel_canvases_for_generic_folio"
                    if len(unique) > 1
                    else "single_panel_canvas_for_generic_folio"
                ),
                [
                    {k: v for k, v in row.items() if k != "canvas"}
                    for row in candidates
                ],
            )

    # Yale's manifest often names an entire foldout scan by a coarse physical
    # side, while the analytical transcription uses panel IDs. Apply only
    # explicitly documented rules.
    coarse_rule = PANEL_COARSE_LABEL_RULES.get(target_folio)
    if coarse_rule is not None:
        coarse_matches = []
        for index, canvas in enumerate(canvases):
            label = label_text(canvas).strip()
            if coarse_rule.fullmatch(label):
                coarse_matches.append(
                    {
                        "score": 80,
                        "reason": "documented_coarse_foldout_label",
                        "index": index,
                        "label": label,
                        "canvas_id": canvas_identifier(canvas),
                        "expanded_folios": canvas_folios(canvas),
                        "matched_folios": [target_folio],
                        "canvas": canvas,
                    }
                )

        if len(coarse_matches) == 1:
            return (
                [coarse_matches[0]["canvas"]],
                "documented_coarse_foldout_label",
                [
                    {k: v for k, v in row.items() if k != "canvas"}
                    for row in coarse_matches
                ],
            )

        if len(coarse_matches) > 1:
            return (
                [],
                "ambiguous_coarse_foldout_label",
                [
                    {k: v for k, v in row.items() if k != "canvas"}
                    for row in coarse_matches
                ],
            )

    return (
        [],
        "no_match",
        [
            {k: v for k, v in row.items() if k != "canvas"}
            for row in candidates
        ],
    )


def first_service_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        service_id = value.get("id") or value.get("@id")
        if service_id:
            return str(service_id).rstrip("/")
    if isinstance(value, list):
        for item in value:
            found = first_service_id(item)
            if found:
                return found
    return None


def extract_image_source(
    canvas: Mapping[str, Any],
    *,
    max_width: int,
) -> Tuple[str, str]:
    # IIIF Presentation 3.
    for annotation_page in canvas.get("items", []) or []:
        if not isinstance(annotation_page, dict):
            continue
        for annotation in annotation_page.get("items", []) or []:
            if not isinstance(annotation, dict):
                continue
            body = annotation.get("body")
            bodies = body if isinstance(body, list) else [body]
            for candidate in bodies:
                if not isinstance(candidate, dict):
                    continue
                service_id = first_service_id(
                    candidate.get("service")
                )
                if service_id:
                    return (
                        f"{service_id}/full/{max_width},/0/default.jpg",
                        "iiif_image_service",
                    )
                body_id = (
                    candidate.get("id")
                    or candidate.get("@id")
                )
                if body_id:
                    return str(body_id), "annotation_body"

    # IIIF Presentation 2.
    for annotation in canvas.get("images", []) or []:
        if not isinstance(annotation, dict):
            continue
        resource = annotation.get("resource")
        if not isinstance(resource, dict):
            continue
        service_id = first_service_id(resource.get("service"))
        if service_id:
            return (
                f"{service_id}/full/{max_width},/0/default.jpg",
                "iiif_image_service",
            )
        resource_id = (
            resource.get("@id")
            or resource.get("id")
        )
        if resource_id:
            return str(resource_id), "image_resource"

    raise ValueError(
        "No downloadable image resource found for canvas "
        f"{canvas_identifier(canvas)!r}"
    )


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_gallery(
    path: Path,
    pages: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    loci_rows: Sequence[Mapping[str, Any]],
) -> None:
    images_by_folio: Dict[str, List[Mapping[str, Any]]] = {}
    for row in image_rows:
        images_by_folio.setdefault(
            str(row["folio"]),
            [],
        ).append(row)

    loci_by_folio: Dict[str, List[Mapping[str, Any]]] = {}
    for row in loci_rows:
        loci_by_folio.setdefault(
            str(row["folio"]),
            [],
        ).append(row)

    sections: List[str] = []

    for page in pages:
        folio_raw = str(page["folio"])
        folio = html.escape(folio_raw)

        image_blocks: List[str] = []
        for row in images_by_folio.get(folio_raw, []):
            local_image = str(row.get("local_image", ""))
            canvas_label = html.escape(
                str(row.get("canvas_label", ""))
            )
            canvas_url = html.escape(
                str(row.get("canvas_id", ""))
            )
            image_url = html.escape(
                str(row.get("image_url", ""))
            )

            if local_image:
                image_html = (
                    f'<a href="{html.escape(local_image)}">'
                    f'<img loading="lazy" '
                    f'src="{html.escape(local_image)}" '
                    f'alt="{folio}"></a>'
                )
            else:
                image_html = (
                    "<div class='missing'>"
                    "No image downloaded"
                    "</div>"
                )

            image_blocks.append(
                f"""
                <div class="scan">
                  <h3>{canvas_label}</h3>
                  {image_html}
                  <p>
                    <a href="{canvas_url}">IIIF canvas</a>
                    ·
                    <a href="{image_url}">image source</a>
                  </p>
                </div>
                """
            )

        locus_blocks: List[str] = []
        for locus in loci_by_folio.get(folio_raw, []):
            locus_id = html.escape(str(locus.get("locus", "")))
            text = html.escape(
                str(locus.get("text_normalized", ""))
            )
            locus_blocks.append(
                f"""
                <div class="locus">
                  <b>{locus_id}</b>
                  <code>{text}</code>
                </div>
                """
            )

        status = (
            "resolved"
            if image_blocks
            else "unresolved"
        )

        sections.append(
            f"""
            <section class="card">
              <h2>{folio} <small>{status}</small></h2>
              <div class="images">
                {''.join(image_blocks) or "<p>No scan resolved.</p>"}
              </div>
              <h3>Sampled ZL3b loci</h3>
              <div class="loci">
                {''.join(locus_blocks) or "<p>No locus rows found.</p>"}
              </div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VOYAGER — Yale manuscript image QC</title>
<style>
body {{
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 1500px;
  margin: 2rem auto;
  padding: 0 1rem;
}}
.notice {{
  padding: 1rem;
  border: 1px solid #888;
  margin-bottom: 2rem;
}}
.card {{
  margin-bottom: 4rem;
  padding-bottom: 3rem;
  border-bottom: 2px solid #aaa;
}}
.images {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
  gap: 1.5rem;
}}
.scan img {{
  width: 100%;
  max-height: 1050px;
  object-fit: contain;
  border: 1px solid #ccc;
}}
.locus {{
  margin: 0.6rem 0;
  padding: 0.7rem;
  border: 1px solid #ddd;
}}
.locus code {{
  display: block;
  white-space: pre-wrap;
  margin-top: 0.35rem;
  font-size: 1rem;
}}
small {{ font-weight: normal; }}
.missing {{ padding: 3rem; border: 1px dashed #999; }}
</style>
</head>
<body>
<h1>VOYAGER — frozen manuscript-image validation sample</h1>
<div class="notice">
Images are retrieved automatically from Yale's IIIF service. Human review is
still required for glyph identity, spacing, and locus alignment. The text
shown below each scan is the sampled ZL3b transcription, not OCR.
</div>
{''.join(sections)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--loci", type=Path, default=DEFAULT_LOCI)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest-url",
        action="append",
        default=[],
    )
    parser.add_argument("--max-width", type=int, default=1800)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    try:
        pages = read_csv(args.pages)
        loci = read_csv(args.loci)

        manifest_urls = (
            tuple(args.manifest_url)
            if args.manifest_url
            else DEFAULT_MANIFEST_URLS
        )
        manifest, manifest_url = fetch_json(manifest_urls)
        canvases = iter_canvases(manifest)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = args.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_rows: List[dict] = []
        unresolved_folios: List[str] = []

        for page in pages:
            folio = str(page["folio"]).strip()

            (
                resolved_canvases,
                resolution_kind,
                candidates,
            ) = resolve_canvases(folio, canvases)

            if not resolved_canvases:
                unresolved_folios.append(folio)
                print(
                    f"[UNRESOLVED] {folio}: "
                    f"{resolution_kind}"
                )
                image_rows.append(
                    {
                        "sample_index": page.get("sample_index", ""),
                        "folio": folio,
                        "leaf_group": page.get("leaf_group", ""),
                        "variant_index": "",
                        "variant_count": 0,
                        "resolution_kind": resolution_kind,
                        "fetch_status": "UNRESOLVED",
                        "canvas_label": "",
                        "canvas_id": "",
                        "expanded_canvas_folios": "",
                        "image_url": "",
                        "local_image": "",
                        "sha256": "",
                        "manifest_url": manifest_url,
                        "candidate_matches_json": json.dumps(
                            candidates,
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            variant_count = len(resolved_canvases)

            for variant_index, canvas in enumerate(
                resolved_canvases,
                start=1,
            ):
                label = label_text(canvas)
                image_url, image_kind = extract_image_source(
                    canvas,
                    max_width=args.max_width,
                )

                if variant_count == 1:
                    filename = f"{folio}.jpg"
                else:
                    filename = (
                        f"{folio}__{variant_index:02d}.jpg"
                    )

                local_rel = ""
                digest = ""
                status = "RESOLVED"

                if not args.no_download:
                    destination = images_dir / filename
                    print(
                        f"[FETCH] {folio:<8} "
                        f"[{variant_index}/{variant_count}] "
                        f"{label!r}"
                    )
                    payload = http_get_bytes(image_url)
                    if len(payload) < 10_000:
                        raise RuntimeError(
                            f"Image for {folio} is unexpectedly "
                            f"small ({len(payload)} bytes)"
                        )
                    destination.write_bytes(payload)
                    digest = sha256_file(destination)
                    local_rel = str(
                        Path("images") / filename
                    )
                    status = "DOWNLOADED"
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                image_rows.append(
                    {
                        "sample_index": page.get("sample_index", ""),
                        "folio": folio,
                        "leaf_group": page.get("leaf_group", ""),
                        "variant_index": variant_index,
                        "variant_count": variant_count,
                        "resolution_kind": resolution_kind,
                        "fetch_status": status,
                        "canvas_label": label,
                        "canvas_id": canvas_identifier(canvas),
                        "expanded_canvas_folios": "|".join(
                            canvas_folios(canvas)
                        ),
                        "image_kind": image_kind,
                        "image_url": image_url,
                        "local_image": local_rel,
                        "sha256": digest,
                        "manifest_url": manifest_url,
                        "candidate_matches_json": json.dumps(
                            candidates,
                            ensure_ascii=False,
                        ),
                    }
                )

        mapping_path = args.output_dir / "yale_image_map.csv"
        write_csv(mapping_path, image_rows)

        resolved_folios = {
            row["folio"]
            for row in image_rows
            if row["fetch_status"] != "UNRESOLVED"
        }

        provenance = {
            "schema_version": "3.0",
            "manifest_url": manifest_url,
            "manifest_id": (
                manifest.get("id")
                or manifest.get("@id")
                or ""
            ),
            "manifest_label": flatten_text(
                manifest.get("label")
            ),
            "canvas_count": len(canvases),
            "sample_pages": len(pages),
            "resolved_folios": len(resolved_folios),
            "unresolved_folios": unresolved_folios,
            "downloaded_or_resolved_canvases": sum(
                1
                for row in image_rows
                if row["fetch_status"] != "UNRESOLVED"
            ),
            "max_width": args.max_width,
            "resolver": (
                "exact labels + multi-part generic folios + documented coarse foldout map"
            ),
            "validation_split_accessed": False,
            "locked_test_accessed": False,
        }
        (args.output_dir / "manifest_provenance.json").write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        gallery_path = args.output_dir / "review_gallery.html"
        build_gallery(
            gallery_path,
            pages,
            image_rows,
            loci,
        )

        checksums = [
            f"{row['sha256']}  {row['local_image']}"
            for row in image_rows
            if row.get("sha256") and row.get("local_image")
        ]
        (args.output_dir / "SHA256SUMS").write_text(
            "\n".join(checksums)
            + ("\n" if checksums else ""),
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("YALE IIIF IMAGE RETRIEVAL V3 — YALE COARSE-FOLDOUT AWARE")
        print("=" * 78)
        print(f"Manifest canvases:   {len(canvases):>5}")
        print(f"Sample folios:       {len(pages):>5}")
        print(f"Resolved folios:     {len(resolved_folios):>5}")
        print(f"Unresolved folios:   {len(unresolved_folios):>5}")
        print(
            f"Resolved canvases:   "
            f"{provenance['downloaded_or_resolved_canvases']:>5}"
        )
        print("Validation split:    NOT ACCESSED")
        print("Locked test:         NOT ACCESSED")
        print()
        print(f"Mapping: {mapping_path}")
        print(f"Gallery: {gallery_path}")

        if unresolved_folios:
            print()
            print(
                "Still unresolved: "
                + ", ".join(unresolved_folios)
            )
            print(
                "Do not guess. Inspect candidate_matches_json "
                "for those rows."
            )
            return 1

        print()
        print("All sampled folios resolved.")
        if not args.no_download:
            print(f"Open gallery:\n  open {gallery_path}")
        return 0

    except (HTTPError, URLError) as exc:
        print(f"NETWORK ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"YALE IMAGE FETCH ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
