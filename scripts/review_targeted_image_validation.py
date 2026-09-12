#!/usr/bin/env python3
"""Targeted human-in-the-loop manuscript-image audit for VOYAGER.

This reviewer is intentionally narrower than the original 57-locus reviewer.

Reads
-----
results/qc/image_validation/v1/cross_transcription/recommended_manual_review.csv
results/qc/image_validation/v1/yale_images/yale_image_map.csv
results/qc/image_validation/v1/yale_images/images/*.jpg

Writes
------
results/qc/image_validation/v1/cross_transcription/targeted_manual_review.csv

The original 57-locus loci.csv/pages.csv files are NOT modified.

Review scope
------------
MANDATORY_DISAGREEMENT / BOUNDARY_ONLY_DISAGREEMENT:
    Inspect locus alignment and whether the disputed spacing is visually
    clear, ambiguous, or unresolved. Do not re-transcribe the line.

TARGETED_DISAGREEMENT:
    Confirm the correct manuscript region is shown. Decide only whether the
    transcription difference looks like an obvious QC problem, a plausible
    paleographic/transcription variation, or is unresolved. Do not guess
    subtle EVA glyph identities.

CONSENSUS_CONTROL:
    Confirm that the exact-agreement control points to the expected manuscript
    region. This is an alignment sanity check, not a fresh transcription.

No validation or locked-test split file is read.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse


DEFAULT_ROOT = Path("results/qc/image_validation/v1")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766

SHORTLIST_REL = Path("cross_transcription/recommended_manual_review.csv")
OUTPUT_REL = Path("cross_transcription/targeted_manual_review.csv")
MAP_REL = Path("yale_images/yale_image_map.csv")

ALLOWED_STATUS = {
    "",
    "NO_OBVIOUS_QC_PROBLEM",
    "PARSER_OR_ALIGNMENT_ISSUE",
    "TRANSCRIPTION_CONCERN",
    "UNRESOLVED",
}
ALLOWED_ALIGNMENT = {"", "YES", "NO", "UNCERTAIN"}
ALLOWED_BOUNDARY = {
    "",
    "CLEAR_VISIBLE_GAP",
    "NO_CLEAR_VISIBLE_GAP",
    "AMBIGUOUS",
    "UNRESOLVED",
    "NOT_APPLICABLE",
}
ALLOWED_GLYPH_ASSESSMENT = {
    "",
    "PLAUSIBLE_TRANSCRIPTION_VARIATION",
    "OBVIOUS_TRANSCRIPTION_CONCERN",
    "UNRESOLVED",
    "NOT_APPLICABLE",
}


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows in {path}")
    return fields, rows


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


OUTPUT_FIELDS = [
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
    "review_status",
    "locus_alignment",
    "boundary_assessment",
    "glyph_disagreement_assessment",
    "reviewer_notes",
]


def initialize_output(shortlist: list[dict], output_path: Path) -> tuple[list[str], list[dict]]:
    """Create or load a persistent 10-case review worksheet."""
    if output_path.is_file():
        fields, rows = read_csv(output_path)
        required = set(OUTPUT_FIELDS)
        missing = required - set(fields)
        if missing:
            raise ValueError(
                f"{output_path} missing required fields: {', '.join(sorted(missing))}"
            )

        expected = [(r["review_rank"], r["locus"]) for r in shortlist]
        observed = [(r["review_rank"], r["locus"]) for r in rows]
        if observed != expected:
            raise ValueError(
                "Existing targeted_manual_review.csv does not match the frozen "
                "recommended_manual_review.csv. Refusing to silently realign it."
            )
        return fields, rows

    rows: list[dict] = []
    for source in shortlist:
        role = source.get("review_role", "")
        boundary = "NOT_APPLICABLE"
        glyph = "NOT_APPLICABLE"

        if source.get("comparison_class") == "BOUNDARY_ONLY_DISAGREEMENT":
            boundary = ""
        elif role == "TARGETED_DISAGREEMENT":
            glyph = ""

        rows.append(
            {
                "review_rank": source.get("review_rank", ""),
                "review_role": role,
                "folio": source.get("folio", ""),
                "locus": source.get("locus", ""),
                "comparison_class": source.get("comparison_class", ""),
                "review_reason": source.get("review_reason", ""),
                "primary_support": source.get("primary_support", ""),
                "min_pairwise_glyph_similarity": source.get(
                    "min_pairwise_glyph_similarity", ""
                ),
                "ZL3b_native": source.get("ZL3b_native", ""),
                "GC2a_native": source.get("GC2a_native", ""),
                "IT2a_native": source.get("IT2a_native", ""),
                "review_status": "",
                "locus_alignment": "",
                "boundary_assessment": boundary,
                "glyph_disagreement_assessment": glyph,
                "reviewer_notes": "",
            }
        )

    write_csv_atomic(output_path, OUTPUT_FIELDS, rows)
    return list(OUTPUT_FIELDS), rows


class TargetedReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.shortlist_path = self.root / SHORTLIST_REL
        self.output_path = self.root / OUTPUT_REL
        self.map_path = self.root / MAP_REL

        shortlist_fields, self.shortlist = read_csv(self.shortlist_path)
        required_shortlist = {
            "review_rank",
            "review_role",
            "folio",
            "locus",
            "comparison_class",
            "review_reason",
            "ZL3b_native",
            "GC2a_native",
            "IT2a_native",
        }
        missing = required_shortlist - set(shortlist_fields)
        if missing:
            raise ValueError(
                "recommended_manual_review.csv missing required fields: "
                + ", ".join(sorted(missing))
            )

        if len(self.shortlist) != 10:
            raise ValueError(
                "Expected the frozen targeted protocol to contain exactly "
                f"10 cases; found {len(self.shortlist)}."
            )

        _, self.image_rows = read_csv(self.map_path)
        self.fields, self.rows = initialize_output(
            self.shortlist,
            self.output_path,
        )

        self._lock = threading.Lock()
        self._backed_up = False

    def backup_once(self) -> None:
        if self._backed_up:
            return
        if self.output_path.is_file():
            shutil.copy2(
                self.output_path,
                self.output_path.with_suffix(".csv.bak"),
            )
        self._backed_up = True

    def images_for(self, folio: str) -> list[dict]:
        rows = [
            row
            for row in self.image_rows
            if row.get("folio", "") == folio
            and row.get("fetch_status", "") != "UNRESOLVED"
            and row.get("local_image", "")
        ]
        return sorted(
            rows,
            key=lambda row: int(row.get("variant_index") or 1),
        )

    def progress(self) -> dict:
        completed = sum(
            1 for row in self.rows if row.get("review_status", "").strip()
        )
        return {
            "completed": completed,
            "total": len(self.rows),
        }

    def save(self, index: int, payload: dict) -> None:
        if not 0 <= index < len(self.rows):
            raise IndexError("Review index out of range")

        status = str(payload.get("review_status", "")).strip()
        locus_alignment = str(payload.get("locus_alignment", "")).strip()
        boundary = str(payload.get("boundary_assessment", "")).strip()
        glyph = str(
            payload.get("glyph_disagreement_assessment", "")
        ).strip()
        notes = str(payload.get("reviewer_notes", "")).strip()

        if status not in ALLOWED_STATUS:
            raise ValueError(f"Invalid review_status: {status!r}")
        if locus_alignment not in ALLOWED_ALIGNMENT:
            raise ValueError(
                f"Invalid locus_alignment: {locus_alignment!r}"
            )
        if boundary not in ALLOWED_BOUNDARY:
            raise ValueError(
                f"Invalid boundary_assessment: {boundary!r}"
            )
        if glyph not in ALLOWED_GLYPH_ASSESSMENT:
            raise ValueError(
                "Invalid glyph_disagreement_assessment: "
                f"{glyph!r}"
            )

        row = self.rows[index]
        cls = row.get("comparison_class", "")
        role = row.get("review_role", "")

        if cls == "BOUNDARY_ONLY_DISAGREEMENT":
            if glyph not in {"", "NOT_APPLICABLE"}:
                raise ValueError(
                    "Glyph-disagreement assessment is not applicable to "
                    "a boundary-only case."
                )
        elif role == "TARGETED_DISAGREEMENT":
            if boundary not in {"", "NOT_APPLICABLE"}:
                raise ValueError(
                    "Boundary assessment is not applicable to a targeted "
                    "glyph-disagreement case."
                )
        elif role == "CONSENSUS_CONTROL":
            if boundary not in {"", "NOT_APPLICABLE"}:
                raise ValueError(
                    "Boundary assessment is not applicable to a control."
                )
            if glyph not in {"", "NOT_APPLICABLE"}:
                raise ValueError(
                    "Glyph assessment is not applicable to a control."
                )

        with self._lock:
            self.backup_once()
            row["review_status"] = status
            row["locus_alignment"] = locus_alignment
            row["boundary_assessment"] = boundary
            row["glyph_disagreement_assessment"] = glyph
            row["reviewer_notes"] = notes
            write_csv_atomic(
                self.output_path,
                self.fields,
                self.rows,
            )


def safe_image_path(store: TargetedReviewStore, rel: str) -> Path:
    rel_path = Path(unquote(rel))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("Unsafe image path")

    base = (store.root / "yale_images").resolve()
    path = (base / rel_path).resolve()

    if base not in path.parents:
        raise ValueError("Unsafe image path")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def option_tags(values: list[str], selected: str) -> str:
    parts = []
    for value in values:
        display = value or "—"
        sel = " selected" if value == selected else ""
        parts.append(
            f'<option value="{html.escape(value)}"{sel}>'
            f'{html.escape(display)}</option>'
        )
    return "".join(parts)


def role_instruction(row: dict) -> tuple[str, str]:
    cls = row.get("comparison_class", "")
    role = row.get("review_role", "")

    if cls == "BOUNDARY_ONLY_DISAGREEMENT":
        return (
            "BOUNDARY CHECK",
            "All three transcriptions agree on the glyph sequence. "
            "Do not read the line glyph-by-glyph. Confirm you are at the "
            "correct locus, then look only for whether the disputed word "
            "spacing is visually clear or ambiguous.",
        )

    if role == "TARGETED_DISAGREEMENT":
        return (
            "TARGETED TRANSCRIPTION CHECK",
            "Confirm the correct manuscript region. You are NOT required to "
            "decide the correct EVA glyph. Record whether the difference "
            "looks like a plausible transcription/paleographic variation, "
            "an obvious transcription concern, or is unresolved.",
        )

    return (
        "CONSENSUS CONTROL",
        "ZL3b, GC2a, and IT2a agree exactly here. This is only an alignment "
        "sanity check: confirm that the displayed locus points to the expected "
        "manuscript region. Do not re-transcribe the line.",
    )


def page_html(store: TargetedReviewStore, index: int) -> str:
    index = max(0, min(index, len(store.rows) - 1))
    row = store.rows[index]
    progress = store.progress()
    folio = row["folio"]
    images = store.images_for(folio)
    heading, instruction = role_instruction(row)

    image_blocks = []
    for img in images:
        local = img.get("local_image", "")
        label = img.get("canvas_label", "")
        canvas = img.get("canvas_id", "")
        source = img.get("image_url", "")
        src = "/image/" + quote(local, safe="/")
        image_blocks.append(
            f"""
            <div class="scan">
              <div class="scan-title">{html.escape(label)}</div>
              <div class="zoom-wrap">
                <img src="{src}" alt="{html.escape(folio)}">
              </div>
              <div class="links">
                <a href="{html.escape(canvas)}" target="_blank">Yale canvas</a>
                <a href="{html.escape(source)}" target="_blank">full image source</a>
              </div>
            </div>
            """
        )

    status = row.get("review_status", "")
    locus_alignment = row.get("locus_alignment", "")
    boundary = row.get("boundary_assessment", "")
    glyph_assessment = row.get("glyph_disagreement_assessment", "")
    notes = row.get("reviewer_notes", "")

    boundary_case = row.get("comparison_class") == "BOUNDARY_ONLY_DISAGREEMENT"
    glyph_case = row.get("review_role") == "TARGETED_DISAGREEMENT"

    boundary_style = "" if boundary_case else ' style="display:none"'
    glyph_style = "" if glyph_case else ' style="display:none"'

    prev_i = max(0, index - 1)
    next_i = min(len(store.rows) - 1, index + 1)

    statuses = [
        "",
        "NO_OBVIOUS_QC_PROBLEM",
        "PARSER_OR_ALIGNMENT_ISSUE",
        "TRANSCRIPTION_CONCERN",
        "UNRESOLVED",
    ]
    alignments = ["", "YES", "NO", "UNCERTAIN"]
    boundaries = [
        "",
        "CLEAR_VISIBLE_GAP",
        "NO_CLEAR_VISIBLE_GAP",
        "AMBIGUOUS",
        "UNRESOLVED",
    ]
    glyph_values = [
        "",
        "PLAUSIBLE_TRANSCRIPTION_VARIATION",
        "OBVIOUS_TRANSCRIPTION_CONCERN",
        "UNRESOLVED",
    ]

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VOYAGER targeted image QC — {html.escape(folio)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  margin: 0;
}}
header {{
  position: sticky; top: 0; z-index: 20;
  padding: .75rem 1rem;
  background: Canvas;
  border-bottom: 1px solid GrayText;
  display: flex; justify-content: space-between; gap: 1rem;
}}
main {{
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(390px, 1.25fr);
  gap: 1rem;
  padding: 1rem;
}}
.scans {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1rem;
}}
.scan {{ border: 1px solid GrayText; padding: .5rem; }}
.scan-title {{ font-weight: 700; margin-bottom: .4rem; }}
.zoom-wrap {{ height: 78vh; overflow: auto; background: #222; }}
.scan img {{
  width: 100%; height: auto; cursor: zoom-in; transform-origin: top left;
}}
.links {{ display: flex; gap: 1rem; margin-top: .4rem; }}
.panel {{
  position: sticky; top: 4.5rem; align-self: start;
  border: 1px solid GrayText; padding: 1rem; border-radius: .5rem;
  max-height: calc(100vh - 6rem); overflow-y: auto;
}}
.scope {{
  border: 2px solid GrayText; padding: .8rem; border-radius: .5rem;
  margin-bottom: .9rem;
}}
.scope h3 {{ margin: 0 0 .35rem 0; }}
.transcription {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; overflow-wrap: anywhere;
  padding: .65rem; border: 1px solid GrayText; margin: .35rem 0 .8rem;
}}
.transcription-label {{ font-weight: 700; margin-top: .7rem; }}
.small {{ font-size: .9rem; opacity: .82; }}
label {{ display: block; margin-top: .8rem; font-weight: 700; }}
select, textarea, button {{
  width: 100%; box-sizing: border-box; padding: .55rem; font: inherit;
}}
textarea {{ min-height: 90px; }}
.buttons {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;
  margin-top: .9rem;
}}
.quick {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .4rem;
  margin-top: .7rem;
}}
.quick button {{ min-height: 3rem; }}
.reason {{
  border-left: 4px solid GrayText; padding-left: .7rem; margin: .7rem 0;
}}
@media (max-width: 1000px) {{
  main {{ grid-template-columns: 1fr; }}
  .panel {{ position: static; max-height: none; }}
}}
</style>
</head>
<body>
<header>
  <div>
    <b>Targeted manuscript-image audit</b>
    · case {index + 1}/{len(store.rows)}
    · reviewed {progress['completed']}/{progress['total']}
  </div>
  <div>
    <b>{html.escape(folio)}</b> · {html.escape(row.get('locus', ''))}
  </div>
</header>

<main>
<section>
  <div class="scans">
    {''.join(image_blocks) or '<p>No local Yale image found for this folio.</p>'}
  </div>
</section>

<aside class="panel">
  <div class="small">
    Review rank #{html.escape(row.get('review_rank', ''))}
    · {html.escape(row.get('review_role', ''))}
    · {html.escape(row.get('comparison_class', ''))}
  </div>

  <div class="scope">
    <h3>{html.escape(heading)}</h3>
    <div>{html.escape(instruction)}</div>
  </div>

  <div class="reason">
    <b>Why this case was selected</b><br>
    {html.escape(row.get('review_reason', ''))}
  </div>

  <div class="transcription-label">ZL3b</div>
  <div class="transcription">{html.escape(row.get('ZL3b_native', ''))}</div>

  <div class="transcription-label">GC2a</div>
  <div class="transcription">{html.escape(row.get('GC2a_native', ''))}</div>

  <div class="transcription-label">IT2a</div>
  <div class="transcription">{html.escape(row.get('IT2a_native', ''))}</div>

  <div class="quick">
    <button onclick="quickStatus('NO_OBVIOUS_QC_PROBLEM')">✓ no obvious QC problem</button>
    <button onclick="quickStatus('UNRESOLVED')">? unresolved</button>
    <button onclick="quickStatus('PARSER_OR_ALIGNMENT_ISSUE')">⚠ alignment issue</button>
    <button onclick="quickStatus('TRANSCRIPTION_CONCERN')">≠ transcription concern</button>
  </div>

  <label>Overall limited-scope review status</label>
  <select id="review_status">
    {option_tags(statuses, status)}
  </select>

  <label>Locus / region alignment</label>
  <select id="locus_alignment">
    {option_tags(alignments, locus_alignment)}
  </select>

  <div{boundary_style}>
    <label>Boundary visibility</label>
    <select id="boundary_assessment">
      {option_tags(boundaries, boundary if boundary != 'NOT_APPLICABLE' else '')}
    </select>
    <div class="small">
      Do not choose which transcription is “correct” unless the image makes the
      gap obvious. AMBIGUOUS is a valid scientific result.
    </div>
  </div>

  <div{glyph_style}>
    <label>Glyph-disagreement assessment</label>
    <select id="glyph_disagreement_assessment">
      {option_tags(glyph_values, glyph_assessment if glyph_assessment != 'NOT_APPLICABLE' else '')}
    </select>
    <div class="small">
      Do not guess an EVA character. If the distinction is not obvious,
      choose UNRESOLVED.
    </div>
  </div>

  <label>Notes</label>
  <textarea id="reviewer_notes">{html.escape(notes)}</textarea>

  <div class="buttons">
    <button onclick="saveReview({index}, false)">Save</button>
    <button onclick="saveReview({index}, true)">Save + next</button>
    <button onclick="location.href='/?i={prev_i}'">← Previous</button>
    <button onclick="location.href='/?i={next_i}'">Next →</button>
  </div>

  <div id="saved" class="small"></div>
</aside>
</main>

<script>
let imageScale = 1;
document.querySelectorAll('.scan img').forEach(img => {{
  img.addEventListener('click', () => {{
    imageScale = imageScale >= 3 ? 1 : imageScale + 0.5;
    img.style.width = (imageScale * 100) + '%';
  }});
}});

function quickStatus(status) {{
  document.getElementById('review_status').value = status;
  if (status === 'NO_OBVIOUS_QC_PROBLEM') {{
    document.getElementById('locus_alignment').value = 'YES';
  }}
  if (status === 'PARSER_OR_ALIGNMENT_ISSUE') {{
    document.getElementById('locus_alignment').value = 'NO';
  }}
  if (status === 'UNRESOLVED') {{
    if (!document.getElementById('locus_alignment').value) {{
      document.getElementById('locus_alignment').value = 'UNCERTAIN';
    }}
  }}
}}

async function saveReview(index, advance) {{
  const boundary = document.getElementById('boundary_assessment');
  const glyph = document.getElementById('glyph_disagreement_assessment');

  const payload = {{
    review_status: document.getElementById('review_status').value,
    locus_alignment: document.getElementById('locus_alignment').value,
    boundary_assessment: boundary ? boundary.value : 'NOT_APPLICABLE',
    glyph_disagreement_assessment: glyph ? glyph.value : 'NOT_APPLICABLE',
    reviewer_notes: document.getElementById('reviewer_notes').value
  }};

  const response = await fetch('/api/review/' + index, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload)
  }});
  const result = await response.json();

  if (!response.ok) {{
    document.getElementById('saved').textContent = 'ERROR: ' + result.error;
    return;
  }}

  document.getElementById('saved').textContent = 'Saved.';
  if (advance) {{
    location.href = '/?i=' + Math.min(index + 1, {len(store.rows) - 1});
  }}
}}
</script>
</body>
</html>"""


def make_handler(store: TargetedReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_json(
            self,
            obj: dict,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_bytes(
                json.dumps(obj).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)

            if parsed.path == "/":
                query = parse_qs(parsed.query)
                try:
                    index = int(query.get("i", ["0"])[0])
                except ValueError:
                    index = 0
                self.send_bytes(
                    page_html(store, index).encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if parsed.path.startswith("/image/"):
                rel = parsed.path[len("/image/"):]
                try:
                    path = safe_image_path(store, rel)
                    self.send_bytes(path.read_bytes(), "image/jpeg")
                except Exception as exc:
                    self.send_json(
                        {"error": str(exc)},
                        HTTPStatus.NOT_FOUND,
                    )
                return

            if parsed.path == "/api/progress":
                self.send_json(store.progress())
                return

            self.send_json(
                {"error": "not found"},
                HTTPStatus.NOT_FOUND,
            )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("Request too large")

                payload = json.loads(
                    self.rfile.read(length).decode("utf-8")
                    if length
                    else "{}"
                )

                if parsed.path.startswith("/api/review/"):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    store.save(index, payload)
                    self.send_json(
                        {"ok": True, "progress": store.progress()}
                    )
                    return

                self.send_json(
                    {"error": "not found"},
                    HTTPStatus.NOT_FOUND,
                )
            except Exception as exc:
                self.send_json(
                    {"error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 10-case targeted human manuscript-image audit."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser.",
    )
    args = parser.parse_args()

    try:
        store = TargetedReviewStore(args.root)
        server = ThreadingHTTPServer(
            (args.host, args.port),
            make_handler(store),
        )
    except Exception as exc:
        print(
            f"TARGETED IMAGE REVIEW ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    url = f"http://{args.host}:{args.port}/"

    print("=" * 78)
    print("VOYAGER TARGETED MANUSCRIPT-IMAGE AUDIT")
    print("=" * 78)
    print(f"Review URL:        {url}")
    print(f"Frozen cases:      {len(store.rows)}")
    print(f"Already reviewed:  {store.progress()['completed']}")
    print("Validation split:  NOT ACCESSED")
    print("Locked test:       NOT ACCESSED")
    print()
    print("Writes only to:")
    print(f"  {store.output_path}")
    print()
    print("Does NOT modify:")
    print(f"  {store.root / 'loci.csv'}")
    print(f"  {store.root / 'pages.csv'}")
    print()
    print("Press Ctrl+C when finished.")

    if not args.no_browser:
        threading.Timer(
            0.5,
            lambda: webbrowser.open(url),
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTargeted review server stopped.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
