#!/usr/bin/env python3
"""Local human-in-the-loop review app for Voynich manuscript image QC.

This automates review bookkeeping, not paleographic judgment.

Reads:
  results/qc/image_validation/v1/pages.csv
  results/qc/image_validation/v1/loci.csv
  results/qc/image_validation/v1/yale_images/yale_image_map.csv
  results/qc/image_validation/v1/yale_images/images/*.jpg

Provides a browser UI that:
  - presents one sampled locus at a time;
  - shows all Yale scans associated with that folio;
  - shows the ZL3b transcription and metadata;
  - records one-click review status;
  - records glyph / spacing / locus alignment judgments;
  - autosaves notes directly to loci.csv;
  - tracks progress;
  - lets the reviewer finalize page-level status in pages.csv;
  - creates timestamp-free rolling .bak files before each first mutation in
    the current server session.

No validation/test split file is read. The locked test is not accessed.
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
DEFAULT_PORT = 8765

ALLOWED_STATUS = {
    "",
    "PASS",
    "TRANSCRIPTION_DISAGREEMENT",
    "PARSER_OR_ALIGNMENT_ISSUE",
    "UNRESOLVED",
}
ALLOWED_ALIGNMENT = {"", "YES", "NO", "UNCERTAIN"}


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
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.pages_path = self.root / "pages.csv"
        self.loci_path = self.root / "loci.csv"
        self.map_path = self.root / "yale_images" / "yale_image_map.csv"
        self.images_dir = self.root / "yale_images" / "images"

        self.page_fields, self.pages = read_csv(self.pages_path)
        self.locus_fields, self.loci = read_csv(self.loci_path)
        _, self.image_rows = read_csv(self.map_path)

        required_locus = {
            "folio",
            "locus",
            "text_normalized",
            "check_status",
            "glyph_alignment",
            "spacing_alignment",
            "locus_alignment",
            "notes",
        }
        missing = required_locus - set(self.locus_fields)
        if missing:
            raise ValueError(
                "loci.csv missing required fields: "
                + ", ".join(sorted(missing))
            )

        required_page = {"folio", "review_status", "reviewer_notes"}
        missing = required_page - set(self.page_fields)
        if missing:
            raise ValueError(
                "pages.csv missing required fields: "
                + ", ".join(sorted(missing))
            )

        self._lock = threading.Lock()
        self._backed_up = False

    def backup_once(self) -> None:
        if self._backed_up:
            return
        shutil.copy2(self.pages_path, self.pages_path.with_suffix(".csv.bak"))
        shutil.copy2(self.loci_path, self.loci_path.with_suffix(".csv.bak"))
        self._backed_up = True

    def images_for(self, folio: str) -> list[dict]:
        rows = [
            row
            for row in self.image_rows
            if row.get("folio", "") == folio
            and row.get("fetch_status", "") != "UNRESOLVED"
        ]
        return sorted(
            rows,
            key=lambda row: int(row.get("variant_index") or 1),
        )

    def page_for(self, folio: str) -> dict | None:
        for row in self.pages:
            if row.get("folio") == folio:
                return row
        return None

    def page_loci(self, folio: str) -> list[dict]:
        return [row for row in self.loci if row.get("folio") == folio]

    def progress(self) -> dict:
        completed_loci = sum(
            1 for row in self.loci if row.get("check_status", "").strip()
        )
        completed_pages = sum(
            1 for row in self.pages if row.get("review_status", "").strip()
        )
        return {
            "completed_loci": completed_loci,
            "total_loci": len(self.loci),
            "completed_pages": completed_pages,
            "total_pages": len(self.pages),
        }

    def save_locus(self, index: int, payload: dict) -> None:
        if not 0 <= index < len(self.loci):
            raise IndexError("Locus index out of range")

        status = str(payload.get("check_status", "")).strip()
        glyph = str(payload.get("glyph_alignment", "")).strip()
        spacing = str(payload.get("spacing_alignment", "")).strip()
        locus_alignment = str(payload.get("locus_alignment", "")).strip()
        notes = str(payload.get("notes", "")).strip()

        if status not in ALLOWED_STATUS:
            raise ValueError(f"Invalid check_status: {status!r}")
        for name, value in (
            ("glyph_alignment", glyph),
            ("spacing_alignment", spacing),
            ("locus_alignment", locus_alignment),
        ):
            if value not in ALLOWED_ALIGNMENT:
                raise ValueError(f"Invalid {name}: {value!r}")

        with self._lock:
            self.backup_once()
            row = self.loci[index]
            row["check_status"] = status
            row["glyph_alignment"] = glyph
            row["spacing_alignment"] = spacing
            row["locus_alignment"] = locus_alignment
            row["notes"] = notes
            write_csv_atomic(
                self.loci_path,
                self.locus_fields,
                self.loci,
            )

    def save_page(self, folio: str, payload: dict) -> None:
        status = str(payload.get("review_status", "")).strip()
        notes = str(payload.get("reviewer_notes", "")).strip()
        page_alignment = str(payload.get("page_alignment", "")).strip()
        transcription = str(
            payload.get("overall_transcription_agreement", "")
        ).strip()

        if status not in ALLOWED_STATUS:
            raise ValueError(f"Invalid review_status: {status!r}")
        for name, value in (
            ("page_alignment", page_alignment),
            ("overall_transcription_agreement", transcription),
        ):
            if value not in ALLOWED_ALIGNMENT:
                raise ValueError(f"Invalid {name}: {value!r}")

        with self._lock:
            self.backup_once()
            page = self.page_for(folio)
            if page is None:
                raise KeyError(f"Unknown folio: {folio}")
            page["review_status"] = status
            page["page_alignment"] = page_alignment
            page["overall_transcription_agreement"] = transcription
            page["reviewer_notes"] = notes
            write_csv_atomic(
                self.pages_path,
                self.page_fields,
                self.pages,
            )

    def all_page_loci_reviewed(self, folio: str) -> bool:
        rows = self.page_loci(folio)
        return bool(rows) and all(
            row.get("check_status", "").strip()
            for row in rows
        )

    def suggested_page_status(self, folio: str) -> str:
        rows = self.page_loci(folio)
        statuses = {
            row.get("check_status", "").strip()
            for row in rows
            if row.get("check_status", "").strip()
        }
        if not self.all_page_loci_reviewed(folio):
            return ""
        if "PARSER_OR_ALIGNMENT_ISSUE" in statuses:
            return "PARSER_OR_ALIGNMENT_ISSUE"
        if "UNRESOLVED" in statuses:
            return "UNRESOLVED"
        if "TRANSCRIPTION_DISAGREEMENT" in statuses:
            return "TRANSCRIPTION_DISAGREEMENT"
        return "PASS"


def safe_image_path(store: ReviewStore, rel: str) -> Path:
    # yale_image_map.csv stores paths like images/f41r.jpg.
    rel_path = Path(unquote(rel))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("Unsafe image path")
    path = (store.root / "yale_images" / rel_path).resolve()
    allowed = (store.root / "yale_images").resolve()
    if allowed not in path.parents:
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


def page_html(store: ReviewStore, locus_index: int) -> str:
    locus_index = max(0, min(locus_index, len(store.loci) - 1))
    row = store.loci[locus_index]
    folio = row["folio"]
    page = store.page_for(folio) or {}
    images = store.images_for(folio)
    progress = store.progress()

    status = row.get("check_status", "")
    glyph = row.get("glyph_alignment", "")
    spacing = row.get("spacing_alignment", "")
    locus_alignment = row.get("locus_alignment", "")
    notes = row.get("notes", "")

    image_blocks = []
    for img in images:
        local = img.get("local_image", "")
        label = img.get("canvas_label", "")
        canvas = img.get("canvas_id", "")
        source = img.get("image_url", "")
        if local:
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

    statuses = [
        "",
        "PASS",
        "TRANSCRIPTION_DISAGREEMENT",
        "PARSER_OR_ALIGNMENT_ISSUE",
        "UNRESOLVED",
    ]
    alignments = ["", "YES", "NO", "UNCERTAIN"]

    prev_i = max(0, locus_index - 1)
    next_i = min(len(store.loci) - 1, locus_index + 1)

    page_done = store.all_page_loci_reviewed(folio)
    suggested = store.suggested_page_status(folio)
    page_status = page.get("review_status", "")

    locus_rows_for_page = store.page_loci(folio)
    page_completed_count = sum(
        1 for x in locus_rows_for_page if x.get("check_status", "").strip()
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VOYAGER image QC — {html.escape(folio)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  margin: 0;
}}
header {{
  position: sticky; top: 0; z-index: 20;
  padding: .7rem 1rem;
  background: Canvas;
  border-bottom: 1px solid GrayText;
  display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
}}
main {{
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(340px, 1fr);
  gap: 1rem;
  padding: 1rem;
}}
.scans {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1rem;
}}
.scan {{
  border: 1px solid GrayText; padding: .5rem;
}}
.scan-title {{ font-weight: 700; margin-bottom: .4rem; }}
.zoom-wrap {{
  height: 75vh;
  overflow: auto;
  background: #222;
}}
.scan img {{
  width: 100%;
  height: auto;
  cursor: zoom-in;
  transform-origin: top left;
}}
.links {{
  display: flex; gap: 1rem; margin-top: .4rem;
}}
.panel {{
  position: sticky; top: 5rem; align-self: start;
  border: 1px solid GrayText;
  padding: 1rem;
  border-radius: .5rem;
}}
.transcription {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  font-size: 1.1rem;
  padding: 1rem;
  border: 1px solid GrayText;
  margin: .8rem 0;
}}
label {{ display: block; margin-top: .8rem; font-weight: 600; }}
select, textarea, button {{
  width: 100%; box-sizing: border-box;
  padding: .55rem; font: inherit;
}}
textarea {{ min-height: 90px; }}
.buttons {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;
  margin-top: .8rem;
}}
.status-buttons {{
  display: grid; grid-template-columns: 1fr 1fr; gap: .4rem;
  margin: .8rem 0;
}}
.status-buttons button {{ min-height: 3rem; }}
.progress {{ font-weight: 700; }}
.saved {{ min-height: 1.4rem; margin-top: .5rem; }}
.page-box {{
  margin-top: 1.2rem;
  border-top: 1px solid GrayText;
  padding-top: 1rem;
}}
.small {{ font-size: .9rem; opacity: .8; }}
kbd {{
  border: 1px solid GrayText; border-radius: .25rem;
  padding: .1rem .35rem;
}}
@media (max-width: 1000px) {{
  main {{ grid-template-columns: 1fr; }}
  .panel {{ position: static; }}
}}
</style>
</head>
<body>
<header>
  <div class="progress">
    Locus {locus_index + 1}/{len(store.loci)}
    · reviewed {progress['completed_loci']}/{progress['total_loci']}
    · pages {progress['completed_pages']}/{progress['total_pages']}
  </div>
  <div>
    Folio <b>{html.escape(folio)}</b>
    · page loci {page_completed_count}/{len(locus_rows_for_page)}
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
    Sample #{html.escape(row.get('sample_index', ''))}
    · locus check #{html.escape(row.get('locus_check_index', ''))}
  </div>
  <h2>{html.escape(row.get('locus', ''))}</h2>
  <div class="transcription">{html.escape(row.get('text_normalized', ''))}</div>

  <div class="small">
    source line {html.escape(row.get('source_line_number', ''))}
    · uncertain spaces {html.escape(row.get('uncertain_spaces', '0'))}
    · alternatives {html.escape(row.get('alternative_readings', '0'))}
    · unknown ? {html.escape(row.get('unknown_question_marks', '0'))}
  </div>

  <div class="status-buttons">
    <button onclick="quickStatus('PASS')">✓ PASS</button>
    <button onclick="quickStatus('TRANSCRIPTION_DISAGREEMENT')">≠ TRANSCRIPTION</button>
    <button onclick="quickStatus('PARSER_OR_ALIGNMENT_ISSUE')">⚠ PARSER/ALIGNMENT</button>
    <button onclick="quickStatus('UNRESOLVED')">? UNRESOLVED</button>
  </div>

  <label>Status</label>
  <select id="check_status">
    {option_tags(statuses, status)}
  </select>

  <label>Glyph alignment</label>
  <select id="glyph_alignment">
    {option_tags(alignments, glyph)}
  </select>

  <label>Spacing alignment</label>
  <select id="spacing_alignment">
    {option_tags(alignments, spacing)}
  </select>

  <label>Locus alignment</label>
  <select id="locus_alignment">
    {option_tags(alignments, locus_alignment)}
  </select>

  <label>Notes</label>
  <textarea id="notes">{html.escape(notes)}</textarea>

  <div class="buttons">
    <button onclick="saveLocus({locus_index}, false)">Save</button>
    <button onclick="saveLocus({locus_index}, true)">Save + next</button>
    <button onclick="location.href='/?i={prev_i}'">← Previous</button>
    <button onclick="location.href='/?i={next_i}'">Next →</button>
  </div>
  <div class="saved" id="saved"></div>

  <div class="small">
    Keyboard: <kbd>1</kbd> PASS · <kbd>2</kbd> transcription disagreement ·
    <kbd>3</kbd> parser/alignment issue · <kbd>4</kbd> unresolved ·
    <kbd>Enter</kbd> save + next
  </div>

  <div class="page-box">
    <h3>Finalize {html.escape(folio)}</h3>
    <div class="small">
      All sampled loci reviewed: <b>{'YES' if page_done else 'NO'}</b><br>
      Suggested page status: <b>{html.escape(suggested or '—')}</b>
    </div>

    <label>Page review status</label>
    <select id="page_status">
      {option_tags(statuses, page_status)}
    </select>

    <label>Page alignment</label>
    <select id="page_alignment">
      {option_tags(alignments, page.get('page_alignment', ''))}
    </select>

    <label>Overall transcription agreement</label>
    <select id="page_transcription">
      {option_tags(alignments, page.get('overall_transcription_agreement', ''))}
    </select>

    <label>Page notes</label>
    <textarea id="page_notes">{html.escape(page.get('reviewer_notes', ''))}</textarea>
    <button onclick="savePage('{html.escape(folio)}')">Save page review</button>
  </div>
</aside>
</main>

<script>
let imageScale = 1;
document.querySelectorAll('.scan img').forEach(img => {{
  img.addEventListener('click', () => {{
    imageScale = imageScale >= 2.5 ? 1 : imageScale + 0.5;
    img.style.width = (imageScale * 100) + '%';
  }});
}});

function quickStatus(status) {{
  document.getElementById('check_status').value = status;
  if (status === 'PASS') {{
    document.getElementById('glyph_alignment').value = 'YES';
    document.getElementById('spacing_alignment').value = 'YES';
    document.getElementById('locus_alignment').value = 'YES';
  }}
}}

async function saveLocus(index, advance) {{
  const payload = {{
    check_status: document.getElementById('check_status').value,
    glyph_alignment: document.getElementById('glyph_alignment').value,
    spacing_alignment: document.getElementById('spacing_alignment').value,
    locus_alignment: document.getElementById('locus_alignment').value,
    notes: document.getElementById('notes').value
  }};
  const response = await fetch('/api/locus/' + index, {{
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
    location.href = '/?i=' + Math.min(index + 1, {len(store.loci) - 1});
  }}
}}

async function savePage(folio) {{
  const payload = {{
    review_status: document.getElementById('page_status').value,
    page_alignment: document.getElementById('page_alignment').value,
    overall_transcription_agreement: document.getElementById('page_transcription').value,
    reviewer_notes: document.getElementById('page_notes').value
  }};
  const response = await fetch('/api/page/' + encodeURIComponent(folio), {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload)
  }});
  const result = await response.json();
  document.getElementById('saved').textContent =
    response.ok ? 'Page review saved.' : 'ERROR: ' + result.error;
}}

document.addEventListener('keydown', event => {{
  if (event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {{
    return;
  }}
  if (event.key === '1') quickStatus('PASS');
  if (event.key === '2') quickStatus('TRANSCRIPTION_DISAGREEMENT');
  if (event.key === '3') quickStatus('PARSER_OR_ALIGNMENT_ISSUE');
  if (event.key === '4') quickStatus('UNRESOLVED');
  if (event.key === 'Enter') saveLocus({locus_index}, true);
}});
</script>
</body>
</html>"""


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # Keep terminal output quiet except errors.
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
                payload = page_html(store, index).encode("utf-8")
                self.send_bytes(
                    payload,
                    "text/html; charset=utf-8",
                )
                return

            if parsed.path.startswith("/image/"):
                rel = parsed.path[len("/image/"):]
                try:
                    path = safe_image_path(store, rel)
                    payload = path.read_bytes()
                    self.send_bytes(payload, "image/jpeg")
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

                if parsed.path.startswith("/api/locus/"):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    store.save_locus(index, payload)
                    self.send_json(
                        {
                            "ok": True,
                            "progress": store.progress(),
                        }
                    )
                    return

                if parsed.path.startswith("/api/page/"):
                    folio = unquote(
                        parsed.path[len("/api/page/"):]
                    )
                    store.save_page(folio, payload)
                    self.send_json(
                        {
                            "ok": True,
                            "progress": store.progress(),
                        }
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
            "Run a local human-in-the-loop reviewer for the frozen "
            "Voynich manuscript-image QC sample."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the review app in a browser.",
    )
    args = parser.parse_args()

    try:
        store = ReviewStore(args.root)
        handler = make_handler(store)
        server = ThreadingHTTPServer(
            (args.host, args.port),
            handler,
        )
    except Exception as exc:
        print(f"IMAGE REVIEW APP ERROR: {exc}", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{args.port}/"
    print("=" * 76)
    print("VOYAGER MANUSCRIPT-IMAGE REVIEW APP")
    print("=" * 76)
    print(f"Review URL:   {url}")
    print(f"Loci:         {len(store.loci)}")
    print(f"Pages:        {len(store.pages)}")
    print("Validation:   NOT ACCESSED")
    print("Locked test:  NOT ACCESSED")
    print()
    print("The app writes directly to:")
    print(f"  {store.loci_path}")
    print(f"  {store.pages_path}")
    print()
    print("Backups are created on the first saved judgment:")
    print(f"  {store.loci_path}.bak")
    print(f"  {store.pages_path}.bak")
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
        print("\nReview server stopped.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
