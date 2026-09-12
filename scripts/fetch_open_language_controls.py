#!/usr/bin/env python3
"""Fetch and freeze open historical-language controls for VOYAGER.

This replaces the earlier live-Wikisource acquisition path with fixed,
versioned open corpora:

- Latin: UD Latin-UDante, UD release 2.18, CC BY-NC-SA 3.0
- Italian: UD Italian-Old, UD release 2.18, CC BY-SA 4.0
- German: Referenzkorpus Mittelhochdeutsch (ReM) v2.1, CC BY-SA 4.0
- French: UD Old French-PROFITEROLE, UD release 2.18, CC BY-NC-SA 3.0

The script freezes exactly 20,000 normalized alphabetic tokens per source.
It does not read any Voynich validation or locked-test file.

Default output:
    data/controls/languages_open_v1/

A successful v1 freeze is write-once. Re-running verifies hashes and performs
no network requests. Interrupted runs can resume safely because all upstream
resources are version-pinned.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import md5, sha256
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import unicodedata
import xml.etree.ElementTree as ET
import zipfile


SCHEMA_VERSION = "1.0"
FREEZE_ID = "open-language-controls-v1"
TARGET_TOKENS = 20_000

DEFAULT_OUTPUT_ROOT = Path("data/controls/languages_open_v1")
DEFAULT_DOWNLOAD_ROOT = Path("data/controls/downloads/open_language_v1")
DEFAULT_DATA_MANIFEST = Path("data/manifest.yaml")

USER_AGENT = (
    "VOYAGER-Voynich-Research/1.0 "
    "(open historical language control acquisition)"
)
REQUEST_DELAY_SECONDS = 0.5
MAX_HTTP_ATTEMPTS = 7
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

UD_RELEASE = "2.18"
UD_TAG = "r2.18"
UD_RELEASE_DATE = "2026-05-15"

# The ReM file MD5 is published on the Zenodo v2.1 record.
REM_TEI_MD5 = "c57828a32f2634f3c5e72c3009e958a2"

SOURCES = (
    {
        "source_id": "latin_udante_ud218",
        "language": "Latin",
        "language_code": "la",
        "kind": "ud_conllu",
        "corpus": "UD Latin-UDante",
        "version": f"UD {UD_RELEASE}",
        "date_scope": "14th-century literary Medieval Latin",
        "description": (
            "Latin works associated with Dante Alighieri; UDante describes "
            "itself as literary Medieval Latin (XIVth century)."
        ),
        "license": "CC BY-NC-SA 3.0",
        "homepage": "https://universaldependencies.org/treebanks/la_udante/index.html",
        "repository": "https://github.com/UniversalDependencies/UD_Latin-UDante",
        "files": (
            {
                "name": "la_udante-ud-train.conllu",
                "url": (
                    "https://raw.githubusercontent.com/UniversalDependencies/"
                    "UD_Latin-UDante/r2.18/la_udante-ud-train.conllu"
                ),
            },
        ),
        "limitations": (
            "UD treebank tokenization/edition rather than a diplomatic manuscript transcription.",
            "Only the fixed UD train file is sampled; this is a structural language control, not a complete edition.",
        ),
    },
    {
        "source_id": "italian_old_ud218",
        "language": "Italian",
        "language_code": "it",
        "kind": "ud_conllu",
        "corpus": "UD Italian-Old",
        "version": f"UD {UD_RELEASE}",
        "date_scope": "Old Italian / Florentine, Dante's Commedia (c. 1306-1321)",
        "description": (
            "Old Italian treebank containing Dante Alighieri's Commedia, "
            "based on the Petrocchi edition via DanteSearch."
        ),
        "license": "CC BY-SA 4.0",
        "homepage": "https://universaldependencies.org/treebanks/it_old/index.html",
        "repository": "https://github.com/UniversalDependencies/UD_Italian-Old",
        "files": (
            {
                "name": "it_old-ud-train.conllu",
                "url": (
                    "https://raw.githubusercontent.com/UniversalDependencies/"
                    "UD_Italian-Old/r2.18/it_old-ud-train.conllu"
                ),
            },
        ),
        "limitations": (
            "Critical/edited treebank text rather than a diplomatic manuscript transcription.",
            "Chronologically earlier than the Voynich manuscript.",
        ),
    },
    {
        "source_id": "german_rem_v21",
        "language": "German",
        "language_code": "gmh",
        "kind": "rem_tei_zip",
        "corpus": "Referenzkorpus Mittelhochdeutsch (ReM)",
        "version": "2.1",
        "date_scope": "Middle High German, 1050-1350",
        "description": (
            "Reference Corpus of Middle High German: diplomatically "
            "transcribed and annotated texts, around two million word forms."
        ),
        "license": "CC BY-SA 4.0",
        "homepage": "https://www.linguistics.rub.de/rem/access/index.en.html",
        "repository": "https://zenodo.org/records/13982324",
        "doi": "10.5281/zenodo.13982324",
        "files": (
            {
                "name": "ReM-v2.1_tei.zip",
                "url": (
                    "https://zenodo.org/records/13982324/files/"
                    "ReM-v2.1_tei.zip?download=1"
                ),
                "expected_md5": REM_TEI_MD5,
            },
        ),
        "limitations": (
            "Chronologically broader/earlier than the Voynich manuscript.",
            "Deterministic archive-member order is used to select the first 20,000 normalized tokens.",
            "TEI line/sentence boundaries are used when present; they are not claimed equivalent to Voynich manuscript loci.",
        ),
    },
    {
        "source_id": "french_profiterole_ud218",
        "language": "French",
        "language_code": "fro",
        "kind": "ud_conllu",
        "corpus": "UD Old French-PROFITEROLE",
        "version": f"UD {UD_RELEASE}",
        "date_scope": "Old French, 9th-13th centuries",
        "description": (
            "Old French PROFITEROLE treebank containing twelve medieval texts."
        ),
        "license": "CC BY-NC-SA 3.0",
        "homepage": "https://universaldependencies.org/treebanks/fro_profiterole/index.html",
        "repository": "https://github.com/UniversalDependencies/UD_Old_French-PROFITEROLE",
        "files": (
            {
                "name": "fro_profiterole-ud-train.conllu",
                "url": (
                    "https://raw.githubusercontent.com/UniversalDependencies/"
                    "UD_Old_French-PROFITEROLE/r2.18/"
                    "fro_profiterole-ud-train.conllu"
                ),
            },
        ),
        "limitations": (
            "Chronologically earlier than the Voynich manuscript.",
            "UD treebank representation rather than a diplomatic manuscript transcription.",
        ),
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, destination: Path) -> dict:
    """Download once with retry/backoff; reuse an existing completed file."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused_existing_download": True,
        }

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    last_error: Optional[BaseException] = None

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
            },
        )

        try:
            with urlopen(request, timeout=120) as response, partial.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)

            partial.replace(destination)
            time.sleep(REQUEST_DELAY_SECONDS)

            return {
                "url": url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "reused_existing_download": False,
            }

        except HTTPError as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if exc.code not in RETRYABLE_STATUS or attempt >= MAX_HTTP_ATTEMPTS:
                raise

            retry_after = exc.headers.get("Retry-After")
            wait = 0.0
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = 0.0
            if wait <= 0:
                wait = min(120.0, 5.0 * (2 ** (attempt - 1)))
            else:
                wait = min(wait, 300.0)

            print(
                f"  HTTP {exc.code}; retry {attempt + 1}/{MAX_HTTP_ATTEMPTS} "
                f"in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)

        except URLError as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt >= MAX_HTTP_ATTEMPTS:
                raise
            wait = min(120.0, 5.0 * (2 ** (attempt - 1)))
            print(
                f"  network error; retry {attempt + 1}/{MAX_HTTP_ATTEMPTS} "
                f"in {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Download failed without an exception: {url}")


def normalize_form(value: str) -> Tuple[str, ...]:
    """Apply the frozen minimal normalization to one upstream token form."""
    value = unicodedata.normalize("NFKC", value).lower()
    tokens: List[str] = []
    current: List[str] = []

    for char in value:
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tuple(tokens)


def crop_lines(lines: Iterable[Sequence[str]], target: int) -> Tuple[List[Tuple[str, ...]], int]:
    """Take exactly target tokens while preserving source line/sentence order."""
    output: List[Tuple[str, ...]] = []
    total = 0

    for line in lines:
        clean = tuple(token for token in line if token)
        if not clean:
            continue

        remaining = target - total
        if remaining <= 0:
            break

        selected = clean[:remaining]
        if selected:
            output.append(tuple(selected))
            total += len(selected)

        if total == target:
            break

    return output, total


def iter_conllu_sentences(path: Path) -> Iterator[Tuple[str, ...]]:
    """Yield normalized alphabetic tokens sentence by sentence from CoNLL-U."""
    current: List[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")

            if not line:
                if current:
                    yield tuple(current)
                    current = []
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 4:
                raise ValueError(f"Malformed CoNLL-U row in {path}: {line!r}")

            token_id = fields[0]

            # Skip multiword-token range rows (e.g. 1-2) and empty nodes (1.1).
            if not token_id.isdigit():
                continue

            form = fields[1]
            for token in normalize_form(form):
                current.append(token)

    if current:
        yield tuple(current)


def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


REM_TOKEN_TAGS = {"w", "word", "tok", "token"}
REM_BOUNDARY_TAGS = {"s", "sentence", "l", "line", "lb", "p"}


def iter_rem_tei_lines(xml_bytes: bytes) -> Iterator[Tuple[str, ...]]:
    """Yield token groups from one ReM TEI XML document.

    TEI-compatible corpora conventionally encode word tokens with <w>.
    A few common token aliases are accepted defensively. Boundaries are
    flushed at sentence/line/paragraph markers where available.
    """
    current: List[str] = []
    seen_token_elements = 0

    stream = io.BytesIO(xml_bytes)

    for event, elem in ET.iterparse(stream, events=("start", "end")):
        name = _localname(elem.tag).lower()

        if event == "end" and name in REM_TOKEN_TAGS:
            seen_token_elements += 1
            raw = "".join(elem.itertext())
            for token in normalize_form(raw):
                current.append(token)
            elem.clear()
            continue

        if event == "end" and name in REM_BOUNDARY_TAGS:
            if current:
                yield tuple(current)
                current = []
            elem.clear()

    if current:
        yield tuple(current)

    if seen_token_elements == 0:
        raise ValueError(
            "ReM TEI parser found no <w>/<word>/<tok>/<token> elements. "
            "The upstream TEI schema may have changed; do not fall back to "
            "untested extraction silently."
        )


def iter_rem_archive_lines(path: Path) -> Iterator[Tuple[str, ...]]:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml")
            and not name.startswith("__MACOSX/")
        )

        if not members:
            raise ValueError("ReM TEI archive contains no XML files")

        for member in members:
            data = archive.read(member)
            yield from iter_rem_tei_lines(data)


def source_registry_payload() -> dict:
    def jsonable(source: Mapping[str, object]) -> dict:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in source.items()
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "target_normalized_tokens_per_source": TARGET_TOKENS,
        "normalization": {
            "unicode_normalization": "NFKC",
            "case": "lower",
            "token_definition": "contiguous_unicode_alphabetic_characters",
            "punctuation_digits": "separators",
            "accent_stripping": False,
            "language_specific_modernization": False,
        },
        "ud_release": UD_RELEASE,
        "ud_tag": UD_TAG,
        "ud_release_date": UD_RELEASE_DATE,
        "sources": [jsonable(source) for source in SOURCES],
    }


def registry_sha256() -> str:
    payload = json.dumps(
        source_registry_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def write_source_text(path: Path, lines: Sequence[Sequence[str]]) -> None:
    path.write_text(
        "\n".join(" ".join(line) for line in lines) + "\n",
        encoding="utf-8",
    )


def verify_frozen_source(
    source: Mapping[str, object],
    *,
    output_root: Path,
    expected_registry_sha: str,
) -> Optional[dict]:
    source_dir = output_root / str(source["source_id"])
    source_path = source_dir / "source.txt"
    manifest_path = source_dir / "manifest.json"

    if not source_dir.exists():
        return None

    if not source_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"Incomplete frozen source directory: {source_dir}. "
            "Remove only that directory and retry."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("registry_sha256") != expected_registry_sha:
        raise ValueError(
            f"{source['source_id']}: registry hash differs from this script. "
            "Create a new freeze version rather than overwriting."
        )

    observed_hash = sha256_file(source_path)
    if observed_hash != manifest.get("source_sha256"):
        raise ValueError(
            f"{source['source_id']}: source.txt SHA-256 verification failed"
        )

    observed_tokens = sum(
        len(line.split())
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if observed_tokens != TARGET_TOKENS:
        raise ValueError(
            f"{source['source_id']}: expected {TARGET_TOKENS:,} tokens; "
            f"found {observed_tokens:,}"
        )

    return manifest


def freeze_source(
    source: Mapping[str, object],
    *,
    output_root: Path,
    download_root: Path,
    registry_sha: str,
) -> dict:
    source_id = str(source["source_id"])
    source_dir = output_root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    local_files: List[Path] = []

    for file_spec in source["files"]:
        filename = str(file_spec["name"])
        destination = download_root / source_id / filename
        record = download_file(str(file_spec["url"]), destination)

        expected_md5 = file_spec.get("expected_md5")
        if expected_md5 is not None:
            observed_md5 = md5_file(destination)
            if observed_md5 != expected_md5:
                raise ValueError(
                    f"{source_id}: MD5 mismatch for {filename}: "
                    f"{observed_md5} != {expected_md5}"
                )
            record["md5"] = observed_md5
            record["expected_md5"] = expected_md5

        downloaded.append(record)
        local_files.append(destination)

    if source["kind"] == "ud_conllu":
        line_iter = (
            line
            for path in local_files
            for line in iter_conllu_sentences(path)
        )
    elif source["kind"] == "rem_tei_zip":
        if len(local_files) != 1:
            raise ValueError("ReM v2.1 expects exactly one TEI archive")
        line_iter = iter_rem_archive_lines(local_files[0])
    else:
        raise ValueError(f"Unknown source kind: {source['kind']!r}")

    frozen_lines, count = crop_lines(line_iter, TARGET_TOKENS)

    if count != TARGET_TOKENS:
        raise ValueError(
            f"{source_id}: only {count:,} normalized tokens were available; "
            f"need exactly {TARGET_TOKENS:,}"
        )

    source_path = source_dir / "source.txt"
    write_source_text(source_path, frozen_lines)

    upstream_manifest_path = source_dir / "upstream_files.json"
    upstream_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "files": downloaded,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": FREEZE_ID,
        "registry_sha256": registry_sha,
        "source_id": source_id,
        "language": source["language"],
        "language_code": source["language_code"],
        "corpus": source["corpus"],
        "upstream_version": source["version"],
        "date_scope": source["date_scope"],
        "description": source["description"],
        "license": source["license"],
        "homepage": source["homepage"],
        "repository": source["repository"],
        "doi": source.get("doi"),
        "limitations": list(source["limitations"]),
        "normalization": source_registry_payload()["normalization"],
        "selection_rule": (
            "first 20,000 normalized alphabetic tokens encountered in the "
            "version-pinned upstream resource order"
        ),
        "frozen_normalized_tokens": count,
        "frozen_lines": len(frozen_lines),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "upstream_files_path": str(upstream_manifest_path),
        "upstream_files_sha256": sha256_file(upstream_manifest_path),
        "retrieved_utc": utc_now(),
        "voynich_validation_accessed": False,
        "voynich_locked_test_accessed": False,
    }

    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_paths = (source_path, upstream_manifest_path, manifest_path)
    (source_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def append_project_manifest(
    data_manifest: Path,
    *,
    master_manifest: Path,
    manifests: Sequence[Mapping[str, object]],
) -> None:
    """Append a versioned YAML block without touching an earlier control block."""
    if not data_manifest.is_file():
        print(
            f"NOTE: {data_manifest} not found; project-manifest registration skipped.",
            file=sys.stderr,
        )
        return

    text = data_manifest.read_text(encoding="utf-8")
    key = "historical_language_controls_open_v1:"

    if key in text:
        return

    lines = [
        "",
        "# Versioned open historical-language controls.",
        "historical_language_controls_open_v1:",
        f'  freeze_id: "{FREEZE_ID}"',
        f"  target_normalized_tokens_per_source: {TARGET_TOKENS}",
        f'  master_manifest: "{master_manifest.as_posix()}"',
        "  sources:",
    ]

    for manifest in manifests:
        lines.extend(
            [
                f'    - id: "{manifest["source_id"]}"',
                f'      language: "{manifest["language"]}"',
                f'      corpus: "{manifest["corpus"]}"',
                f'      version: "{manifest["upstream_version"]}"',
                f'      license: "{manifest["license"]}"',
                f'      local_path: "{manifest["source_path"]}"',
                f'      sha256: "{manifest["source_sha256"]}"',
            ]
        )

    data_manifest.write_text(
        text.rstrip() + "\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def verify_complete_freeze(output_root: Path, registry_sha: str) -> Optional[dict]:
    master_path = output_root / "language_controls_open_v1_manifest.json"
    if not master_path.is_file():
        return None

    master = json.loads(master_path.read_text(encoding="utf-8"))

    if master.get("freeze_id") != FREEZE_ID:
        raise ValueError("Existing master manifest has the wrong freeze_id")
    if master.get("registry_sha256") != registry_sha:
        raise ValueError(
            "Existing open-language v1 master manifest was created by a "
            "different source registry. Create v2 instead of overwriting."
        )

    for source in SOURCES:
        manifest = verify_frozen_source(
            source,
            output_root=output_root,
            expected_registry_sha=registry_sha,
        )
        if manifest is None:
            raise ValueError(
                f"Master manifest exists but source is missing: "
                f"{source['source_id']}"
            )

    return master


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and freeze four versioned open historical-language "
            "control corpora without using the Wikisource API."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=DEFAULT_DATA_MANIFEST,
    )
    args = parser.parse_args()

    try:
        registry_sha = registry_sha256()
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.download_root.mkdir(parents=True, exist_ok=True)

        existing_master = verify_complete_freeze(args.output_root, registry_sha)
        if existing_master is not None:
            print("=" * 92)
            print("OPEN HISTORICAL LANGUAGE CONTROLS v1 — ALREADY FROZEN")
            print("=" * 92)
            print(f"Freeze ID:      {FREEZE_ID}")
            print(f"Registry SHA:   {registry_sha}")
            print("Sources:        4/4 verified")
            print("Tokens/source:  20,000")
            print("Network:        NOT ACCESSED")
            print("Voynich test:   NOT ACCESSED")
            return 0

        manifests = []

        print("=" * 92)
        print("FREEZE VERSIONED OPEN HISTORICAL LANGUAGE CONTROLS")
        print("=" * 92)
        print(f"Freeze ID:      {FREEZE_ID}")
        print(f"Registry SHA:   {registry_sha}")
        print(f"Tokens/source:  {TARGET_TOKENS:,}")
        print(f"UD release:     {UD_RELEASE} ({UD_RELEASE_DATE})")
        print("German corpus:  ReM 2.1 / Zenodo 13982324")
        print("Wikisource API: NOT USED")
        print("Voynich test:   NOT ACCESSED")
        print()

        for source in SOURCES:
            print(f"[{source['language']}] {source['source_id']}")

            existing = verify_frozen_source(
                source,
                output_root=args.output_root,
                expected_registry_sha=registry_sha,
            )
            if existing is not None:
                manifest = existing
                print("  existing partial freeze: VERIFIED; reused")
            else:
                manifest = freeze_source(
                    source,
                    output_root=args.output_root,
                    download_root=args.download_root,
                    registry_sha=registry_sha,
                )

            manifests.append(manifest)
            print(
                f"  tokens={manifest['frozen_normalized_tokens']:,}  "
                f"lines={manifest['frozen_lines']:,}"
            )
            print(f"  SHA256={manifest['source_sha256']}")

        if len(manifests) != 4:
            raise RuntimeError("Expected exactly four frozen source manifests")

        registry_path = args.output_root / "source_registry.json"
        registry_path.write_text(
            json.dumps(
                source_registry_payload(),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        master = {
            "schema_version": SCHEMA_VERSION,
            "freeze_id": FREEZE_ID,
            "registry_sha256": registry_sha,
            "source_registry_path": str(registry_path),
            "source_registry_file_sha256": sha256_file(registry_path),
            "target_normalized_tokens_per_source": TARGET_TOKENS,
            "sources": [
                {
                    "source_id": manifest["source_id"],
                    "language": manifest["language"],
                    "corpus": manifest["corpus"],
                    "upstream_version": manifest["upstream_version"],
                    "license": manifest["license"],
                    "source_path": manifest["source_path"],
                    "source_sha256": manifest["source_sha256"],
                    "manifest_path": str(
                        args.output_root
                        / manifest["source_id"]
                        / "manifest.json"
                    ),
                }
                for manifest in manifests
            ],
            "voynich_validation_accessed": False,
            "voynich_locked_test_accessed": False,
        }

        master_path = args.output_root / "language_controls_open_v1_manifest.json"
        master_path.write_text(
            json.dumps(master, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        append_project_manifest(
            args.data_manifest,
            master_manifest=master_path,
            manifests=manifests,
        )

        top_checksums = [registry_path, master_path]
        (args.output_root / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in top_checksums
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 92)
        print("OPEN HISTORICAL LANGUAGE CONTROL FREEZE COMPLETE")
        print("=" * 92)
        print("Sources:        4")
        print("Tokens/source:  20,000")
        print(f"Master:         {master_path}")
        print(f"Downloads:      {args.download_root}")
        print("Wikisource API: NOT USED")
        print("Voynich validation/test: NOT ACCESSED")
        print()
        print(
            "Commit the frozen source.txt files + manifests before "
            "generating cipher controls."
        )
        return 0

    except Exception as exc:
        print(
            f"OPEN-LANGUAGE FREEZE ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
