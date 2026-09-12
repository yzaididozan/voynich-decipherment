#!/usr/bin/env python3
"""Fetch, normalize, and freeze the four historical-language control sources.

The script uses the MediaWiki API for Wikisource. It records the exact page
revision IDs used, extracts readable text from page HTML, applies the same
minimal Unicode-letter tokenization used by the historical cipher-control
generator, and freezes exactly the configured number of tokens per source.

No Voynich validation or test file is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_REGISTRY = Path("configs/corpus/language_controls_v1.json")
DEFAULT_OUTPUT = Path("data/controls/languages")
DEFAULT_DATA_MANIFEST = Path("data/manifest.yaml")
USER_AGENT = (
    "VOYAGER-Voynich-Research/1.0 "
    "(historical language-control corpus acquisition)"
)


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def api_get(endpoint: str, params: Mapping[str, object]) -> dict:
    query = urlencode(
        {key: str(value) for key, value in params.items()}
    )
    url = f"{endpoint}?{query}"
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


class LinkCollector(HTMLParser):
    """Collect ordered /wiki/... links from a parsed Wikisource page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        if not href or not href.startswith("/wiki/"):
            return

        title = unquote(href[len("/wiki/"):].split("#", 1)[0])
        title = title.replace("_", " ")
        if title:
            self.links.append(title)


class TextExtractor(HTMLParser):
    """Extract prose/verse body text while excluding common apparatus.

    We capture text only inside paragraph/list/poem-like blocks. This avoids
    most Wikisource header metadata and navigation without requiring
    language-specific selectors.
    """

    CAPTURE_TAGS = {"p", "li", "dd", "blockquote", "pre"}
    BLOCK_BREAK_TAGS = {
        "p", "li", "dd", "blockquote", "pre", "div", "br",
    }
    SKIP_TAGS = {"script", "style", "table"}

    SKIP_CLASS_FRAGMENTS = (
        "mw-editsection",
        "reference",
        "references",
        "navbox",
        "ws-noexport",
        "pagenum",
        "pagequality",
        "mw-collapsible",
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.capture_depth = 0
        self.skip_depth = 0
        self._stack: List[Tuple[str, bool, bool]] = []
        self.parts: List[str] = []

    def _skip_for_attrs(self, attrs) -> bool:
        values = dict(attrs)
        classes = values.get("class", "")
        return any(
            fragment in classes
            for fragment in self.SKIP_CLASS_FRAGMENTS
        )

    def handle_starttag(self, tag: str, attrs) -> None:
        inherited_skip = self.skip_depth > 0
        starts_skip = (
            tag in self.SKIP_TAGS
            or self._skip_for_attrs(attrs)
        )
        if starts_skip:
            self.skip_depth += 1

        starts_capture = (
            tag in self.CAPTURE_TAGS
            and not inherited_skip
            and not starts_skip
        )
        if starts_capture:
            self.capture_depth += 1

        self._stack.append(
            (tag, starts_skip, starts_capture)
        )

        if (
            tag == "br"
            and self.capture_depth > 0
            and self.skip_depth == 0
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        # HTML from MediaWiki is well-enough nested for a stack unwind.
        if not self._stack:
            return

        # Find the most recent matching tag.
        index = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                index = i
                break
        if index is None:
            return

        trailing = self._stack[index:]
        self._stack = self._stack[:index]

        for _, started_skip, started_capture in reversed(trailing):
            if started_capture:
                self.capture_depth = max(
                    0, self.capture_depth - 1
                )
            if started_skip:
                self.skip_depth = max(0, self.skip_depth - 1)

        if tag in self.BLOCK_BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if (
            self.capture_depth > 0
            and self.skip_depth == 0
        ):
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = []
        for line in raw.splitlines():
            compact = " ".join(line.split())
            if compact:
                lines.append(compact)
        return "\n".join(lines)


@dataclass(frozen=True)
class PageRecord:
    title: str
    pageid: int
    revid: int
    html: str


def fetch_page(
    endpoint: str,
    title: str,
    *,
    oldid: Optional[int] = None,
) -> PageRecord:
    params = {
        "action": "parse",
        "format": "json",
        "formatversion": 2,
        "prop": "text|revid",
        "disableeditsection": 1,
        "disabletoc": 1,
    }
    if oldid is None:
        params["page"] = title
    else:
        params["oldid"] = int(oldid)

    payload = api_get(endpoint, params)
    if "error" in payload:
        raise RuntimeError(
            f"MediaWiki API error for {title!r}: "
            f"{payload['error']}"
        )

    parsed = payload["parse"]
    return PageRecord(
        title=str(parsed["title"]),
        pageid=int(parsed["pageid"]),
        revid=int(parsed["revid"]),
        html=str(parsed["text"]),
    )


def ordered_child_links(
    page: PageRecord,
    *,
    root_page: str,
    direct_only: bool,
    exclude_pattern: Optional[re.Pattern[str]],
) -> List[str]:
    collector = LinkCollector()
    collector.feed(page.html)

    prefix = root_page + "/"
    root_depth = root_page.count("/")
    result = []
    seen = set()

    for title in collector.links:
        if not title.startswith(prefix):
            continue

        if direct_only and title.count("/") != root_depth + 1:
            continue

        if (
            exclude_pattern is not None
            and exclude_pattern.search(title)
        ):
            continue

        if title not in seen:
            seen.add(title)
            result.append(title)

    return result


def discover_recursive_leaves(
    endpoint: str,
    root_page: str,
    *,
    max_depth: int,
    exclude_pattern: Optional[re.Pattern[str]],
) -> Tuple[List[PageRecord], List[PageRecord]]:
    """DFS in displayed link order; return leaves and all discovery pages."""
    root_depth = root_page.count("/")
    discovery: List[PageRecord] = []
    leaves: List[PageRecord] = []
    visited = set()

    def visit(title: str) -> None:
        if title in visited:
            return
        visited.add(title)

        page = fetch_page(endpoint, title)
        discovery.append(page)

        current_depth = title.count("/") - root_depth
        children = ordered_child_links(
            page,
            root_page=root_page,
            direct_only=False,
            exclude_pattern=exclude_pattern,
        )
        children = [
            child
            for child in children
            if child.count("/") > title.count("/")
            and child.startswith(title + "/")
        ]

        if (
            not children
            or current_depth >= max_depth
        ):
            if title != root_page:
                leaves.append(page)
            return

        for child in children:
            visit(child)

    visit(root_page)
    return leaves, discovery


def source_pages(
    source: Mapping[str, object],
) -> Tuple[List[PageRecord], List[PageRecord]]:
    endpoint = str(source["api_endpoint"])
    acquisition = source["acquisition"]
    if not isinstance(acquisition, Mapping):
        raise ValueError("Invalid acquisition block")

    strategy = acquisition["strategy"]
    exclude = acquisition.get("exclude_title_regex")
    exclude_pattern = (
        re.compile(str(exclude), flags=re.IGNORECASE)
        if exclude
        else None
    )

    if strategy == "exact_pages":
        pages = [
            fetch_page(endpoint, str(title))
            for title in acquisition["pages"]
        ]
        return pages, list(pages)

    if strategy == "ordered_direct_subpages":
        root_title = str(acquisition["root_page"])
        root = fetch_page(endpoint, root_title)
        children = ordered_child_links(
            root,
            root_page=root_title,
            direct_only=True,
            exclude_pattern=exclude_pattern,
        )
        pages = [
            fetch_page(endpoint, title)
            for title in children
        ]
        return pages, [root] + pages

    if strategy == "ordered_recursive_leaf_subpages":
        return discover_recursive_leaves(
            endpoint,
            str(acquisition["root_page"]),
            max_depth=int(acquisition["max_depth"]),
            exclude_pattern=exclude_pattern,
        )

    raise ValueError(
        f"Unknown acquisition strategy: {strategy!r}"
    )


def normalize_line(line: str) -> Tuple[str, ...]:
    normalized = unicodedata.normalize(
        "NFKC", line
    ).lower()

    tokens: List[str] = []
    current: List[str] = []

    for char in normalized:
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tuple(tokens)


def extract_page_lines(page: PageRecord) -> List[str]:
    extractor = TextExtractor()
    extractor.feed(page.html)
    text = extractor.text()

    lines = []
    for raw_line in text.splitlines():
        tokens = normalize_line(raw_line)
        if tokens:
            lines.append(" ".join(tokens))
    return lines


def crop_lines(
    lines: Sequence[str],
    *,
    target_tokens: int,
) -> Tuple[List[str], int]:
    output: List[str] = []
    count = 0

    for line in lines:
        tokens = tuple(
            token
            for token in line.split()
            if token
        )
        if not tokens:
            continue

        remaining = target_tokens - count
        if remaining <= 0:
            break

        selected = tokens[:remaining]
        if selected:
            output.append(" ".join(selected))
            count += len(selected)

        if count == target_tokens:
            break

    return output, count


def freeze_source(
    source: Mapping[str, object],
    *,
    target_tokens: int,
    output_root: Path,
    registry_hash: str,
) -> dict:
    source_id = str(source["source_id"])
    pages, discovery_pages = source_pages(source)

    if not pages:
        raise ValueError(
            f"{source_id}: no content pages discovered"
        )

    extracted_lines: List[str] = []
    page_rows = []

    for page in pages:
        page_lines = extract_page_lines(page)
        page_token_count = sum(
            len(line.split())
            for line in page_lines
        )
        page_rows.append(
            {
                "title": page.title,
                "pageid": page.pageid,
                "revid": page.revid,
                "extracted_tokens": page_token_count,
                "extracted_text_sha256": sha256_bytes(
                    ("\n".join(page_lines) + "\n").encode(
                        "utf-8"
                    )
                ),
            }
        )
        extracted_lines.extend(page_lines)

    frozen_lines, frozen_token_count = crop_lines(
        extracted_lines,
        target_tokens=target_tokens,
    )

    if frozen_token_count != target_tokens:
        raise ValueError(
            f"{source_id}: source produced only "
            f"{frozen_token_count:,} normalized tokens; "
            f"frozen target is {target_tokens:,}. "
            "Do not silently duplicate or pad the source."
        )

    source_dir = output_root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "source.txt"
    source_path.write_text(
        "\n".join(frozen_lines) + "\n",
        encoding="utf-8",
    )

    page_manifest_path = source_dir / "source_pages.json"
    page_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_id": source_id,
                "pages_in_text_order": page_rows,
                "discovery_pages": [
                    {
                        "title": page.title,
                        "pageid": page.pageid,
                        "revid": page.revid,
                    }
                    for page in discovery_pages
                ],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.0",
        "freeze_id": "language-controls-v1",
        "source_id": source_id,
        "language": source["language"],
        "language_code": source["language_code"],
        "work": source["work"],
        "author": source["author"],
        "work_date_note": source["work_date_note"],
        "edition_note": source["edition_note"],
        "landing_url": source["landing_url"],
        "api_endpoint": source["api_endpoint"],
        "rights_note": source["rights_note"],
        "limitations": source["limitations"],
        "retrieved_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "registry_sha256": registry_hash,
        "target_normalized_tokens": target_tokens,
        "frozen_normalized_tokens": frozen_token_count,
        "normalization": {
            "unicode_normalization": "NFKC",
            "case": "lower",
            "token_definition": (
                "contiguous_unicode_alphabetic_characters"
            ),
            "punctuation_digits": "separators",
            "language_specific_modernization": False,
            "accent_stripping": False,
        },
        "content_page_count": len(pages),
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_pages_path": str(page_manifest_path),
        "source_pages_sha256": sha256_file(
            page_manifest_path
        ),
        "voynich_validation_accessed": False,
        "voynich_locked_test_accessed": False,
    }

    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checksum_paths = [
        source_path,
        page_manifest_path,
        manifest_path,
    ]
    (source_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def append_data_manifest(
    data_manifest: Path,
    manifests: Sequence[Mapping[str, object]],
    *,
    master_manifest_path: Path,
) -> None:
    """Append one versioned top-level control section to data/manifest.yaml."""
    if not data_manifest.is_file():
        raise FileNotFoundError(
            f"Project data manifest not found: {data_manifest}"
        )

    text = data_manifest.read_text(encoding="utf-8")
    section_key = "historical_language_controls:"

    if section_key in text:
        # Frozen v1 already registered. Do not rewrite it automatically.
        return

    lines = [
        "",
        "# Frozen historical natural-language controls (generated by",
        "# scripts/fetch_freeze_language_controls.py).",
        "historical_language_controls:",
        '  freeze_id: "language-controls-v1"',
        "  target_normalized_tokens_per_source: 20000",
        f"  master_manifest: {json.dumps(str(master_manifest_path), ensure_ascii=False)}",
        "  sources:",
    ]

    for manifest in manifests:
        lines.extend(
            [
                f"    - id: {json.dumps(manifest['source_id'], ensure_ascii=False)}",
                f"      language: {json.dumps(manifest['language'], ensure_ascii=False)}",
                '      role: "historical_natural_language_control"',
                f"      local_path: {json.dumps(manifest['source_path'], ensure_ascii=False)}",
                f"      source_url: {json.dumps(manifest['landing_url'], ensure_ascii=False)}",
                f"      download_date_utc: {json.dumps(str(manifest['retrieved_utc']).split('T', 1)[0], ensure_ascii=False)}",
                f"      version_record: {json.dumps(manifest['source_pages_path'], ensure_ascii=False)}",
                f"      sha256: {json.dumps(manifest['source_sha256'], ensure_ascii=False)}",
                f"      license_or_terms: {json.dumps(manifest['rights_note'], ensure_ascii=False)}",
            ]
        )

    data_manifest.write_text(
        text.rstrip() + "\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and freeze Latin, Italian, German, and French "
            "historical-language control corpora from Wikisource."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=DEFAULT_DATA_MANIFEST,
    )
    args = parser.parse_args()

    try:
        if not args.registry.is_file():
            raise ValueError(
                f"Registry not found: {args.registry}"
            )

        registry_bytes = args.registry.read_bytes()
        registry_hash = sha256_bytes(registry_bytes)
        registry = json.loads(
            registry_bytes.decode("utf-8")
        )

        if registry.get("schema_version") != "1.0":
            raise ValueError(
                "Expected registry schema_version=1.0"
            )

        target_tokens = int(
            registry[
                "target_normalized_tokens_per_source"
            ]
        )
        if target_tokens != 20000:
            raise ValueError(
                "language-controls-v1 is frozen at exactly "
                "20,000 normalized tokens per source"
            )

        args.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        master_path = (
            args.output_root
            / "language_controls_v1_manifest.json"
        )

        # Once v1 exists, never re-fetch silently. Verify the committed
        # frozen files instead. Source changes require a new versioned
        # registry/freeze ID rather than overwriting v1.
        if master_path.is_file():
            master = json.loads(
                master_path.read_text(encoding="utf-8")
            )

            if (
                master.get("registry_sha256")
                != registry_hash
            ):
                raise ValueError(
                    "Frozen language-controls-v1 exists but the "
                    "registry hash changed. Create v2 rather than "
                    "overwriting the frozen v1 sources."
                )

            for entry in master["source_manifests"]:
                source_id = entry["source_id"]
                source_path = (
                    args.output_root
                    / source_id
                    / "source.txt"
                )
                manifest_path = (
                    args.output_root
                    / source_id
                    / "manifest.json"
                )

                if (
                    not source_path.is_file()
                    or not manifest_path.is_file()
                ):
                    raise ValueError(
                        f"Frozen source files missing for {source_id}"
                    )

                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                observed = sha256_file(source_path)

                if observed != manifest["source_sha256"]:
                    raise ValueError(
                        f"Frozen source hash mismatch for {source_id}: "
                        f"{observed} != {manifest['source_sha256']}"
                    )

            print("=" * 92)
            print("HISTORICAL LANGUAGE CONTROLS v1 — ALREADY FROZEN")
            print("=" * 92)
            print(f"Master manifest: {master_path}")
            print(f"Registry SHA:    {registry_hash}")
            print("All frozen source hashes: VERIFIED")
            if args.data_manifest.is_file():
                manifest_text = args.data_manifest.read_text(
                    encoding="utf-8"
                )
                if "historical_language_controls:" not in manifest_text:
                    raise ValueError(
                        "Frozen sources exist but data/manifest.yaml "
                        "does not register historical_language_controls."
                    )
            print("Project data manifest registration: VERIFIED")
            print("No network fetch performed.")
            return 0

        # Refuse ambiguous partial state rather than silently mixing revisions.
        partial = [
            source["source_id"]
            for source in registry["sources"]
            if (
                args.output_root
                / source["source_id"]
                / "manifest.json"
            ).exists()
        ]
        if partial:
            raise ValueError(
                "Partial language-control freeze exists without the "
                "master manifest: "
                + ", ".join(partial)
                + ". Remove the incomplete attempt before the first "
                "successful v1 freeze."
            )

        manifests = []

        print("=" * 92)
        print(
            "FREEZE HISTORICAL LANGUAGE CONTROLS — "
            "LATIN / ITALIAN / GERMAN / FRENCH"
        )
        print("=" * 92)
        print(f"Registry:       {args.registry}")
        print(f"Registry SHA:   {registry_hash}")
        print(f"Tokens/source:  {target_tokens:,}")
        print("Voynich validation/test: NOT ACCESSED")
        print()

        for source in registry["sources"]:
            print(
                f"[{source['language']}] "
                f"{source['source_id']}"
            )
            manifest = freeze_source(
                source,
                target_tokens=target_tokens,
                output_root=args.output_root,
                registry_hash=registry_hash,
            )
            manifests.append(manifest)
            print(
                f"  pages={manifest['content_page_count']}  "
                f"tokens={manifest['frozen_normalized_tokens']:,}"
            )
            print(
                f"  SHA256={manifest['source_sha256']}"
            )

        master = {
            "schema_version": "1.0",
            "freeze_id": "language-controls-v1",
            "registry_path": str(args.registry),
            "registry_sha256": registry_hash,
            "target_normalized_tokens_per_source": (
                target_tokens
            ),
            "source_ids": [
                manifest["source_id"]
                for manifest in manifests
            ],
            "source_manifests": [
                {
                    "source_id": manifest["source_id"],
                    "language": manifest["language"],
                    "source_sha256": (
                        manifest["source_sha256"]
                    ),
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

        master_path.write_text(
            json.dumps(
                master,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        append_data_manifest(
            args.data_manifest,
            manifests,
            master_manifest_path=master_path,
        )

        print()
        print(f"Saved master manifest: {master_path}")
        print(f"Updated project manifest: {args.data_manifest}")
        print(
            "Freeze complete. Commit source.txt + manifests before "
            "generating or benchmarking controls."
        )
        return 0

    except Exception as exc:
        print(
            f"LANGUAGE-CONTROL FREEZE ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
