#!/usr/bin/env python3
# Patch VOYAGER's language-control fetcher for HTTP 429 + safe resume.
#
# Run from repository root:
#   python patch_fetch_freeze_language_controls.py
#
# It modifies scripts/fetch_freeze_language_controls.py, creates a backup,
# syntax-checks the result, and automatically rolls back on failure.

from __future__ import annotations

from pathlib import Path
import py_compile
import re
import shutil
import sys


TARGET = Path("scripts/fetch_freeze_language_controls.py")
BACKUP = Path("scripts/fetch_freeze_language_controls.py.pre_429_patch")


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match; found {count}. "
            "Refusing a potentially unsafe patch."
        )


def main() -> int:
    if not TARGET.is_file():
        print(f"PATCH ERROR: target not found: {TARGET}", file=sys.stderr)
        return 1

    original = TARGET.read_text(encoding="utf-8")

    if (
        "REQUEST_DELAY_SECONDS = 1.0" in original
        and "load_verified_partial_source(" in original
    ):
        print("Fetcher already contains the 429/resume patch.")
        return 0

    if BACKUP.exists():
        print(
            f"PATCH ERROR: backup already exists: {BACKUP}\n"
            "Inspect/remove it before patching again.",
            file=sys.stderr,
        )
        return 1

    shutil.copy2(TARGET, BACKUP)

    try:
        text = original

        import_line = "from urllib.request import Request, urlopen\n"
        require_once(text, import_line, "urllib import")
        text = text.replace(
            import_line,
            import_line
            + "from urllib.error import HTTPError, URLError\n"
            + "import time\n",
            1,
        )

        ua_block = '''USER_AGENT = (
    "VOYAGER-Voynich-Research/1.0 "
    "(historical language-control corpus acquisition)"
)
'''
        require_once(text, ua_block, "USER_AGENT block")
        text = text.replace(
            ua_block,
            ua_block
            + '''
# Transport-only safeguards. These do not alter source selection,
# normalization, or frozen corpus contents.
REQUEST_DELAY_SECONDS = 1.0
MAX_HTTP_ATTEMPTS = 8
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
''',
            1,
        )

        match = re.search(
            r"def api_get\(endpoint: str, params: Mapping\[str, object\]\) -> dict:\n"
            r".*?(?=\n\nclass LinkCollector)",
            text,
            flags=re.S,
        )
        if not match:
            raise RuntimeError("Could not locate api_get() block")

        new_api_get = '''def api_get(endpoint: str, params: Mapping[str, object]) -> dict:
    # GET MediaWiki JSON with throttling and bounded retry/backoff.
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

    last_error = None

    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()

            time.sleep(REQUEST_DELAY_SECONDS)
            return json.loads(payload.decode("utf-8"))

        except HTTPError as exc:
            last_error = exc

            if (
                exc.code not in RETRYABLE_HTTP_STATUS
                or attempt >= MAX_HTTP_ATTEMPTS
            ):
                raise

            retry_after = exc.headers.get("Retry-After")
            wait = 0.0

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = 0.0

            if wait <= 0.0:
                wait = min(
                    120.0,
                    5.0 * (2 ** (attempt - 1)),
                )
            else:
                wait = min(wait, 300.0)

            print(
                f"  HTTP {exc.code}; retry "
                f"{attempt + 1}/{MAX_HTTP_ATTEMPTS} "
                f"in {wait:.0f}s",
                file=sys.stderr,
            )
            time.sleep(wait)

        except URLError as exc:
            last_error = exc

            if attempt >= MAX_HTTP_ATTEMPTS:
                raise

            wait = min(
                120.0,
                5.0 * (2 ** (attempt - 1)),
            )
            print(
                f"  network error; retry "
                f"{attempt + 1}/{MAX_HTTP_ATTEMPTS} "
                f"in {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)

    if last_error is not None:
        raise last_error
    raise RuntimeError("MediaWiki request exhausted without a result")'''

        text = text[:match.start()] + new_api_get + text[match.end():]

        marker = "\ndef freeze_source(\n"
        require_once(text, marker, "freeze_source marker")

        helper = '''
def load_verified_partial_source(
    source: Mapping[str, object],
    *,
    output_root: Path,
    registry_hash: str,
    target_tokens: int,
) -> Optional[dict]:
    # Reuse a source fully completed before an interrupted first freeze.
    source_id = str(source["source_id"])
    source_dir = output_root / source_id
    source_path = source_dir / "source.txt"
    manifest_path = source_dir / "manifest.json"
    pages_path = source_dir / "source_pages.json"

    required = (source_path, manifest_path, pages_path)
    present = [path.exists() for path in required]

    if not any(present):
        return None

    if not all(present):
        missing = [
            path.name
            for path, exists in zip(required, present)
            if not exists
        ]
        raise ValueError(
            f"{source_id}: incomplete source directory from an interrupted "
            f"freeze; missing {missing}. Remove only "
            f"{source_dir} and retry."
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    expected_fields = {
        "source_id": source_id,
        "registry_sha256": registry_hash,
        "target_normalized_tokens": target_tokens,
        "frozen_normalized_tokens": target_tokens,
    }
    for key, expected in expected_fields.items():
        observed = manifest.get(key)
        if observed != expected:
            raise ValueError(
                f"{source_id}: saved partial manifest mismatch for {key}: "
                f"{observed!r} != {expected!r}. "
                "Refusing to mix freeze protocols."
            )

    observed_source_hash = sha256_file(source_path)
    if observed_source_hash != manifest.get("source_sha256"):
        raise ValueError(
            f"{source_id}: source.txt SHA-256 mismatch"
        )

    observed_pages_hash = sha256_file(pages_path)
    if observed_pages_hash != manifest.get("source_pages_sha256"):
        raise ValueError(
            f"{source_id}: source_pages.json SHA-256 mismatch"
        )

    observed_tokens = sum(
        len(line.split())
        for line in source_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    if observed_tokens != target_tokens:
        raise ValueError(
            f"{source_id}: frozen source has {observed_tokens:,} tokens; "
            f"expected {target_tokens:,}"
        )

    return manifest


'''
        text = text.replace(marker, "\n" + helper + "def freeze_source(\n", 1)

        old_partial = '''        # Refuse ambiguous partial state rather than silently mixing revisions.
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
'''
        require_once(text, old_partial, "old partial-state block")
        new_partial = '''        # Interrupted first freezes are resumable. Fully completed sources
        # are verified against this exact registry and reused; only missing
        # sources are fetched.
        verified_partial = {}
        for source in registry["sources"]:
            manifest = load_verified_partial_source(
                source,
                output_root=args.output_root,
                registry_hash=registry_hash,
                target_tokens=target_tokens,
            )
            if manifest is not None:
                verified_partial[source["source_id"]] = manifest

        manifests = []
'''
        text = text.replace(old_partial, new_partial, 1)

        old_loop = '''        for source in registry["sources"]:
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
'''
        require_once(text, old_loop, "source-freeze loop")
        new_loop = '''        for source in registry["sources"]:
            source_id = source["source_id"]
            print(
                f"[{source['language']}] "
                f"{source_id}"
            )

            if source_id in verified_partial:
                manifest = verified_partial[source_id]
                print(
                    "  existing partial freeze: VERIFIED; "
                    "no network re-fetch"
                )
            else:
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
'''
        text = text.replace(old_loop, new_loop, 1)

        TARGET.write_text(text, encoding="utf-8")
        py_compile.compile(str(TARGET), doraise=True)

    except Exception as exc:
        shutil.copy2(BACKUP, TARGET)
        print(
            f"PATCH ERROR: {type(exc).__name__}: {exc}\n"
            f"Original restored from {BACKUP}.",
            file=sys.stderr,
        )
        return 1

    print("Patched successfully:")
    print(f"  {TARGET}")
    print(f"Backup:")
    print(f"  {BACKUP}")
    print()
    print("Behavior now:")
    print("  - completed Latin freeze is verified and reused")
    print("  - no Latin network re-fetch")
    print("  - MediaWiki requests are spaced by 1 second")
    print("  - HTTP 429 / transient 5xx responses retry with backoff")
    print("  - Retry-After is honored when supplied")
    print("  - corpus-selection and normalization rules are unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
