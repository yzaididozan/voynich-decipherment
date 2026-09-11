"""Loss-aware parser for IVTFF Voynich transliteration files.

The raw IVTFF source is treated as evidence and is never destructively
normalized. Every parsed locus preserves:

- the exact original physical source line(s);
- the transliterated text as written, including continuation syntax;
- file-level provenance such as transcription ID and IVTFF alphabet;
- a separate logical/normalized representation in which only IVTFF
  presentation-level line wrapping is removed.

Semantic IVTFF markup such as uncertain spaces, alternative readings,
ligatures, drawing interruptions, inline comments, paragraph markers,
and text tags is preserved in ``text_normalized``.

This parser is intended for IVTFF 2.x corpora including ZL, GC, IT, and RF.
It deliberately does not harmonize their different alphabets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


class IVTFFParseError(ValueError):
    """Raised when IVTFF input violates a structural assumption."""


_PAGE_HEADER_RE = re.compile(
    r"^<(?P<page>f\d+[rv]\d*)>(?P<rest>.*)$"
)

_LOCUS_RE = re.compile(
    r"^<"
    r"(?P<page>f\d+[rv]\d*)"
    r"\."
    r"(?P<number>\d{1,3})"
    r","
    r"(?P<locator>.)"
    r"(?P<complete_type>[A-Z][a-z0-9])"
    r"(?:;(?P<transcriber>.))?"
    r">"
)

_PAGE_VAR_RE = re.compile(r"\$([A-Z])=([^\s>])")
_TEXT_TAG_RE = re.compile(r"<@([A-Z])=([^>])>")

_FILE_HEADER_RE = re.compile(
    r"^#=IVTFF\s+(?P<alphabet>\S+)\s+(?P<version>\S+)"
    r"(?:\s+(?P<options>.*))?$"
)

_SECTION_NAMES: Mapping[str, str] = {
    "A": "astronomical",
    "B": "biological",
    "C": "cosmological",
    "H": "herbal",
    "P": "pharmaceutical",
    "S": "marginal_stars",
    "T": "text_only",
    "Z": "zodiac",
}


@dataclass(frozen=True)
class IVTFFFileHeader:
    """Parsed IVTFF file header."""

    original: str
    alphabet: Optional[str]
    version: Optional[str]
    options: Optional[str]


@dataclass
class PageState:
    """Mutable parsing state for the active IVTFF page."""

    page: str
    page_vars: Dict[str, Optional[str]]
    page_header_original: str
    paragraph_counter: int = 0
    active_paragraph: Optional[int] = None
    active_text_tags: Dict[str, Optional[str]] = field(default_factory=dict)

    def effective_vars(self) -> Dict[str, Optional[str]]:
        values: Dict[str, Optional[str]] = dict(self.page_vars)
        for key, value in self.active_text_tags.items():
            values[key] = value
        return values


@dataclass(frozen=True)
class LocusRecord:
    """One logical IVTFF locus with preserved provenance."""

    # Dataset/file provenance.
    transcription_id: Optional[str]
    ivtff_alphabet: Optional[str]
    ivtff_version: Optional[str]
    ivtff_options: Optional[str]
    file_header_original: str

    # Core research fields.
    folio: str
    quire: Optional[int]
    currier: Optional[str]
    scribe: Optional[str]
    section: Optional[str]
    locus: str
    locus_type: str
    paragraph: Optional[int]
    line: int
    text_raw: str
    text_normalized: str

    # Loss-preserving / provenance fields.
    original: str
    original_lines: Tuple[str, ...]
    source_line_number: int
    source_line_numbers: Tuple[int, ...]
    page_header_original: str
    quire_code: Optional[str]
    section_code: Optional[str]
    locus_raw: str
    locus_subtype: str
    locator: str
    transcriber: Optional[str]
    physical_leaf: Optional[int]
    recto_verso: Optional[str]
    foldout_panel: Optional[int]
    page_vars: Mapping[str, Optional[str]]
    text_tags: Mapping[str, Optional[str]]
    paragraph_start: bool
    paragraph_end: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON/pandas-friendly dictionary."""
        return asdict(self)


def _quire_number(code: Optional[str]) -> Optional[int]:
    """Convert IVTFF quire code A..T to 1..20 while preserving raw code elsewhere."""
    if code is None or code == "@":
        return None
    if len(code) == 1 and "A" <= code <= "T":
        return ord(code) - ord("A") + 1
    return None


def _parse_page_name(page: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """Return ``(physical_leaf, recto_verso, foldout_panel)``."""
    match = re.fullmatch(r"f(\d+)([rv])(\d*)", page)
    if not match:
        return None, None, None

    leaf = int(match.group(1))
    side = match.group(2)
    panel = int(match.group(3)) if match.group(3) else None
    return leaf, side, panel


def _section_name(code: Optional[str]) -> Optional[str]:
    if code is None or code == "@":
        return None
    return _SECTION_NAMES.get(code, f"unknown_{code}")


def _parse_page_variables(rest: str) -> Dict[str, Optional[str]]:
    """Extract ``$X=y`` page variables from a page header."""
    variables: Dict[str, Optional[str]] = {}
    for key, value in _PAGE_VAR_RE.findall(rest):
        variables[key] = None if value == "@" else value
    return variables


def _extract_text_tags(text: str) -> Dict[str, Optional[str]]:
    """Return ``<@X=y>`` text-tag updates on the current locus."""
    updates: Dict[str, Optional[str]] = {}
    for key, value in _TEXT_TAG_RE.findall(text):
        updates[key] = None if value == "@" else value
    return updates


def normalize_text_lossless(text_raw: str) -> str:
    """Create a conservative logical representation of IVTFF text.

    Only presentation-level line wrapping is removed.

    Preserved markup includes:

    - ``.`` confident spaces;
    - ``,`` uncertain spaces;
    - ``<->`` / ``<~>`` drawing interruptions;
    - ``[a:b]`` alternative readings;
    - ``{...}`` ligature notation;
    - ``@221;``-style encoded glyphs;
    - ``?`` / ``???`` unreadable text;
    - inline comments;
    - ``<%>`` / ``<$>`` paragraph markers;
    - ``<@X=y>`` text tags.

    No alphabet conversion or semantic normalization occurs here.
    """
    if "\n" not in text_raw:
        return text_raw.strip()

    parts = text_raw.splitlines()
    if not parts:
        return ""

    logical = parts[0].rstrip()

    for continuation in parts[1:]:
        piece = continuation

        if piece.startswith("/"):
            piece = piece[1:]
        piece = piece.lstrip()

        if logical.endswith("/"):
            logical = logical[:-1]

        logical += piece.rstrip()

    return logical.strip()


def parse_file_header(line: str) -> IVTFFFileHeader:
    """Parse an IVTFF file header without enforcing a specific alphabet."""
    stripped = line.rstrip("\r\n")
    match = _FILE_HEADER_RE.match(stripped)

    if not match:
        return IVTFFFileHeader(
            original=stripped,
            alphabet=None,
            version=None,
            options=None,
        )

    return IVTFFFileHeader(
        original=stripped,
        alphabet=match.group("alphabet"),
        version=match.group("version"),
        options=match.group("options"),
    )


def _logical_lines(
    lines: Sequence[str],
) -> Iterator[Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Group physical source lines into IVTFF logical lines."""
    i = 0
    total = len(lines)

    while i < total:
        current = lines[i].rstrip("\r\n")
        physical = [current]
        numbers = [i + 1]

        if current.startswith("#"):
            yield tuple(physical), tuple(numbers)
            i += 1
            continue

        while physical[-1].endswith("/") and i + 1 < total:
            nxt = lines[i + 1].rstrip("\r\n")
            physical.append(nxt)
            numbers.append(i + 2)
            i += 1

        yield tuple(physical), tuple(numbers)
        i += 1


def parse_ivtff_lines(
    lines: Iterable[str],
    *,
    transcription_id: Optional[str] = None,
    strict: bool = True,
) -> Iterator[LocusRecord]:
    """Parse IVTFF source lines into :class:`LocusRecord` objects.

    Parameters
    ----------
    lines:
        Iterable of source lines.
    transcription_id:
        Dataset identifier supplied by the caller, e.g. ``"ZL3b"``,
        ``"GC2a"``, ``"IT2a"``, ``"RF1b-full"``. The parser never guesses it.
    strict:
        If ``True``, structural inconsistencies raise
        :class:`IVTFFParseError`.

    Notes
    -----
    This parser intentionally does not harmonize alphabets, tokenize
    Voynichese, resolve alternatives, decompose ligatures, or reinterpret
    uncertain spaces.
    """
    source_lines = list(lines)
    if not source_lines:
        return

    file_header = parse_file_header(source_lines[0])

    if strict and not source_lines[0].startswith("#=IVTFF"):
        raise IVTFFParseError("Input does not begin with an IVTFF file header")

    page_state: Optional[PageState] = None

    for physical_lines, source_numbers in _logical_lines(source_lines):
        first = physical_lines[0]

        if first.startswith("#"):
            continue

        if first == "":
            continue

        if first.startswith("/"):
            if strict:
                raise IVTFFParseError(
                    f"Orphan continuation line at source line {source_numbers[0]}"
                )
            continue

        page_match = _PAGE_HEADER_RE.match(first)
        locus_match = _LOCUS_RE.match(first)

        if page_match and not locus_match:
            page = page_match.group("page")
            rest = page_match.group("rest")
            page_state = PageState(
                page=page,
                page_vars=_parse_page_variables(rest),
                page_header_original=first,
            )
            continue

        if not locus_match:
            if strict:
                raise IVTFFParseError(
                    f"Unrecognised IVTFF data line at source line "
                    f"{source_numbers[0]}: {first!r}"
                )
            continue

        if page_state is None:
            raise IVTFFParseError(
                f"Locus encountered before any page header at source line "
                f"{source_numbers[0]}"
            )

        locus_page = locus_match.group("page")
        if locus_page != page_state.page and strict:
            raise IVTFFParseError(
                f"Locus page {locus_page!r} does not match active page "
                f"{page_state.page!r} at source line {source_numbers[0]}"
            )

        locus_identifier_end = locus_match.end()
        locus_raw = first[:locus_identifier_end]
        locus = locus_raw[1:-1]

        first_text = first[locus_identifier_end:].lstrip(" ")

        text_parts: List[str] = [first_text]
        if len(physical_lines) > 1:
            text_parts.extend(physical_lines[1:])

        text_raw = "\n".join(text_parts)
        text_normalized = normalize_text_lossless(text_raw)

        complete_type = locus_match.group("complete_type")
        generic_type = complete_type[0]

        tag_updates = _extract_text_tags(text_normalized)
        page_state.active_text_tags.update(tag_updates)
        effective = page_state.effective_vars()

        paragraph_start = generic_type == "P" and "<%>" in text_normalized
        paragraph_end = generic_type == "P" and "<$>" in text_normalized

        if paragraph_start:
            if page_state.active_paragraph is not None and strict:
                raise IVTFFParseError(
                    f"Nested/unclosed paragraph start at "
                    f"{page_state.page}.{locus_match.group('number')}"
                )
            page_state.paragraph_counter += 1
            page_state.active_paragraph = page_state.paragraph_counter

        paragraph: Optional[int]
        if generic_type == "P":
            paragraph = page_state.active_paragraph
        else:
            paragraph = None

        leaf, side, panel = _parse_page_name(locus_page)

        record = LocusRecord(
            transcription_id=transcription_id,
            ivtff_alphabet=file_header.alphabet,
            ivtff_version=file_header.version,
            ivtff_options=file_header.options,
            file_header_original=file_header.original,
            folio=locus_page,
            quire=_quire_number(effective.get("Q")),
            currier=effective.get("L"),
            scribe=effective.get("H"),
            section=_section_name(effective.get("I")),
            locus=locus,
            locus_type=generic_type,
            paragraph=paragraph,
            line=int(locus_match.group("number")),
            text_raw=text_raw,
            text_normalized=text_normalized,
            original="\n".join(physical_lines),
            original_lines=physical_lines,
            source_line_number=source_numbers[0],
            source_line_numbers=source_numbers,
            page_header_original=page_state.page_header_original,
            quire_code=effective.get("Q"),
            section_code=effective.get("I"),
            locus_raw=locus_raw,
            locus_subtype=complete_type,
            locator=locus_match.group("locator"),
            transcriber=locus_match.group("transcriber"),
            physical_leaf=leaf,
            recto_verso=side,
            foldout_panel=panel,
            page_vars=dict(effective),
            text_tags=dict(page_state.active_text_tags),
            paragraph_start=paragraph_start,
            paragraph_end=paragraph_end,
        )

        yield record

        if paragraph_end:
            if page_state.active_paragraph is None and strict:
                raise IVTFFParseError(
                    f"Paragraph end without active paragraph at "
                    f"{page_state.page}.{locus_match.group('number')}"
                )
            page_state.active_paragraph = None


def iter_ivtff_records(
    path: PathLike,
    *,
    transcription_id: Optional[str] = None,
    encoding: str = "ascii",
    strict: bool = True,
) -> Iterator[LocusRecord]:
    """Stream parsed records from an IVTFF file."""
    source = Path(path)

    with source.open("r", encoding=encoding, newline="") as handle:
        yield from parse_ivtff_lines(
            handle,
            transcription_id=transcription_id,
            strict=strict,
        )


def load_ivtff(
    path: PathLike,
    *,
    transcription_id: Optional[str] = None,
    encoding: str = "ascii",
    strict: bool = True,
) -> List[LocusRecord]:
    """Load an IVTFF file into memory as records."""
    return list(
        iter_ivtff_records(
            path,
            transcription_id=transcription_id,
            encoding=encoding,
            strict=strict,
        )
    )


def load_ivtff_dicts(
    path: PathLike,
    *,
    transcription_id: Optional[str] = None,
    encoding: str = "ascii",
    strict: bool = True,
) -> List[Dict[str, object]]:
    """Load an IVTFF file as dictionaries suitable for pandas/JSON."""
    return [
        record.to_dict()
        for record in iter_ivtff_records(
            path,
            transcription_id=transcription_id,
            encoding=encoding,
            strict=strict,
        )
    ]


__all__ = [
    "IVTFFFileHeader",
    "IVTFFParseError",
    "LocusRecord",
    "iter_ivtff_records",
    "load_ivtff",
    "load_ivtff_dicts",
    "normalize_text_lossless",
    "parse_file_header",
    "parse_ivtff_lines",
]
