"""Loss-aware parser for IVTFF Voynich transliteration files.

This module is intentionally conservative.

The raw IVTFF source is treated as evidence and is never destructively
normalised.  Every parsed locus preserves:

* the exact original physical source line(s);
* the transliterated text as written, including continuation syntax;
* a separate logical/normalised representation in which only IVTFF
  presentation-level line wrapping is removed.

Semantic IVTFF markup such as uncertain spaces, alternative readings,
ligatures, drawing interruptions, inline comments, paragraph markers,
and text tags is preserved in ``text_normalized``.

The parser is designed primarily for IVTFF 2.x files such as ZL v3b,
while remaining deliberately small and auditable.

Relevant IVTFF concepts
-----------------------
Page headers
    ``<f1r> <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 ...>``

Locus identifiers
    ``<f1r.1,@P0>``

The three-character locus code consists of:

    ``locator`` + ``generic type`` + ``subtype``

For example, ``@P0`` means locator ``@`` and complete locus type ``P0``.

Paragraph markers
    ``<%>`` starts a paragraph and ``<$>`` ends one.

Text tags
    ``<@X=y>`` override page variable X from that line onward.

Line wrapping
    A physical line ending in ``/`` continues on the next physical line,
    which begins with ``/``.  Those wrapping slashes are layout syntax,
    not Voynich transliteration content.

No semantic cleanup is performed here.  Later analysis code should derive
new representations from these records rather than modifying the raw data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


class IVTFFParseError(ValueError):
    """Raised when input violates a structural assumption of this parser."""


# Page header, e.g.
# <f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
_PAGE_HEADER_RE = re.compile(
    r"^<(?P<page>f\d+[rv]\d*)>"
    r"(?P<rest>.*)$"
)

# Standard locus identifier, e.g.
# <f1r.1,@P0>
# <f1r.1,@P0;Z>       (optional interlinear transcriber ID)
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

# Page variable syntax inside the dedicated page-header comment:
# $Q=A, $L=B, ...
_PAGE_VAR_RE = re.compile(r"\$([A-Z])=([^\s>])")

# Text tag syntax in transliterated text:
# <@L=A>, <@H=2>, <@X=@>
_TEXT_TAG_RE = re.compile(r"<@([A-Z])=([^>])>")

# File header, e.g. #=IVTFF Eva- 2.0 M 5
_FILE_HEADER_RE = re.compile(
    r"^#=IVTFF\s+(?P<alphabet>\S+)\s+(?P<version>\S+)"
    r"(?:\s+(?P<options>.*))?$"
)

# Page illustration/section codes from the IVTFF specification.
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
    """Mutable parsing state for the current IVTFF page."""

    page: str
    page_vars: Dict[str, Optional[str]]
    page_header_original: str
    paragraph_counter: int = 0
    active_paragraph: Optional[int] = None
    active_text_tags: Dict[str, Optional[str]] = field(default_factory=dict)

    def effective_vars(self) -> Dict[str, Optional[str]]:
        """Return page variables after applying active text-tag overrides."""
        values: Dict[str, Optional[str]] = dict(self.page_vars)
        for key, value in self.active_text_tags.items():
            values[key] = value
        return values


@dataclass(frozen=True)
class LocusRecord:
    """One logical IVTFF locus.

    The first eleven fields correspond closely to the initial research
    schema.  Additional fields preserve information that would otherwise
    be lost and are useful for later QC and robustness analysis.
    """

    # Core requested research fields.
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
    """Convert IVTFF quire code A..T to 1..20 without discarding the code."""
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
    # Preserve unknown/custom codes as an explicit machine-readable label
    # rather than silently dropping them.
    return _SECTION_NAMES.get(code, f"unknown_{code}")


def _parse_page_variables(rest: str) -> Dict[str, Optional[str]]:
    """Extract ``$X=y`` page variables from a page header.

    ``@`` means the variable is intentionally unset pending text tags, so
    it is represented as ``None`` in the effective semantic value.  The raw
    page-header line remains available in every record.
    """
    variables: Dict[str, Optional[str]] = {}
    for key, value in _PAGE_VAR_RE.findall(rest):
        variables[key] = None if value == "@" else value
    return variables


def _extract_text_tags(text: str) -> Dict[str, Optional[str]]:
    """Return text-tag updates appearing on this locus.

    IVTFF specifies ``<@X=y>``.  A value of ``@`` unsets the tag.
    """
    updates: Dict[str, Optional[str]] = {}
    for key, value in _TEXT_TAG_RE.findall(text):
        updates[key] = None if value == "@" else value
    return updates


def normalize_text_lossless(text_raw: str) -> str:
    """Create a conservative logical representation of IVTFF text.

    This function performs *only* presentation-level line-wrap removal:

    * physical continuation newlines are removed;
    * a trailing ``/`` and the matching leading ``/`` are removed;
    * indentation after a continuation marker is removed.

    It deliberately preserves semantic/analytical IVTFF markup, including:

    * ``.`` confident word spaces;
    * ``,`` uncertain word spaces;
    * ``<->`` / ``<~>`` drawing interruptions;
    * ``[a:b]`` alternative readings;
    * ``{...}`` ligature notation;
    * ``?`` / ``???`` unreadable characters;
    * inline comments;
    * ``<%>`` / ``<$>`` paragraph markers;
    * ``<@X=y>`` text tags.

    Because ``text_raw`` and ``original`` are retained, later normalization
    policies can always be recomputed from the original evidence.
    """
    if "\n" not in text_raw:
        return text_raw.strip()

    parts = text_raw.splitlines()
    if not parts:
        return ""

    logical = parts[0].rstrip()

    for continuation in parts[1:]:
        piece = continuation

        # Physical continuation lines are required to begin with "/".
        # Remove only the wrapping syntax, never slash-like material elsewhere.
        if piece.startswith("/"):
            piece = piece[1:]
        piece = piece.lstrip()

        if logical.endswith("/"):
            logical = logical[:-1]

        logical += piece.rstrip()

    return logical.strip()


def parse_file_header(line: str) -> IVTFFFileHeader:
    """Parse the first IVTFF line without rejecting unknown future variants."""
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
    """Yield physical IVTFF lines grouped into logical lines.

    A transliteration locus may be wrapped over multiple physical source
    lines.  This helper groups those lines but does not alter their bytes
    beyond removing the Python newline terminator supplied by ``splitlines``.

    Yields
    ------
    (physical_lines, source_line_numbers)
        ``source_line_numbers`` are 1-based.
    """
    i = 0
    total = len(lines)

    while i < total:
        current = lines[i].rstrip("\r\n")
        physical = [current]
        numbers = [i + 1]

        if current.startswith("#"):
            # Comment lines cannot be continued with "/".
            yield tuple(physical), tuple(numbers)
            i += 1
            continue

        # A locus may continue when the current physical line ends with "/".
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
    strict: bool = True,
) -> Iterator[LocusRecord]:
    """Parse IVTFF source lines into :class:`LocusRecord` objects.

    Parameters
    ----------
    lines:
        Iterable of source lines.  Newline characters are permitted.
    strict:
        If ``True``, structural inconsistencies such as a locus referring
        to a different page than the active page raise
        :class:`IVTFFParseError`.  If ``False``, unrecognised non-comment
        lines are skipped where possible.

    Notes
    -----
    This parser intentionally does not tokenize Voynichese.  Tokenization,
    alternative-reading resolution, ligature decomposition, and treatment
    of uncertain spaces belong in later analytical layers.
    """
    source_lines = list(lines)
    if not source_lines:
        return

    # Validate/parse the file header but do not force a particular alphabet
    # or IVTFF minor version here.
    _ = parse_file_header(source_lines[0])

    page_state: Optional[PageState] = None

    for physical_lines, source_numbers in _logical_lines(source_lines):
        first = physical_lines[0]

        # File header and comments.
        if first.startswith("#"):
            continue

        if first == "":
            # Strict IVTFF generally does not need blank lines, but accepting
            # them is harmless and avoids converting formatting into data.
            continue

        # Continuation lines should already have been consumed by _logical_lines.
        if first.startswith("/"):
            if strict:
                raise IVTFFParseError(
                    f"Orphan continuation line at source line {source_numbers[0]}"
                )
            continue

        # Page header must be distinguished from a locus identifier.
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

        # Preserve the exact transliterated source substring(s).  Optional
        # alignment whitespace between identifier and text is formatting, so
        # it is excluded from text_raw but remains present in ``original``.
        first_text = first[locus_identifier_end:].lstrip(" ")

        text_parts: List[str] = [first_text]
        if len(physical_lines) > 1:
            text_parts.extend(physical_lines[1:])
        text_raw = "\n".join(text_parts)
        text_normalized = normalize_text_lossless(text_raw)

        complete_type = locus_match.group("complete_type")
        generic_type = complete_type[0]

        # Text tags take effect on the line where they appear.
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
    encoding: str = "ascii",
    strict: bool = True,
) -> Iterator[LocusRecord]:
    """Stream parsed records from an IVTFF file.

    The entire source is currently read before logical-line reconstruction so
    that wrapped physical lines can be handled deterministically.  ZL-sized
    IVTFF files are small enough that this is negligible, while keeping the
    implementation simple and testable.
    """
    source = Path(path)
    with source.open("r", encoding=encoding, newline="") as handle:
        yield from parse_ivtff_lines(handle, strict=strict)


def load_ivtff(
    path: PathLike,
    *,
    encoding: str = "ascii",
    strict: bool = True,
) -> List[LocusRecord]:
    """Load an IVTFF file into memory as records."""
    return list(iter_ivtff_records(path, encoding=encoding, strict=strict))


def load_ivtff_dicts(
    path: PathLike,
    *,
    encoding: str = "ascii",
    strict: bool = True,
) -> List[Dict[str, object]]:
    """Load an IVTFF file as dictionaries suitable for pandas/JSON."""
    return [
        record.to_dict()
        for record in iter_ivtff_records(path, encoding=encoding, strict=strict)
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
