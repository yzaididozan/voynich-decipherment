"""Derived STA1 representation for Voynich transliteration corpora.

This module deliberately sits *after* ``src.data.ivtff``:

    raw IVTFF -> lossless IVTFF parsing -> STA1 derivation

It does not modify raw transcription data and it does not hard-code the
external STA1 alphabet definition.  Instead, it reads the official bitrans
rule files locally and applies the native->STA1 direction.

Supported native alphabets used by this project:

- Eva-  -> STA-Eva_def.bit
- EvaT  -> STA-EvaT_def.bit
- v101  -> STA-v101_def.bit

The implementation reproduces the subset of bitrans behavior needed by
these definition files:

- two-column reversible substitution rules;
- greedy longest-match substitution;
- file-order precedence for equal-length rules;
- ``<...>`` comment/metadata spans left unchanged.

For corpus character counting we preserve two explicitly named measures.

``characters``
    Table-reproduction count.  Each STA1 code is counted individually,
    including each ``Z1``.  With the current public ZL3b transcription and
    official STA-Eva rules this reproduces the published Table 9 total.

``characters_collapsed_unknown_runs``
    Documentation-convention count.  Consecutive ``Z1`` codes are collapsed
    to one character, following the explanatory prose accompanying Table 9.

The public ZL3b data currently produce different values under these two
conventions, so both are retained rather than silently choosing one.
Alternative readings ``[first:second]`` are resolved to the first reading
for both measures. IVTFF separators and ``<...>`` metadata/dedicated markers
do not count.

The official bitrans rule files are external research inputs and should be
stored under ``data/reference/sta1/`` (or another caller-supplied directory).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .ivtff import LocusRecord


PathLike = Union[str, Path]

DEFAULT_RULES_DIR = Path("data/reference/sta1")

RULE_FILE_BY_ALPHABET: Mapping[str, str] = {
    "Eva-": "STA-Eva_def.bit",
    "EvaT": "STA-EvaT_def.bit",
    "v101": "STA-v101_def.bit",
}

_STA_CODE_RE = re.compile(r"^[A-Z][1-9a-z]$")
_ALT_RE = re.compile(r"\[([^:\[\]]*):([^\[\]]*)\]")
_ANGLE_SPAN_RE = re.compile(r"<[^>]*>")


class STA1Error(ValueError):
    """Base exception for STA1 conversion/counting problems."""


class STA1RuleError(STA1Error):
    """Raised when a bitrans rule file is missing or unsupported."""


class STA1ConversionError(STA1Error):
    """Raised when native transliteration text cannot be fully converted."""


class STA1CountError(STA1Error):
    """Raised when converted text cannot be interpreted as STA1."""


@dataclass(frozen=True)
class SubstitutionRule:
    source: str
    target: str
    order: int


@dataclass(frozen=True)
class STA1RuleSet:
    """Parsed native->STA1 conversion rules."""

    path: Path
    sha256: str
    sta_alphabet: str
    native_alphabet: str
    rules: Tuple[SubstitutionRule, ...]
    by_first_character: Mapping[str, Tuple[SubstitutionRule, ...]]


@dataclass(frozen=True)
class STA1CorpusCount:
    """Reproducibility metadata for corpus-level STA1 counts.

    ``characters`` is the table-reproduction count (no collapsing of
    consecutive ``Z1`` codes).  ``characters_collapsed_unknown_runs`` is the
    alternative count that applies the prose rule that an unknown sequence
    counts as one character.  The IVTFF-semantic count may be unavailable
    when a source file contains unreadable syntax not defined by IVTFF; this
    condition is recorded instead of guessed.
    """

    characters: int
    characters_ivtff_semantic: Optional[int]
    ivtff_semantic_valid: bool
    ivtff_semantic_error: Optional[str]
    characters_collapsed_unknown_runs: int
    unknown_collapse_delta: int
    native_alphabet: str
    rule_file: str
    rule_sha256: str
    loci_counted: int


def sha256_file(path: PathLike) -> str:
    """Return the SHA-256 digest of a file."""
    h = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rule_file_for_alphabet(
    alphabet: str,
    rules_dir: PathLike = DEFAULT_RULES_DIR,
) -> Path:
    """Return the expected official bitrans rules path for an IVTFF alphabet."""
    try:
        filename = RULE_FILE_BY_ALPHABET[alphabet]
    except KeyError as exc:
        supported = ", ".join(sorted(RULE_FILE_BY_ALPHABET))
        raise STA1RuleError(
            f"No STA1 rule-file mapping for alphabet {alphabet!r}. "
            f"Supported: {supported}"
        ) from exc

    return Path(rules_dir) / filename


def _parse_header(line: str, path: Path) -> Tuple[str, str]:
    """Parse ``##BIT  STA1 <native>`` from an official definition file."""
    fields = line.strip().split()

    if not fields or not fields[0].startswith("##BIT"):
        raise STA1RuleError(f"{path}: missing ##BIT header")

    # The official STA definition files name the two alphabets after ##BIT.
    alphabet_fields = fields[1:]
    if len(alphabet_fields) != 2:
        raise STA1RuleError(
            f"{path}: expected two alphabet names in bitrans header, "
            f"found {alphabet_fields!r}"
        )

    left, right = alphabet_fields
    if left != "STA1":
        raise STA1RuleError(
            f"{path}: expected STA1 in the left column, found {left!r}"
        )

    return left, right


def load_bitrans_rules(
    path: PathLike,
    *,
    expected_native_alphabet: Optional[str] = None,
) -> STA1RuleSet:
    """Load a reversible two-column STA1 bitrans definition.

    The official definition files are oriented ``STA1 -> native``.
    This function reverses each rule so the returned rule set performs
    ``native -> STA1``.

    The parser intentionally rejects homophonic (>2 column) rules and
    rule-block separators because the project should fail loudly if the
    external rule files change into a form this implementation does not
    reproduce.
    """
    source = Path(path)

    if not source.is_file():
        raise STA1RuleError(
            f"STA1 rules file not found: {source}. "
            "Fetch the official bitrans rule files before running STA1 QC."
        )

    lines = source.read_text(encoding="ascii").splitlines()
    if not lines:
        raise STA1RuleError(f"{source}: empty rules file")

    sta_alphabet, native_alphabet = _parse_header(lines[0], source)

    if (
        expected_native_alphabet is not None
        and native_alphabet != expected_native_alphabet
    ):
        raise STA1RuleError(
            f"{source}: header declares native alphabet {native_alphabet!r}, "
            f"expected {expected_native_alphabet!r}"
        )

    reverse_rules: List[SubstitutionRule] = []
    seen_sources: Dict[str, str] = {}

    for line_number, raw in enumerate(lines[1:], start=2):
        stripped = raw.strip()

        if not stripped:
            continue

        # Rules-file metadata / comment declarations.
        if stripped.startswith("#="):
            continue
        if "(comment)" in stripped:
            continue
        if stripped.startswith("#"):
            continue

        # Rule separators change substitution ordering semantics.  The
        # official STA/native definition files used here do not require them.
        if stripped == "------":
            raise STA1RuleError(
                f"{source}:{line_number}: rule-block separators are not "
                "supported by this reproducibility implementation"
            )

        fields = stripped.split()
        if len(fields) != 2:
            raise STA1RuleError(
                f"{source}:{line_number}: expected a two-column reversible "
                f"rule, found {len(fields)} fields"
            )

        sta_token, native_token = fields

        if len(sta_token) % 2 != 0:
            raise STA1RuleError(
                f"{source}:{line_number}: STA1 replacement {sta_token!r} "
                "does not contain complete two-character STA1 codes"
            )

        for i in range(0, len(sta_token), 2):
            code = sta_token[i : i + 2]
            if not _STA_CODE_RE.match(code):
                raise STA1RuleError(
                    f"{source}:{line_number}: invalid STA1 code {code!r}"
                )

        previous = seen_sources.get(native_token)
        if previous is not None and previous != sta_token:
            raise STA1RuleError(
                f"{source}:{line_number}: native token {native_token!r} "
                f"maps ambiguously to {previous!r} and {sta_token!r}"
            )
        seen_sources[native_token] = sta_token

        reverse_rules.append(
            SubstitutionRule(
                source=native_token,
                target=sta_token,
                order=len(reverse_rules),
            )
        )

    if not reverse_rules:
        raise STA1RuleError(f"{source}: no substitution rules found")

    # Bitrans is greedy: longest source token first. Equal-length rules keep
    # source-file order.
    reverse_rules.sort(key=lambda r: (-len(r.source), r.order))

    by_first: Dict[str, List[SubstitutionRule]] = {}
    for rule in reverse_rules:
        if not rule.source:
            raise STA1RuleError(f"{source}: empty source token")
        by_first.setdefault(rule.source[0], []).append(rule)

    frozen_index = {
        key: tuple(value)
        for key, value in by_first.items()
    }

    return STA1RuleSet(
        path=source,
        sha256=sha256_file(source),
        sta_alphabet=sta_alphabet,
        native_alphabet=native_alphabet,
        rules=tuple(reverse_rules),
        by_first_character=frozen_index,
    )


def _replace_rule_non_cumulatively(
    segments: List[Tuple[str, bool]],
    rule: SubstitutionRule,
) -> List[Tuple[str, bool]]:
    """Apply one substitution rule to all still-unconverted text.

    Each tuple is ``(text, protected)``.  Newly generated STA1 output is
    protected immediately so later rules cannot substitute inside it.

    This mirrors bitrans' documented non-cumulative replacement behavior.
    """
    updated: List[Tuple[str, bool]] = []

    for chunk, protected in segments:
        if protected or not chunk:
            updated.append((chunk, protected))
            continue

        start = 0
        while True:
            index = chunk.find(rule.source, start)

            if index < 0:
                if start < len(chunk):
                    updated.append((chunk[start:], False))
                break

            if index > start:
                updated.append((chunk[start:index], False))

            updated.append((rule.target, True))
            start = index + len(rule.source)

    return updated


def _convert_plain_segment(
    text: str,
    rules: STA1RuleSet,
    *,
    strict: bool,
) -> str:
    """Convert text not containing protected ``<...>`` spans.

    Bitrans does *not* tokenize from the left edge of the text.  It orders
    substitution rules by priority, applies each rule everywhere it can, and
    never reprocesses generated output.  Longer source strings have priority;
    equal-length source strings retain rules-file order.

    That distinction matters for real EVA strings such as ``shee'``:
    ``e'`` and ``ee`` have equal source length, and ``e'`` appears earlier in
    the official rule file.  Therefore ``e'`` must be substituted before
    ``ee`` rather than consuming ``ee`` and leaving a stray apostrophe.
    """
    segments: List[Tuple[str, bool]] = [(text, False)]

    # ``rules.rules`` is already sorted by descending source length with
    # source-file order preserved for ties.
    for rule in rules.rules:
        segments = _replace_rule_non_cumulatively(segments, rule)

    passthrough = set(".,[]: \t\r\n")

    if strict:
        for chunk, protected in segments:
            if protected:
                continue

            for offset, ch in enumerate(chunk):
                if ch not in passthrough:
                    # Reconstruct a compact diagnostic from the unresolved
                    # chunk.  We intentionally fail rather than silently
                    # treating unknown native syntax as punctuation.
                    pointer = " " * offset + "^"
                    raise STA1ConversionError(
                        f"Incomplete {rules.native_alphabet}->STA1 "
                        f"substitution: {ch!r}\n{chunk}\n{pointer}"
                    )

    return "".join(chunk for chunk, _ in segments)


def convert_native_text_to_sta1(
    text: str,
    rules: STA1RuleSet,
    *,
    strict: bool = True,
) -> str:
    """Convert one IVTFF text field from its native alphabet to STA1.

    ``<...>`` spans are preserved verbatim, matching the comment protection
    used by the official STA/native bitrans definition files.  This keeps
    paragraph/drawing markers and IVTFF metadata outside the alphabet
    conversion while converting text inside ``[...]`` alternatives and
    ``{...}`` ligature notation according to the rule table.
    """
    output: List[str] = []
    position = 0

    for match in _ANGLE_SPAN_RE.finditer(text):
        if match.start() > position:
            output.append(
                _convert_plain_segment(
                    text[position : match.start()],
                    rules,
                    strict=strict,
                )
            )
        output.append(match.group(0))
        position = match.end()

    if position < len(text):
        output.append(
            _convert_plain_segment(text[position:], rules, strict=strict)
        )

    return "".join(output)


def resolve_alternatives_first(text: str) -> str:
    """Resolve IVTFF ``[first:second]`` alternatives to the first reading."""
    previous = None
    current = text

    # No nested alternatives are expected in IVTFF, but iterating makes the
    # behavior deterministic if more than one independent alternative occurs.
    while current != previous:
        previous = current
        current = _ALT_RE.sub(lambda m: m.group(1), current)

    if "[" in current or "]" in current:
        raise STA1CountError(
            f"Unresolved or malformed alternative-reading syntax: {text!r}"
        )

    return current


def iter_sta1_symbols(
    sta1_text: str,
    *,
    first_alternative: bool = True,
) -> Iterable[Optional[str]]:
    """Yield STA1 symbols and ``None`` boundaries from converted text.

    ``None`` marks a boundary between character runs.  It is useful for the
    documented rule that a consecutive run of unknown ``Z1`` codes counts as
    one character but separated unknowns count independently.
    """
    text = (
        resolve_alternatives_first(sta1_text)
        if first_alternative
        else sta1_text
    )

    # IVTFF comments, text tags, paragraph markers, and drawing-interruption
    # markers are metadata/structure, not STA1 characters.
    text = _ANGLE_SPAN_RE.sub("", text)

    i = 0
    boundary_chars = set("., \t\r\n")

    while i < len(text):
        ch = text[i]

        if ch in boundary_chars:
            yield None
            i += 1
            continue

        if ch in "[]:":
            # Only reachable when first_alternative=False.
            yield None
            i += 1
            continue

        if i + 1 < len(text):
            code = text[i : i + 2]
            if _STA_CODE_RE.match(code):
                yield code
                i += 2
                continue

        raise STA1CountError(
            f"Invalid/unconverted STA1 text at offset {i}: "
            f"{text[i:i+20]!r}"
        )


def count_sta1_characters(
    sta1_text: str,
    *,
    first_alternative: bool = True,
    collapse_unknown_runs: bool = True,
) -> int:
    """Count STA1 characters under an explicit unknown-run convention.

    ``collapse_unknown_runs=True`` applies the prose convention that a
    consecutive ``Z1`` sequence counts as one.  Set it to ``False`` for the
    table-reproduction measure used by corpus QC.
    """
    total = 0
    previous_unknown = False

    for symbol in iter_sta1_symbols(
        sta1_text,
        first_alternative=first_alternative,
    ):
        if symbol is None:
            previous_unknown = False
            continue

        if symbol == "Z1" and collapse_unknown_runs:
            if not previous_unknown:
                total += 1
            previous_unknown = True
            continue

        total += 1
        previous_unknown = False

    return total


def count_native_text_as_sta1(
    text: str,
    rules: STA1RuleSet,
    *,
    strict: bool = True,
) -> int:
    """Convert native text to STA1 and return the documented character count."""
    converted = convert_native_text_to_sta1(text, rules, strict=strict)
    return count_sta1_characters(converted)


def count_native_text_as_sta1_ivtff_semantic(
    text: str,
    rules: STA1RuleSet,
    *,
    strict: bool = True,
) -> int:
    """Count native text using IVTFF's explicit unreadable-text semantics.

    IVTFF distinguishes:

    - ``?``   : one unreadable character;
    - ``???`` : an unknown *number* of unreadable characters.

    Therefore an exact native ``???`` token contributes one character to
    this count, while ``??`` contributes two independently unreadable
    characters. Runs longer than three are rejected rather than guessed.

    Alternative readings are resolved to the first option *before* this
    treatment, matching the project's other character-count conventions.
    """
    native = resolve_alternatives_first(text)

    marker = "<STA1_UNKNOWN_SEQUENCE>"
    unknown_sequences = 0

    def replace_unknown_run(match: re.Match[str]) -> str:
        nonlocal unknown_sequences
        run = match.group(0)

        if len(run) == 3:
            unknown_sequences += 1
            return marker

        if len(run) in (1, 2):
            return run

        raise STA1CountError(
            "IVTFF unreadable-character run longer than three is "
            f"ambiguous under the format semantics: {run!r}"
        )

    protected = re.sub(r"\?+", replace_unknown_run, native)
    converted = convert_native_text_to_sta1(
        protected,
        rules,
        strict=strict,
    )

    # The internal angle-bracket marker is protected from conversion and is
    # ignored by the STA1 symbol counter, so add each semantic unknown
    # sequence back as exactly one character.
    return (
        count_sta1_characters(
            converted,
            first_alternative=False,
            collapse_unknown_runs=False,
        )
        + unknown_sequences
    )


def count_records_as_sta1(
    records: Sequence[LocusRecord],
    *,
    rules_dir: PathLike = DEFAULT_RULES_DIR,
    strict: bool = True,
) -> STA1CorpusCount:
    """Count a parsed IVTFF corpus after explicit native->STA1 conversion."""
    if not records:
        raise STA1CountError("Cannot count an empty record sequence")

    alphabet = records[0].ivtff_alphabet
    if alphabet is None:
        raise STA1CountError("Corpus has no IVTFF alphabet in its file header")

    for record in records:
        if record.ivtff_alphabet != alphabet:
            raise STA1CountError(
                "Mixed IVTFF alphabets encountered in one corpus: "
                f"{alphabet!r} and {record.ivtff_alphabet!r}"
            )

    rule_path = rule_file_for_alphabet(alphabet, rules_dir)
    rules = load_bitrans_rules(
        rule_path,
        expected_native_alphabet=alphabet,
    )

    table_total = 0
    semantic_total: Optional[int] = 0
    semantic_error: Optional[str] = None
    collapsed_total = 0

    for record in records:
        try:
            converted = convert_native_text_to_sta1(
                record.text_normalized,
                rules,
                strict=strict,
            )

            # Codewise measure: every STA1 code counts individually.
            table_total += count_sta1_characters(
                converted,
                collapse_unknown_runs=False,
            )

            # Broad diagnostic: collapse every adjacent Z1 run.
            collapsed_total += count_sta1_characters(
                converted,
                collapse_unknown_runs=True,
            )
        except STA1Error as exc:
            raise STA1CountError(
                f"{record.transcription_id or '<unknown>'} "
                f"{record.locus}: {exc}"
            ) from exc

        # IVTFF-semantic counting is a distinct interpretation layer.  If a
        # corpus contains unreadable syntax outside the IVTFF-defined '?' and
        # '???' forms (for example '????'), do not guess.  Mark this measure
        # unavailable while continuing to report the valid codewise counts.
        if semantic_total is not None:
            try:
                semantic_total += count_native_text_as_sta1_ivtff_semantic(
                    record.text_normalized,
                    rules,
                    strict=strict,
                )
            except STA1CountError as exc:
                semantic_error = (
                    f"{record.transcription_id or '<unknown>'} "
                    f"{record.locus}: {exc}"
                )
                semantic_total = None

    return STA1CorpusCount(
        characters=table_total,
        characters_ivtff_semantic=semantic_total,
        ivtff_semantic_valid=semantic_total is not None,
        ivtff_semantic_error=semantic_error,
        characters_collapsed_unknown_runs=collapsed_total,
        unknown_collapse_delta=table_total - collapsed_total,
        native_alphabet=alphabet,
        rule_file=str(rule_path),
        rule_sha256=rules.sha256,
        loci_counted=len(records),
    )


__all__ = [
    "DEFAULT_RULES_DIR",
    "RULE_FILE_BY_ALPHABET",
    "STA1ConversionError",
    "STA1CorpusCount",
    "STA1CountError",
    "STA1Error",
    "STA1RuleError",
    "STA1RuleSet",
    "SubstitutionRule",
    "convert_native_text_to_sta1",
    "count_native_text_as_sta1",
    "count_native_text_as_sta1_ivtff_semantic",
    "count_records_as_sta1",
    "count_sta1_characters",
    "iter_sta1_symbols",
    "load_bitrans_rules",
    "resolve_alternatives_first",
    "rule_file_for_alphabet",
    "sha256_file",
]
