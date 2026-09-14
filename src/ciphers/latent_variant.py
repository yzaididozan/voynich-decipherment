"""Phase 3A related lexical-variant control generator for VOYAGER.

Intended repository destination:
    src/ciphers/latent_variant.py

This exploratory mechanism is deliberately narrow. The most frequent TRAIN
plaintext words receive a two-glyph nomenclator code family. K controls how
many related surface forms are permitted for each target word:

    K=1  fixed code
    K=2  canonical + one final-position substitution
    K=4  canonical + three final-position substitutions

Across K, the code-token length and 16-symbol code alphabet are held fixed.
Only the number of allowed token-final variants changes.

The mechanism key is fit from CONTROL TRAIN only. Validation never changes
the target-word set, fallback substitution, or nomenclator code families.
Reserved-test source lines are represented by opaque placeholders and are
never encoded by this module in Phase 3A.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]


@dataclass(frozen=True)
class GeneratedLatentVariantControl:
    rows: Tuple[dict, ...]
    key: dict
    summary: dict


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uint64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int.from_bytes(
        sha256(payload.encode("utf-8")).digest()[:8],
        "big",
    )


def read_source_lines(path: Path) -> List[PlainLine]:
    """Read the already-normalized frozen source without renormalizing it."""
    lines: List[PlainLine] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.rstrip("\r\n")
            if not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: empty source line is not allowed"
                )
            tokens = tuple(text.split())
            if not tokens:
                raise ValueError(
                    f"{path}:{line_number}: source line has no tokens"
                )
            lines.append(tokens)

    if not lines:
        raise ValueError(f"Source is empty: {path}")

    return lines


def token_count(lines: Sequence[PlainLine]) -> int:
    return sum(len(line) for line in lines)


def validate_partition(
    line_count: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
) -> None:
    sets = {
        "train": set(int(x) for x in train_indices),
        "validation": set(int(x) for x in validation_indices),
        "reserved": set(int(x) for x in reserved_indices),
    }

    for name, values in sets.items():
        bad = sorted(
            x for x in values
            if x < 1 or x > line_count
        )
        if bad:
            raise ValueError(
                f"{name} contains out-of-range line indices: {bad[:10]}"
            )

    names = list(sets)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} overlap: "
                    f"{sorted(overlap)[:10]}"
                )

    observed = set().union(*sets.values())
    expected = set(range(1, line_count + 1))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Split does not cover source lines exactly; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def select_target_words(
    train_lines: Sequence[PlainLine],
    *,
    top_n: int,
) -> List[str]:
    if top_n < 1 or top_n > 64:
        raise ValueError("top_n must be between 1 and 64")

    counts = Counter(
        token
        for line in train_lines
        for token in line
    )
    if len(counts) < top_n:
        raise ValueError(
            f"TRAIN has only {len(counts)} word types; need {top_n}"
        )

    return [
        word
        for word, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_n]
    ]


def _train_character_alphabet(
    train_lines: Sequence[PlainLine],
) -> List[str]:
    alphabet = sorted(
        {
            char
            for line in train_lines
            for token in line
            for char in token
        }
    )
    if not alphabet:
        raise ValueError("TRAIN character alphabet is empty")
    return alphabet


def build_mechanism_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
    top_n: int = 64,
    code_alphabet_size: int = 16,
    family_slot_width: int = 4,
) -> dict:
    """Fit the mechanism key on TRAIN only.

    Code geometry:
    - 16 N-glyphs are shuffled by seed.
    - each first-position N-glyph supports four lexical families;
    - each family receives a disjoint block of four possible final glyphs;
    - therefore 16 * 4 = 64 target-word families are available;
    - all K variants for one word differ only at the final glyph.
    """
    if code_alphabet_size != 16:
        raise ValueError(
            "v1 freezes code_alphabet_size to exactly 16"
        )
    if family_slot_width != 4:
        raise ValueError(
            "v1 freezes family_slot_width to exactly 4"
        )
    if top_n > code_alphabet_size * (
        code_alphabet_size // family_slot_width
    ):
        raise ValueError("Requested target vocabulary exceeds code geometry")

    target_words = select_target_words(
        train_lines,
        top_n=top_n,
    )

    rng = random.Random(int(seed))

    code_symbols = [
        f"N{i:04d}"
        for i in range(1, code_alphabet_size + 1)
    ]
    rng.shuffle(code_symbols)

    # Randomize which lexical item occupies which family slot while keeping
    # the target set itself determined only by TRAIN frequency.
    assigned_words = list(target_words)
    rng.shuffle(assigned_words)

    families: Dict[str, dict] = {}

    families_per_first = code_alphabet_size // family_slot_width
    if families_per_first * code_alphabet_size < top_n:
        raise ValueError("Insufficient family slots")

    for index, word in enumerate(assigned_words):
        first_index = index // families_per_first
        within_first = index % families_per_first

        first = code_symbols[first_index]
        block_start = within_first * family_slot_width
        finals = code_symbols[
            block_start : block_start + family_slot_width
        ]
        if len(finals) != family_slot_width:
            raise RuntimeError("Incomplete variant block")

        variants = [
            [first, final]
            for final in finals
        ]

        families[word] = {
            "canonical": variants[0],
            "variants_max_k4": variants,
            "variant_position_zero_based": 1,
        }

    # Verify all possible K=4 surface forms are globally collision-free.
    reverse_variant: Dict[Tuple[str, ...], str] = {}
    for word, family in families.items():
        for raw_variant in family["variants_max_k4"]:
            variant = tuple(raw_variant)
            previous = reverse_variant.get(variant)
            if previous is not None and previous != word:
                raise RuntimeError(
                    f"Nomenclator variant collision: {previous} vs {word}"
                )
            reverse_variant[variant] = word

    alphabet = _train_character_alphabet(train_lines)
    mono_glyphs = [
        f"M{i:04d}"
        for i in range(1, len(alphabet) + 1)
    ]
    rng.shuffle(mono_glyphs)
    mono_forward = dict(zip(alphabet, mono_glyphs))
    mono_reverse = {
        glyph: char
        for char, glyph in mono_forward.items()
    }

    train_counts = Counter(
        token
        for line in train_lines
        for token in line
    )

    return {
        "schema_version": "1.0",
        "mechanism": "related_multi_code_nomenclator",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "top_n": int(top_n),
        "code_alphabet_size": int(code_alphabet_size),
        "family_slot_width": int(family_slot_width),
        "code_symbols": code_symbols,
        "target_words_frequency_ranked": target_words,
        "target_words_seed_assigned": assigned_words,
        "target_word_train_counts": {
            word: train_counts[word]
            for word in target_words
        },
        "families": families,
        "mono_forward": mono_forward,
        "mono_reverse": mono_reverse,
        "validation_unseen_character_policy": (
            "U<hex-codepoint>, deterministic and not fit from validation"
        ),
    }


def _unicode_oov_glyph(char: str) -> str:
    return f"U{ord(char):08X}"


def _decode_unicode_oov_glyph(glyph: str) -> str:
    if not glyph.startswith("U") or len(glyph) != 9:
        raise ValueError(f"Invalid Unicode OOV glyph: {glyph!r}")
    return chr(int(glyph[1:], 16))


def encode_token(
    token: str,
    *,
    key: Mapping[str, object],
    variant_count: int,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
) -> GlyphToken:
    if variant_count not in (1, 2, 4):
        raise ValueError("variant_count must be one of 1, 2, 4")

    families = key["families"]
    if token in families:
        variants = families[token]["variants_max_k4"][:variant_count]
        choice = stable_uint64(
            "latent-variant-choice-v1",
            int(occurrence_seed),
            int(line_index),
            int(token_index),
            token,
        ) % variant_count
        return tuple(variants[choice])

    mono_forward = key["mono_forward"]
    encoded = []
    for char in token:
        glyph = mono_forward.get(char)
        if glyph is None:
            glyph = _unicode_oov_glyph(char)
        encoded.append(glyph)

    if not encoded:
        raise ValueError("Cannot encode empty token")

    return tuple(encoded)


def decode_token(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
) -> str:
    token_tuple = tuple(cipher_token)

    reverse_variant: Dict[Tuple[str, ...], str] = {}
    for word, family in key["families"].items():
        for variant in family["variants_max_k4"]:
            reverse_variant[tuple(variant)] = word

    if token_tuple in reverse_variant:
        return reverse_variant[token_tuple]

    mono_reverse = key["mono_reverse"]
    chars = []
    for glyph in token_tuple:
        if glyph in mono_reverse:
            chars.append(mono_reverse[glyph])
        elif glyph.startswith("U"):
            chars.append(_decode_unicode_oov_glyph(glyph))
        else:
            raise ValueError(
                f"Cipher token contains undecodable glyph {glyph!r}"
            )

    return "".join(chars)


def _pairwise_variant_distance(
    variants: Sequence[Sequence[str]],
) -> float | None:
    if len(variants) < 2:
        return None

    distances = []
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            if len(left) != len(right):
                raise ValueError("Variant lengths differ")
            distances.append(
                sum(a != b for a, b in zip(left, right))
            )
    return sum(distances) / len(distances)


def _diagnostics(
    plaintext_by_line: Mapping[int, PlainLine],
    ciphertext_by_line: Mapping[int, Sequence[GlyphToken]],
    *,
    key: Mapping[str, object],
    variant_count: int,
) -> dict:
    target_words = set(key["families"])

    total_tokens = 0
    targeted_tokens = 0
    cipher_counts: Counter[GlyphToken] = Counter()
    glyphs = Counter()
    realized: Dict[str, Counter[GlyphToken]] = defaultdict(Counter)

    decoded_ok = 0

    for line_index in sorted(plaintext_by_line):
        plain = plaintext_by_line[line_index]
        cipher = ciphertext_by_line[line_index]
        if len(plain) != len(cipher):
            raise RuntimeError("Token-count mismatch in diagnostics")

        for word, cipher_token in zip(plain, cipher):
            total_tokens += 1
            cipher_counts[cipher_token] += 1
            glyphs.update(cipher_token)

            decoded = decode_token(
                cipher_token,
                key=key,
            )
            if decoded == word:
                decoded_ok += 1

            if word in target_words:
                targeted_tokens += 1
                realized[word][cipher_token] += 1

    if decoded_ok != total_tokens:
        raise RuntimeError(
            f"Round-trip failure: {decoded_ok}/{total_tokens}"
        )

    same_surface_numerator = 0
    same_plaintext_pair_denominator = 0
    realized_counts = []

    for word in target_words:
        word_counter = realized.get(word, Counter())
        n = sum(word_counter.values())
        if n > 0:
            realized_counts.append(len(word_counter))
        if n >= 2:
            same_plaintext_pair_denominator += n * (n - 1)
            same_surface_numerator += sum(
                count * (count - 1)
                for count in word_counter.values()
            )

    same_surface_probability = (
        same_surface_numerator
        / same_plaintext_pair_denominator
        if same_plaintext_pair_denominator
        else None
    )

    all_active_variants = [
        family["variants_max_k4"][:variant_count]
        for family in key["families"].values()
    ]
    pairwise = [
        _pairwise_variant_distance(variants)
        for variants in all_active_variants
    ]
    finite_pairwise = [
        value for value in pairwise
        if value is not None
    ]

    return {
        "tokens": total_tokens,
        "targeted_tokens": targeted_tokens,
        "targeted_token_fraction": (
            targeted_tokens / total_tokens
            if total_tokens else 0.0
        ),
        "unique_cipher_tokens": len(cipher_counts),
        "exact_cipher_token_recurrence_fraction": (
            (total_tokens - len(cipher_counts)) / total_tokens
            if total_tokens else 0.0
        ),
        "cipher_glyph_alphabet_size": len(glyphs),
        "round_trip_accuracy": (
            decoded_ok / total_tokens
            if total_tokens else 1.0
        ),
        "family_code_collision_count": 0,
        "mean_realized_variants_per_target_word_present": (
            sum(realized_counts) / len(realized_counts)
            if realized_counts else 0.0
        ),
        "max_realized_variants_per_target_word": (
            max(realized_counts)
            if realized_counts else 0
        ),
        "targeted_variant_utilization_fraction": (
            sum(realized_counts)
            / (len(realized_counts) * variant_count)
            if realized_counts else 0.0
        ),
        "same_surface_probability_given_same_target_plaintext_word": (
            same_surface_probability
        ),
        "mean_pairwise_hamming_distance_within_active_variant_family": (
            sum(finite_pairwise) / len(finite_pairwise)
            if finite_pairwise else None
        ),
    }


def generate_control(
    source_lines: Sequence[PlainLine],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
    seed: int,
    variant_count: int,
    top_n: int = 64,
    code_alphabet_size: int = 16,
    family_slot_width: int = 4,
) -> GeneratedLatentVariantControl:
    """Generate one K/seed control without encoding reserved-test lines."""
    line_count = len(source_lines)
    validate_partition(
        line_count,
        train_indices,
        validation_indices,
        reserved_indices,
    )

    train_set = set(int(x) for x in train_indices)
    validation_set = set(int(x) for x in validation_indices)
    reserved_set = set(int(x) for x in reserved_indices)

    train_lines = [
        source_lines[i - 1]
        for i in sorted(train_set)
    ]

    key = build_mechanism_key(
        train_lines,
        seed=int(seed),
        top_n=int(top_n),
        code_alphabet_size=int(code_alphabet_size),
        family_slot_width=int(family_slot_width),
    )

    rows: List[dict] = []
    plaintext_train: Dict[int, PlainLine] = {}
    plaintext_validation: Dict[int, PlainLine] = {}
    cipher_train: Dict[int, List[GlyphToken]] = {}
    cipher_validation: Dict[int, List[GlyphToken]] = {}

    for line_index, plain_line in enumerate(source_lines, start=1):
        if line_index in reserved_set:
            # Physical JSONL row exists so original line numbering is retained,
            # but the reserved plaintext is neither encoded nor copied.
            rows.append(
                {
                    "line_index": line_index,
                    "reserved_test_opaque": True,
                }
            )
            continue

        cipher_line = [
            encode_token(
                word,
                key=key,
                variant_count=int(variant_count),
                occurrence_seed=int(seed),
                line_index=line_index,
                token_index=token_index,
            )
            for token_index, word in enumerate(plain_line)
        ]

        rows.append(
            {
                "line_index": line_index,
                "tokens": [
                    list(token)
                    for token in cipher_line
                ],
            }
        )

        if line_index in train_set:
            plaintext_train[line_index] = plain_line
            cipher_train[line_index] = cipher_line
        elif line_index in validation_set:
            plaintext_validation[line_index] = plain_line
            cipher_validation[line_index] = cipher_line
        else:
            raise RuntimeError("Line belongs to no split")

    train_diag = _diagnostics(
        plaintext_train,
        cipher_train,
        key=key,
        variant_count=int(variant_count),
    )
    validation_diag = _diagnostics(
        plaintext_validation,
        cipher_validation,
        key=key,
        variant_count=int(variant_count),
    )

    active_variants = {}
    for word, family in key["families"].items():
        active_variants[word] = (
            family["variants_max_k4"][: int(variant_count)]
        )

    output_key = dict(key)
    output_key["variant_count"] = int(variant_count)
    output_key["active_variants"] = active_variants

    summary = {
        "schema_version": "1.0",
        "mechanism": "related_multi_code_nomenclator",
        "status": "exploratory",
        "seed": int(seed),
        "variant_count": int(variant_count),
        "variant_operation": "single_token_final_glyph_substitution",
        "fit_scope": "CONTROL TRAIN ONLY",
        "evaluation_scope": "CONTROL VALIDATION ONLY",
        "train_lines": len(train_set),
        "validation_lines": len(validation_set),
        "reserved_lines": len(reserved_set),
        "reserved_test_encoded": False,
        "round_trip_verified_train_validation": True,
        "train_diagnostics": train_diag,
        "validation_diagnostics": validation_diag,
        "guardrails": [
            "Target words and mechanism key are selected from CONTROL TRAIN only.",
            "Validation cannot change the target-word set or codebook.",
            "Reserved-test lines are represented by opaque JSONL placeholders and are not encoded.",
            "K changes only the number of allowed final-position variants; code length and code alphabet geometry are fixed.",
        ],
    }

    return GeneratedLatentVariantControl(
        rows=tuple(rows),
        key=output_key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedLatentVariantControl,
    *,
    provenance: Mapping[str, object],
) -> dict:
    """Write one generated control and return its artifact hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    key_path = output_dir / "key.json"
    summary_path = output_dir / "summary.json"

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in generated.rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    key_payload = dict(generated.key)
    key_payload["provenance"] = dict(provenance)
    key_path.write_text(
        json.dumps(
            key_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_payload = dict(generated.summary)
    summary_payload["provenance"] = dict(provenance)
    summary_path.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "corpus.jsonl": sha256_file(corpus_path),
        "key.json": sha256_file(key_path),
        "summary.json": sha256_file(summary_path),
    }

    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{digest}  {name}"
            for name, digest in artifacts.items()
        )
        + "\n",
        encoding="utf-8",
    )

    return artifacts
