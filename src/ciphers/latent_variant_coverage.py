"""Phase 3A.2 lexical-coverage dose-response generator for VOYAGER.

Intended repository destination:
    src/ciphers/latent_variant_coverage.py

Scientific intervention
-----------------------
Phase 3A showed that increasing related surface multiplicity from K=1 to K=4
moves the exact-token Markov comparison toward the Voynich direction, but at
N=64 only about 36%-48% of validation tokens were targeted.

Phase 3A.2 holds K=4 and code geometry fixed while varying only the number of
TRAIN-selected lexical families that receive related ciphertext variants:

    N = 64, 256, 1024

A single 1024-word master codebook is fit from CONTROL TRAIN for each
source/seed. The N=64 and N=256 conditions activate prefixes of the same
frequency-ranked target list. Therefore every word shared across conditions
has exactly the same code family and, for a given occurrence, exactly the same
chosen variant. Only additional lexical families are activated as N grows.

Target-code geometry
--------------------
The target-code alphabet has 16 symbols. Every target token is four glyphs:

    [family glyph 1][family glyph 2][family glyph 3][variant glyph]

The first three positions identify the lexical family (16^3 = 4096 possible
families). The final position supplies one of exactly four related variants.
All variants for one word therefore have Hamming distance 1 from one another.

Words outside the active top-N set use a train-fitted monoalphabetic character
substitution. Validation-unseen characters receive deterministic reversible
Unicode-codepoint glyphs rather than fitting a validation-derived mapping.

Leakage guard
-------------
Reserved-test source lines are never encoded. Their output JSONL rows are
opaque placeholders so line numbering remains compatible with the frozen
Level-2 reader while reserved content cannot enter fitting or scoring.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping, Sequence, Tuple


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]


@dataclass(frozen=True)
class GeneratedCoverageControl:
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


def ranked_train_words(
    train_lines: Sequence[PlainLine],
) -> List[str]:
    counts = Counter(
        token
        for line in train_lines
        for token in line
    )
    return [
        word
        for word, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
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


def build_master_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
    master_target_count: int = 1024,
    code_alphabet_size: int = 16,
    variant_count: int = 4,
) -> dict:
    """Fit one TRAIN-only master key shared by all N conditions."""
    if code_alphabet_size != 16:
        raise ValueError(
            "Phase 3A.2 v1 freezes code_alphabet_size to exactly 16"
        )
    if variant_count != 4:
        raise ValueError(
            "Phase 3A.2 v1 freezes variant_count to exactly 4"
        )
    if master_target_count < 1:
        raise ValueError("master_target_count must be positive")
    if master_target_count > code_alphabet_size ** 3:
        raise ValueError(
            "master_target_count exceeds 16^3 family identifiers"
        )

    ranked = ranked_train_words(train_lines)
    if len(ranked) < master_target_count:
        raise ValueError(
            f"TRAIN has only {len(ranked)} word types; "
            f"need {master_target_count}"
        )

    master_targets = ranked[:master_target_count]

    rng = random.Random(int(seed))

    code_symbols = [
        f"N{i:04d}"
        for i in range(1, code_alphabet_size + 1)
    ]
    rng.shuffle(code_symbols)

    # Enumerate all possible family triples over the shuffled symbol order,
    # then shuffle the triple inventory once. Because this master assignment is
    # built before choosing N, codes for shared words are invariant across N.
    family_ids = list(
        itertools.product(
            code_symbols,
            repeat=3,
        )
    )
    rng.shuffle(family_ids)

    families: Dict[str, dict] = {}
    seen_surface_forms: Dict[Tuple[str, ...], str] = {}

    for rank_index, word in enumerate(master_targets):
        family_id = family_ids[rank_index]

        # Give every family its own deterministic permutation of the shared
        # 16-symbol final-position alphabet, then retain exactly four variants.
        finals = list(code_symbols)
        family_rng = random.Random(
            stable_uint64(
                "phase3a2-final-order-v1",
                int(seed),
                word,
                *family_id,
            )
        )
        family_rng.shuffle(finals)
        finals = finals[:variant_count]

        variants = [
            [*family_id, final]
            for final in finals
        ]

        for variant in variants:
            surface = tuple(variant)
            previous = seen_surface_forms.get(surface)
            if previous is not None and previous != word:
                raise RuntimeError(
                    f"Target-code collision: {previous!r} vs {word!r}"
                )
            seen_surface_forms[surface] = word

        families[word] = {
            "train_frequency_rank": rank_index + 1,
            "family_id": list(family_id),
            "variants_k4": variants,
            "variant_position_zero_based": 3,
        }

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

    master_fingerprint_payload = {
        "seed": int(seed),
        "master_target_words_frequency_ranked": master_targets,
        "families": families,
        "mono_forward": mono_forward,
        "code_symbols": code_symbols,
    }
    master_codebook_sha256 = sha256(
        json.dumps(
            master_fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0",
        "mechanism": "related_multi_code_nomenclator_coverage",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "master_target_count": int(master_target_count),
        "variant_count": int(variant_count),
        "code_alphabet_size": int(code_alphabet_size),
        "code_token_length": 4,
        "family_identity_positions": [0, 1, 2],
        "variant_position": 3,
        "code_symbols": code_symbols,
        "master_target_words_frequency_ranked": master_targets,
        "target_word_train_counts": {
            word: train_counts[word]
            for word in master_targets
        },
        "families": families,
        "mono_forward": mono_forward,
        "mono_reverse": mono_reverse,
        "master_codebook_sha256": master_codebook_sha256,
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


def active_target_words(
    key: Mapping[str, object],
    *,
    target_count: int,
) -> Tuple[str, ...]:
    master = tuple(
        key["master_target_words_frequency_ranked"]
    )
    if target_count < 1 or target_count > len(master):
        raise ValueError(
            f"target_count must be in 1..{len(master)}"
        )
    return master[: int(target_count)]


def encode_token(
    token: str,
    *,
    key: Mapping[str, object],
    target_count: int,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
    active_word_set: set[str] | None = None,
) -> GlyphToken:
    active = (
        active_word_set
        if active_word_set is not None
        else set(
            active_target_words(
                key,
                target_count=int(target_count),
            )
        )
    )

    if token in active:
        variants = key["families"][token]["variants_k4"]
        choice = stable_uint64(
            "phase3a2-variant-choice-v1",
            int(occurrence_seed),
            int(line_index),
            int(token_index),
            token,
        ) % 4
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


def _build_reverse_target(
    key: Mapping[str, object],
) -> Dict[Tuple[str, ...], str]:
    reverse_target: Dict[Tuple[str, ...], str] = {}
    for word, family in key["families"].items():
        for variant in family["variants_k4"]:
            reverse_target[tuple(variant)] = word
    return reverse_target


def decode_token(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
    reverse_target: Mapping[Tuple[str, ...], str] | None = None,
) -> str:
    token_tuple = tuple(cipher_token)

    if reverse_target is None:
        reverse_target = _build_reverse_target(key)

    if token_tuple in reverse_target:
        return reverse_target[token_tuple]

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


def _pairwise_hamming(
    variants: Sequence[Sequence[str]],
) -> float:
    distances = []
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            if len(left) != len(right):
                raise ValueError("Variant lengths differ")
            distances.append(
                sum(a != b for a, b in zip(left, right))
            )
    if not distances:
        raise ValueError("Need at least two variants")
    return sum(distances) / len(distances)


def _diagnostics(
    plaintext_by_line: Mapping[int, PlainLine],
    ciphertext_by_line: Mapping[int, Sequence[GlyphToken]],
    *,
    key: Mapping[str, object],
    target_count: int,
) -> dict:
    active = set(
        active_target_words(
            key,
            target_count=int(target_count),
        )
    )

    total_tokens = 0
    targeted_tokens = 0
    cipher_counts: Counter[GlyphToken] = Counter()
    glyphs = Counter()
    realized: Dict[str, Counter[GlyphToken]] = defaultdict(Counter)
    decoded_ok = 0
    reverse_target = _build_reverse_target(key)

    for line_index in sorted(plaintext_by_line):
        plain = plaintext_by_line[line_index]
        cipher = ciphertext_by_line[line_index]
        if len(plain) != len(cipher):
            raise RuntimeError("Token-count mismatch in diagnostics")

        for word, cipher_token in zip(plain, cipher):
            total_tokens += 1
            cipher_counts[cipher_token] += 1
            glyphs.update(cipher_token)

            if decode_token(
                cipher_token,
                key=key,
                reverse_target=reverse_target,
            ) == word:
                decoded_ok += 1

            if word in active:
                targeted_tokens += 1
                realized[word][cipher_token] += 1

    if decoded_ok != total_tokens:
        raise RuntimeError(
            f"Round-trip failure: {decoded_ok}/{total_tokens}"
        )

    same_surface_numerator = 0
    same_plaintext_pair_denominator = 0
    realized_counts = []

    for word in active:
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

    pairwise = [
        _pairwise_hamming(
            key["families"][word]["variants_k4"]
        )
        for word in active
    ]

    return {
        "tokens": total_tokens,
        "active_target_word_types": len(active),
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
        "target_code_collision_count": 0,
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
            / (len(realized_counts) * 4)
            if realized_counts else 0.0
        ),
        "same_surface_probability_given_same_target_plaintext_word": (
            same_surface_probability
        ),
        "mean_pairwise_hamming_distance_within_variant_family": (
            sum(pairwise) / len(pairwise)
            if pairwise else None
        ),
    }


def generate_control(
    source_lines: Sequence[PlainLine],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
    seed: int,
    target_count: int,
    master_target_count: int = 1024,
    code_alphabet_size: int = 16,
    variant_count: int = 4,
) -> GeneratedCoverageControl:
    """Generate one N/seed control without encoding reserved-test lines."""
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

    key = build_master_key(
        train_lines,
        seed=int(seed),
        master_target_count=int(master_target_count),
        code_alphabet_size=int(code_alphabet_size),
        variant_count=int(variant_count),
    )

    # Explicitly validate N only after the full master key exists.
    active = active_target_words(
        key,
        target_count=int(target_count),
    )
    active_set = set(active)

    rows: List[dict] = []
    plaintext_train: Dict[int, PlainLine] = {}
    plaintext_validation: Dict[int, PlainLine] = {}
    cipher_train: Dict[int, List[GlyphToken]] = {}
    cipher_validation: Dict[int, List[GlyphToken]] = {}

    for line_index, plain_line in enumerate(source_lines, start=1):
        if line_index in reserved_set:
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
                target_count=int(target_count),
                occurrence_seed=int(seed),
                line_index=line_index,
                token_index=token_index,
                active_word_set=active_set,
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
        target_count=int(target_count),
    )
    validation_diag = _diagnostics(
        plaintext_validation,
        cipher_validation,
        key=key,
        target_count=int(target_count),
    )

    output_key = dict(key)
    output_key["active_target_count"] = int(target_count)
    output_key["active_target_words_frequency_ranked"] = list(active)

    summary = {
        "schema_version": "1.0",
        "mechanism": "related_multi_code_nomenclator_coverage",
        "status": "exploratory",
        "seed": int(seed),
        "target_count": int(target_count),
        "master_target_count": int(master_target_count),
        "variant_count": 4,
        "code_token_length": 4,
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
            "One 1024-word master key is fit from CONTROL TRAIN only.",
            "N activates a frequency-ranked prefix of the same master target list.",
            "Shared words retain the same code family and occurrence-level variant across N.",
            "K=4, code alphabet size=16, code length=4, and fallback mapping are fixed across N.",
            "Validation cannot change the target list or codebook.",
            "Reserved-test rows are opaque placeholders and are not encoded.",
        ],
    }

    return GeneratedCoverageControl(
        rows=tuple(rows),
        key=output_key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedCoverageControl,
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
