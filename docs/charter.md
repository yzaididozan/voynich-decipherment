# Project Charter

## Project Title

**VOYAGER: A Falsifiable Generative Framework for Structural Decipherment of the Voynich Manuscript**

Repository name:

`voynich-decipherment`

## 1. Ultimate Objective

The ultimate objective is to determine whether the Voynich Manuscript contains a recoverable encoded or linguistic system and, if the evidence permits, produce a reproducible translation of substantial previously unseen text.

The project will not initially attempt to assign meanings to individual Voynich tokens.

It will first determine:

1. what the basic textual units are;
2. what kinds of mechanisms can generate their observed behavior;
3. whether one mechanism generalizes across scribes, Currier varieties, sections, folios, and positional contexts;
4. whether that mechanism predicts unseen Voynich material;
5. whether latent textual structure contains recoverable semantic information;
6. whether a historically plausible language/encoding model can explain that structure;
7. whether a frozen decoder can translate unseen text without changing its rules.

The project therefore follows:

**physical manuscript → corpus → constraints → segmentation → structural modeling → mechanism discrimination → semantic structure → candidate language/encoding → decipherment → blind translation**

## 2. Primary Research Question

> **Which generative mechanism or family of mechanisms best explains the Voynich Manuscript's textual structure under out-of-sample testing across transcription uncertainty, scribal hands, Currier variation, manuscript sections, positional effects, and historically plausible controls?**

This question is frozen for Paper I.

## 3. Secondary Decipherment Question

> **Can the surviving structural model be combined with a historically plausible source-language and encoding model to recover reproducible semantic or plaintext information from previously unseen Voynich folios?**

This question becomes confirmatory only if a model survives Paper I.

## 4. Primary Hypothesis

### H1

Voynichese contains a reproducible hierarchical generative structure that enables a model trained on one subset of the manuscript to predict unseen folios significantly better than matched natural-language, historical-cipher, structured-pseudo-text, and randomized control models.

### H0

Once manuscript structure, transcription uncertainty, and appropriate control generators are considered, Voynichese cannot be modeled significantly better by a coherent shared generative system than by competing control mechanisms.

## 5. Secondary Hypotheses

### H2 — Shared-System Hypothesis

A shared core model plus relatively small scribe-, section-, and Currier-specific parameters will explain held-out text better than completely independent models for each manuscript class.

Conceptually:

\[
V = G + S + C + T + P
\]

where:

- \(G\) = shared generative system;
- \(S\) = scribal variation;
- \(C\) = Currier variation;
- \(T\) = thematic/section variation;
- \(P\) = positional effects.

### H3 — Nontrivial-Unit Hypothesis

A learned multi-glyph or hierarchical representation will predict unseen text better than treating conventional EVA glyphs as independent alphabetic letters.

### H4 — Segmentation Hypothesis

Conventional spaces do not contain all—and may not correspond exactly to—the statistically optimal boundaries of Voynichese.

### H5 — Mechanism-Discrimination Hypothesis

At least some candidate mechanism families will reproduce individual Voynich properties but fail when required to reproduce the complete multiscale constraint set simultaneously.

### H6 — Semantic-Structure Hypothesis

If Voynichese conveys semantic information, unsupervised representations learned solely from text should predict independently supplied manuscript categories above matched null baselines.

### H7 — Decipherment Hypothesis

If a natural-language source exists, a fixed historically plausible transformation model should explain unseen Voynich sequences better than alternative source-language/transformation combinations and produce reproducible candidate plaintext without post-hoc rule modification.

## 6. Explicit Non-Hypotheses

The project does **not** assume:

- Voynichese is a natural language.
- Voynichese is ciphertext.
- Voynichese is meaningless.
- Visible glyphs are ordinary letters.
- Spaces are true word boundaries.
- EVA symbols correspond to phonemes.
- Currier A and B are separate languages.
- Five scribal hands imply five authors.
- Illustrations directly identify adjacent words.
- Latin, Italian, German, Hebrew, or another language is the plaintext.
- Statistical language-likeness proves meaningful language.

## 7. Primary Manuscript Object

**Beinecke Rare Book and Manuscript Library, Yale University, MS 408.**

High-resolution Yale manuscript images are treated as the authoritative image source.

Images are preserved unchanged in read-only raw storage.

No OCR output, AI reconstruction, or third-party enhanced image will replace the archival image source.

## 8. Primary Textual Corpus

### Primary Transcription

**Zandbergen–Landini ZL v3b**

Planned primary file:

`ZL3b-n.txt`

Role:

**confirmatory primary transcription**

Reasons:

- preserves alternative readings;
- preserves uncertain spaces;
- preserves paragraph information;
- preserves locus metadata;
- includes Currier information;
- includes scribal-hand metadata;
- supports IVTFF processing.

### Robustness Transcriptions

Potential robustness corpora:

- RF v1b;
- GC/v101 v2a;
- IT v2a.

These are robustness corpora only. They may not be selected post hoc because they generate more attractive results.

## 9. Canonical Data Layers

### Layer 1 — Raw

Exact downloaded files.

Never modified.

### Layer 2 — Parsed

Structured records preserving all relevant IVTFF annotations.

### Layer 3 — Analytical

Experiment-specific derived representations, including:

- normalized EVA;
- space-free text;
- uncertain-space variants;
- decomposed/merged glyph representations;
- learned segmentation.

No analytical representation may overwrite raw or parsed data.

## 10. Canonical Record Schema

Every textual record should preserve, where available:

```text
manuscript_id
folio_id
physical_leaf_id
recto_verso
quire
page_order
section
currier_class
scribe
locus_id
locus_type
locus_subtype
paragraph_id
line_id
line_position
paragraph_position
raw_transcription
normalized_transcription
uncertain_spaces
alternative_readings
drawing_intrusions
transcriber
transcription_version
```

Glyph-level records additionally preserve:

```text
glyph_index
token_index
position_in_token
glyph_raw
glyph_normalized
ambiguity_flag
alternative_glyph
```

## 11. Data Split

Primary target:

- 70% training;
- 15% validation;
- 15% locked test.

Atomic split unit:

**physical leaf**

Recto and verso stay together.

The split should be group-stratified, as feasible, by:

- Currier class;
- scribal hand;
- manuscript section;
- locus composition.

The final folio lists must be committed before test evaluation.

## 12. Core Benchmark Constraints

Every candidate mechanism should be tested against:

1. fifteenth-century physical plausibility;
2. multiple-scribe compatibility;
3. Currier variation;
4. low conditional character entropy and strong within-token positional restrictions;
5. long-range section-specific distributions and local clustering;
6. line-, paragraph-, and illustration-relative positional effects where reported;
7. robustness to alternative glyph parsing and uncertain spaces;
8. coverage across large, predeclared manuscript portions;
9. held-out prediction without changing rules;
10. comparison against natural-language, historical-cipher, structured-pseudo-text, and randomized controls.

## 13. Development Hierarchy

### Tier 0

Descriptive statistics.

### Tier 1

n-gram models.

### Tier 2

Markov, HMM, finite-state, and topic models.

### Tier 3

Explicit generative mechanisms.

### Tier 4

Small neural sequence models.

### Tier 5

Hierarchical latent models.

### Tier 6

Source-language plus encoding-channel decipherment models.

A complex model must justify itself through improved held-out prediction or explanatory value.

## 14. Semantic Analysis Gate

Semantic interpretation is prohibited during the initial structural-model selection phase.

Semantic analysis begins only after:

- the primary structural benchmark is complete;
- segmentation is frozen;
- core model-selection decisions are frozen;
- held-out structural performance has been measured.

## 15. Translation Acceptance Criteria

A convincing decipherment must satisfy:

### Mapping Consistency

The same units retain the same values except where variation is governed by previously specified rules.

### Coverage

Large contiguous portions of the manuscript are handled automatically.

### Grammar

Recovered text shows systematic, repeatable grammatical or morphological structure.

### Historical Plausibility

The proposed language and mechanism are compatible with the fifteenth-century context.

### Multi-Scribe Compatibility

The same core system functions across scribal hands.

### Currier Explanation

A/B variation is accounted for by the model or a compatible auxiliary mechanism.

### Blind Prediction

Previously unseen folios are decoded without changing the rules.

### Semantic Validation

Recovered content predicts independent manuscript information not used to construct the decoder.

### Replication

An independent researcher can run the decoder and obtain substantially the same result.

## 16. Valid Project Outcomes

### Outcome A — Exclusion

The project narrows the viable mechanism space.

### Outcome B — Structural Mechanism

A generative mechanism family clearly outperforms competitors.

### Outcome C — Structural Decipherment

Stable units, transformations, and syntax-like relations are recovered.

### Outcome D — Semantic Decipherment

A fixed system produces reproducible, historically plausible plaintext from unseen material.

Translation is therefore the highest level of evidence, not the only scientifically useful outcome.

## 17. Publication Strategy

### Paper I

Focus:

- corpus;
- benchmark;
- structural constraints;
- competing generative mechanisms;
- held-out prediction;
- robustness and ablation;
- surviving/rejected mechanism families.

### Paper II

Activated only if at least one strong structural mechanism survives.

Focus:

- language/channel testing;
- semantic structure;
- candidate decoding;
- blind translation;
- independent replication.

## 18. Governing Principle

> **No interpretation earns credibility merely because it explains data that were already visible when the interpretation was invented.**

The project privileges prospective prediction over retrospective pattern matching.
