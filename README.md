# Voynich Decipherment

A falsifiable research program for structural and semantic decipherment of the Voynich Manuscript (Beinecke MS 408).

## Primary Research Question

> Which generative mechanism or family of mechanisms best explains the Voynich Manuscript's textual structure under out-of-sample testing across transcription uncertainty, scribal hands, Currier variation, manuscript sections, positional effects, and historically plausible controls?

## Project Goal

The ultimate goal is to determine whether the Voynich Manuscript contains a recoverable encoded or linguistic system and, if justified by the evidence, produce a reproducible translation of substantial previously unseen text.

The project does **not** begin by assuming that:

- Voynich glyphs are ordinary alphabetic letters.
- Visible spaces are true word boundaries.
- Voynichese is a natural language.
- Voynichese is ciphertext.
- Voynichese is meaningless pseudo-text.
- Currier A and B are separate languages.
- Nearby illustrations reveal nearby word meanings.

Instead, the project proceeds through:

**manuscript → transcription → constraints → segmentation → structural modeling → mechanism discrimination → semantic structure → candidate language/encoding → decipherment → blind translation**

## Core Principle

> Prediction precedes interpretation.

A theory is not considered strong because it can explain text that was already visible when the theory was invented. It becomes scientifically interesting when it correctly predicts held-out material without changing its rules.

## Primary Data

- **Manuscript:** Beinecke MS 408, Yale University
- **Primary transcription:** Zandbergen–Landini ZL v3b
- **Primary format:** IVTFF 2.0
- **Primary processing tool:** IVTT v2.4
- **Primary split unit:** physical folio/leaf
- **Primary objective:** generative-mechanism discrimination
- **Primary evaluation:** held-out folio prediction

Robustness transcriptions may include:

- RF v1b
- GC/v101 v2a
- IT v2a

## Repository Structure

```text
voynich-decipherment/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements-lock.txt
│
├── docs/
│   ├── charter.md
│   ├── preregistration.md
│   ├── decisions.md
│   └── threat-model.md
│
├── data/
│   ├── README.md
│   ├── manifest.yaml
│   ├── raw/
│   │   ├── zl/
│   │   ├── rf/
│   │   ├── gc/
│   │   ├── it/
│   │   └── images/
│   ├── parsed/
│   ├── analytical/
│   ├── controls/
│   └── splits/
│
├── configs/
├── src/
├── scripts/
├── tests/
├── experiments/
├── results/
└── paper/
```

## Research Stages

### Stage 0 — Project Freeze

- Finalize project charter.
- Finalize preregistration.
- Record methodological decisions.
- Define threats to validity.
- Freeze corpus versions before model evaluation.

### Stage 1 — Data Acquisition and Preservation

- Acquire ZL v3b.
- Acquire robustness transcriptions.
- Record source metadata and SHA-256 hashes.
- Preserve raw files unchanged.

### Stage 2 — Parsing and Quality Control

- Implement an IVTFF parser.
- Preserve folio, quire, Currier, scribe, section, locus, line, paragraph, uncertain-space, and alternative-reading metadata.
- Reproduce source corpus counts.
- Validate representative pages manually against manuscript images.

### Stage 3 — Frozen Dataset Split

Primary target:

- 70% training
- 15% validation
- 15% locked test

Split at the physical-leaf level, keeping recto and verso together.

### Stage 4 — Reproduce Known Voynich Structure

Before training neural models, reproduce:

- character frequencies;
- token-length distributions;
- bigram/trigram statistics;
- conditional entropy;
- token-edge restrictions;
- Currier A/B differences;
- section-specific distributions;
- line and paragraph positional effects;
- local token similarity.

### Stage 5 — Build the Voynich Exclusion Benchmark

Every candidate mechanism is tested against:

1. Fifteenth-century physical plausibility.
2. Multiple-scribe compatibility.
3. Currier variation.
4. Low conditional character entropy and positional restrictions.
5. Long-range section-specific structure and local clustering.
6. Line-, paragraph-, and illustration-relative positional effects.
7. Robustness to alternative glyph parsing and uncertain spaces.
8. Large-scale coverage.
9. Held-out prediction.
10. Comparison against natural-language, cipher, pseudo-text, and randomized controls.

### Stage 6 — Baseline Models

Implement simple models first:

- unigram;
- n-gram;
- HMM;
- finite-state models;
- topic models;
- copy-and-transform models;
- slot grammars.

### Stage 7 — Learned Segmentation

Compare:

- conventional EVA segmentation;
- space-free representations;
- uncertain-space variants;
- MDL segmentation;
- probabilistic segmentation;
- neural segmental models.

### Stage 8 — Neural Modeling

Only after the data and benchmark are validated:

- small raw-glyph Transformer;
- hierarchical manuscript model;
- latent-state models;
- metadata-conditioned models.

The Voynich corpus is small, so large models are discouraged because memorization could masquerade as understanding.

### Stage 9 — Competing Generative Mechanisms

Test:

- natural-language models;
- homophonic and verbose ciphers;
- historically plausible substitution systems;
- null insertion;
- copy-edit generation;
- structured pseudo-text;
- slot grammars;
- hybrid mechanisms.

### Stage 10 — Semantic Analysis

Semantic testing begins only after structural model selection is frozen.

Text-derived latent classes are tested against independent manuscript categories such as:

- botanical;
- astronomical;
- zodiac;
- balneological/biological;
- cosmological;
- pharmaceutical;
- recipe-like;
- labels versus running text.

### Stage 11 — Candidate Language and Decipherment

Candidate languages are evaluated only through fixed, historically plausible language-plus-channel models.

No arbitrary:

- anagramming;
- language switching;
- per-passage symbol reassignment;
- unexplained null insertion;
- visually motivated word guessing.

### Stage 12 — Locked Blind Translation

If a decipherment candidate survives:

- freeze segmentation;
- freeze mapping;
- freeze grammar;
- freeze channel rules;
- freeze source language;
- run automatically on held-out folios;
- archive predictions before interpretation.

## Success Levels

### Level 0 — No mechanism survives

Still publishable as an exclusion study.

### Level 1 — Structural prediction

The model predicts unseen textual structure.

### Level 2 — Mechanism identification

One mechanism family consistently outperforms competitors.

### Level 3 — Structural decipherment

Stable units, transformations, and syntax-like relationships are recovered.

### Level 4 — Partial semantic decipherment

Stable mappings predict unseen text.

### Level 5 — Reproducible translation

A fixed system produces coherent, historically plausible plaintext from previously unseen material and survives independent replication.

## Immediate First Milestone

Do **not** train the AI model first.

The immediate target is:

1. Create the repository.
2. Acquire `ZL3b-n.txt`.
3. Record its SHA-256 hash.
4. Implement the IVTFF parser.
5. Reproduce published corpus counts.
6. Validate the parser against representative pages.
7. Write parser tests.
8. Freeze the dataset split.
9. Only then begin structural experiments.

## Research Integrity Rules

- Raw source files are immutable.
- No locked-test tuning.
- No cherry-picked translations.
- No semantic interpretation before structural model freeze.
- Failed experiments are retained.
- All important model changes are logged in `docs/decisions.md`.
- Any translation claim must survive held-out prediction and independent reproduction.

## Publication Plan

### Paper I

Working title:

**Beyond Pattern Matching: A Held-Out Generative Benchmark for the Voynich Manuscript**

Focus:

- corpus;
- exclusion benchmark;
- structural constraints;
- competing generative mechanisms;
- held-out evaluation;
- surviving/rejected model families.

### Paper II

Activated only if a strong mechanism survives Paper I.

Focus:

- source-language testing;
- constrained decipherment;
- semantic validation;
- blind held-out translation;
- independent replication.
