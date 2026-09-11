# Decisions Log

This file records methodological decisions that materially affect the project.

Rules:

1. Never delete an earlier decision entry.
2. Superseded decisions remain visible and link to the replacing decision.
3. Every major change should record:
   - date;
   - decision ID;
   - status;
   - rationale;
   - alternatives considered;
   - effect on prior work.
4. No locked-test result may be used to justify retroactively changing a confirmatory decision.

---

## D-001 — Primary Transcription

**Status:** Accepted  
**Decision:** Use **Zandbergen–Landini ZL v3b** as the primary confirmatory transcription.

### Rationale

The project needs a transcription that preserves:

- uncertain spaces;
- alternative readings;
- paragraph structure;
- locus metadata;
- Currier information;
- scribal metadata;
- manuscript sections.

### Alternatives

- RF v1b
- GC/v101 v2a
- IT v2a

### Resolution

RF, GC, and IT may be used for robustness and sensitivity analyses but may not replace ZL post hoc merely because they produce more favorable model results.

---

## D-002 — Primary Text Format

**Status:** Accepted  
**Decision:** Use **IVTFF 2.0** as the primary transcription format.

### Rationale

IVTFF preserves manuscript-level and locus-level information needed for hierarchical analysis and robustness testing.

---

## D-003 — Primary Processing Tool Version

**Status:** Accepted  
**Decision:** Pin **IVTT v2.4** for IVTFF-related processing where the tool is used.

### Rationale

Version pinning is required for reproducibility.

---

## D-004 — Raw Data Immutability

**Status:** Accepted  
**Decision:** Raw manuscript transcriptions and images are read-only.

### Rationale

Normalization and parsing must never destroy the exact source representation.

### Implementation

Use three layers:

1. `data/raw/`
2. `data/parsed/`
3. `data/analytical/`

---

## D-005 — Dataset Split Unit

**Status:** Accepted  
**Decision:** Split at the **physical folio/leaf level**, not individual token, line, or glyph level.

### Rationale

Random token or line splits would leak local folio structure between training and test data.

### Additional Rule

Recto and verso remain together.

---

## D-006 — Primary Split Target

**Status:** Accepted  
**Decision:** Target:

- 70% train
- 15% validation
- 15% locked test

### Rationale

Provides sufficient development data while preserving a meaningful test partition.

### Note

Exact percentages may vary slightly to preserve physical-leaf integrity and reasonable metadata balance.

---

## D-007 — Deterministic Split Seed

**Status:** Accepted  
**Decision:** Use:

`40814041438`

as the primary dataset split seed.

### Rationale

Ensures deterministic reproduction.

---

## D-008 — Locked-Test Prohibition

**Status:** Accepted  
**Decision:** The locked test set may not be used for:

- architecture choice;
- hyperparameter tuning;
- segmentation choice;
- language choice;
- cipher choice;
- threshold selection;
- error-driven model redesign.

### Rationale

Prevents adaptive overfitting to the final evaluation corpus.

---

## D-009 — No Translation Before Structural Freeze

**Status:** Accepted  
**Decision:** Do not conduct confirmatory semantic translation before the structural model is selected and frozen.

### Rationale

Illustrations and apparent word resemblance create a severe confirmation-bias risk.

### Allowed

Exploratory semantic observations may be privately noted but cannot be used as confirmatory model evidence.

---

## D-010 — Prediction Before Interpretation

**Status:** Accepted  
**Decision:** Prefer held-out predictive performance over retrospective explanation.

### Rationale

A model gains credibility by predicting unseen material, not by explaining material used to construct it.

---

## D-011 — Simple Models Before Neural Models

**Status:** Accepted  
**Decision:** Complete descriptive and interpretable baselines before training Transformers.

### Required Early Baselines

- unigram;
- n-gram;
- HMM or equivalent;
- segmentation baselines;
- random/shuffled controls.

### Rationale

A neural model must demonstrate value beyond local-statistical baselines.

---

## D-012 — Small Neural Models

**Status:** Accepted  
**Decision:** Initial neural models should remain below approximately **10 million parameters** unless later evidence strongly justifies larger systems.

### Rationale

The Voynich corpus is small. Excessively large models risk memorization rather than structural discovery.

---

## D-013 — Multiple Random Seeds

**Status:** Accepted  
**Decision:** Final neural comparisons use five seeds:

- 40801
- 40802
- 40803
- 40804
- 40805

### Rationale

Reduces the chance of treating a favorable random initialization as a stable result.

---

## D-014 — Uncertain Spaces Remain Data

**Status:** Accepted  
**Decision:** Uncertain spaces are preserved during ingestion and explicitly varied during segmentation experiments.

### Rationale

The status of Voynich spaces is itself part of the research question.

---

## D-015 — Alternative Readings Remain Data

**Status:** Accepted  
**Decision:** Alternative transcription readings must not be discarded at ingestion.

### Rationale

Transcription ambiguity is a source of model uncertainty and must be available for sensitivity analysis.

---

## D-016 — Robustness Transcriptions

**Status:** Accepted  
**Decision:** At least one non-ZL representation must be used for robustness testing of major conclusions.

### Candidate Sources

- RF v1b
- GC/v101 v2a
- IT v2a

---

## D-017 — Strong Controls Required

**Status:** Accepted  
**Decision:** Random shuffling alone is an insufficient null hypothesis.

### Required Control Families

- natural-language corpora;
- historical ciphers;
- structured pseudo-text;
- frequency-preserving/randomized surrogates.

### Rationale

A meaningful mechanism should outperform strong alternative generators, not merely noise.

---

## D-018 — Multi-Scale Evaluation

**Status:** Accepted  
**Decision:** Evaluate structure at multiple levels.

### Levels

- glyph;
- learned unit;
- token;
- line;
- paragraph;
- folio;
- section.

### Rationale

A model can match local statistics while failing long-range organization.

---

## D-019 — Mechanism Survival Classification

**Status:** Accepted  
**Decision:** Candidate mechanisms are classified as:

- Rejected
- Weak Survivor
- Strong Survivor

### Strong Survivor Rule

Pass at least 8/10 benchmark constraints and necessarily pass:

- Coverage
- Held-out Prediction
- Control Comparison

Historical plausibility must not be clearly anachronistic.

---

## D-020 — General-Purpose LLMs Are Not Semantic Ground Truth

**Status:** Accepted  
**Decision:** Conversational LLM output cannot be treated as evidence that a Voynich passage means something.

### Allowed Uses

- coding assistance;
- literature organization;
- experiment documentation;
- historical-language tooling support;
- output summarization.

### Prohibited Core Use

Free-form semantic guessing presented as decipherment evidence.

---

## D-021 — Known-Cipher Calibration

**Status:** Accepted  
**Decision:** Before claiming semantic Voynich decipherment, the pipeline must be tested on concealed known-plaintext/cipher tasks.

### Rationale

A method that cannot recover controlled historical encipherments should not be trusted on an unknown script.

---

## D-022 — No Cherry-Picked Translation

**Status:** Accepted  
**Decision:** Candidate decipherments must be evaluated on large predefined contiguous text regions.

### Rationale

Selected phrases create excessive researcher degrees of freedom.

---

## D-023 — Paper I Does Not Require Translation

**Status:** Accepted  
**Decision:** The exclusion/generative benchmark is independently publishable.

### Rationale

A rigorous negative or structural result remains scientifically useful even if no semantic decoding survives.

---

## D-024 — Paper II Requires a Strong Survivor

**Status:** Accepted  
**Decision:** Confirmatory semantic decipherment work begins only after at least one mechanism family satisfies the project's Strong Survivor rule.

---

## Decision Template

Use this for future entries:

```markdown
## D-XXX — Short Title

**Date:** YYYY-MM-DD  
**Status:** Proposed / Accepted / Superseded  
**Decision:** ...

### Rationale

...

### Alternatives Considered

- ...
- ...

### Consequences

...

### Supersedes

None / D-XXX
```
