# Threat Model

This document defines the major threats to validity in the Voynich decipherment project and the safeguards required to reduce them.

The threat model is part of the research design, not an afterthought.

---

# T1 — Transcription Error

## Threat

Existing Voynich transcriptions may contain:

- misread glyphs;
- inconsistent glyph distinctions;
- transcription-specific conventions;
- omitted marks;
- incorrect boundary choices.

A model could therefore learn properties of a transcription system rather than properties of the manuscript.

## Mitigation

- Use ZL v3b as the frozen primary corpus.
- Preserve raw transcription strings.
- Retain alternative readings.
- Compare major findings with at least one alternative transcription.
- Label conclusions that disappear under transcription changes as fragile.

---

# T2 — Incorrect Glyph Segmentation

## Threat

A visually distinct glyph may be:

- a ligature;
- multiple components;
- a scribal variant;
- an ornamental form.

Conversely, what appears to be multiple glyphs may function as one unit.

## Mitigation

Evaluate multiple representations:

- raw EVA;
- decomposed candidates;
- merged frequent combinations;
- automatically learned multi-glyph units.

Do not assume one visible glyph equals one plaintext letter.

---

# T3 — Incorrect Space Interpretation

## Threat

Voynich spaces may not correspond directly to modern word boundaries.

Uncertain spaces increase the risk.

## Mitigation

Test:

- conventional spaces;
- uncertain spaces as boundaries;
- uncertain spaces removed;
- all spaces removed;
- learned segmentation.

Treat segmentation as a model component rather than fixed truth.

---

# T4 — Folio Leakage

## Threat

Random token, glyph, or line splits can place highly related material from one folio in both training and test sets.

This inflates predictive performance.

## Mitigation

- Split at physical-leaf level.
- Keep recto and verso together.
- Use leave-one-quire-out and leave-one-scribe-out stress tests.

---

# T5 — Metadata Leakage

## Threat

A classifier may appear to recover Currier or section structure simply because correlated metadata indirectly reveal the answer.

For example, scribe and section may correlate with Currier class.

## Mitigation

- Run controlled ablations.
- Report performance with and without each metadata source.
- Use stratified analyses where feasible.
- Distinguish textual prediction from metadata prediction.

---

# T6 — Neural Memorization

## Threat

The manuscript corpus is small.

A large neural model could memorize training sequences while appearing highly predictive.

## Mitigation

- Use small models.
- Compare with n-grams and finite-state baselines.
- Evaluate on held-out physical leaves.
- Report training/validation/test gaps.
- Inspect duplication between train and test corpora.
- Prefer interpretable generative models where possible.

---

# T7 — Illustration Confirmation Bias

## Threat

Researchers may see an image and choose a translation that fits it.

This is especially dangerous for ambiguous botanical illustrations.

## Mitigation

- Perform text-first modeling.
- Freeze textual representations before introducing illustration metadata.
- Use images as validation, not initial labels.
- Prefer externally constrained anchors over subjective visual resemblance.

---

# T8 — Candidate-Language Fishing

## Threat

Testing many languages, spellings, transliterations, and transformations makes accidental matches inevitable.

## Mitigation

- Predefine the initial candidate-language protocol.
- Apply comparable model complexity across languages.
- Separate confirmatory and exploratory language tests.
- Report all tested languages, not only favorable ones.
- Evaluate source-language hypotheses on held-out text.

---

# T9 — Post-Hoc Decoding Rules

## Threat

A proposed decipherment can always be made to work if the researcher is allowed to:

- change symbol values;
- add exceptions;
- invoke new abbreviations;
- change segmentation;
- introduce nulls;
- switch languages.

## Mitigation

Freeze before final evaluation:

- unit inventory;
- symbol/channel mappings;
- grammar;
- segmentation;
- language;
- decoding algorithm.

Any modification after test reveal invalidates that confirmatory run.

---

# T10 — Cherry-Picked Translation

## Threat

A theory may successfully interpret a handful of attractive words or passages while failing on the rest of the manuscript.

## Mitigation

- Predefine evaluation passages.
- Require contiguous test regions.
- Report total coverage.
- Require an automatic `UNKNOWN` output instead of silently omitting failures.
- Do not count manually selected examples as primary evidence.

---

# T11 — Multiple Comparisons

## Threat

Thousands of statistical tests can produce apparently significant results by chance.

## Mitigation

- Predefine primary tests.
- Separate exploratory from confirmatory analysis.
- Apply multiple-testing correction where appropriate.
- Report effect sizes and confidence intervals, not only p-values.

---

# T12 — Historical Anachronism

## Threat

A statistically attractive model may require technology, mathematics, cryptographic machinery, or writing conventions unavailable in the fifteenth century.

## Mitigation

Every serious mechanism receives a historical-plausibility assessment covering:

- materials;
- tools;
- computational requirements;
- cryptographic precedent;
- learnability by multiple scribes.

Clearly anachronistic systems may remain computational baselines but cannot advance as historical explanations.

---

# T13 — Overfitting the Benchmark

## Threat

Once the exclusion benchmark is known, model design may gradually become tailored to the specific benchmark metrics rather than the manuscript-generating process.

## Mitigation

- Maintain a locked folio test set.
- Use stress tests not included in primary optimization.
- Prefer broad generative fit over optimizing each metric independently.
- Report benchmark tradeoffs rather than hiding failures.

---

# T14 — Control Weakness

## Threat

A model may appear impressive simply because it is compared only with random text.

## Mitigation

Use increasingly strong controls:

1. shuffled characters;
2. frequency-preserving shuffles;
3. n-gram-preserving surrogates;
4. natural-language corpora;
5. historical cipher simulations;
6. structured pseudo-text;
7. copy-and-transform generators.

Beating random noise alone is not treated as strong evidence.

---

# T15 — Synthetic Control Mismatch

## Threat

A pseudo-text or cipher control may be too simplistic, causing Voynich to appear unique unfairly.

## Mitigation

- Match approximate corpus size.
- Match alphabet size where appropriate.
- Match line/token-length distributions where appropriate.
- Use multiple parameterizations.
- Report sensitivity to control-generator assumptions.

---

# T16 — Section Imbalance

## Threat

Some manuscript sections may dominate the corpus and therefore dominate learned structure.

## Mitigation

- Report counts by section.
- Use balanced evaluation where appropriate.
- Perform leave-one-section-out analyses.
- Separate global from section-specific performance.

---

# T17 — Scribe Imbalance

## Threat

One scribal hand may account for substantially more text than others.

A "shared" model could mostly reflect the dominant hand.

## Mitigation

- Report text volume by hand.
- Use hand-balanced evaluation where feasible.
- Perform leave-one-scribe-out experiments.
- Compare shared versus hand-specific models.

---

# T18 — Currier Confounding

## Threat

Currier A/B may correlate with scribe, section, or physical manuscript location.

A model could appear to explain Currier while actually exploiting correlated variables.

## Mitigation

- Evaluate Currier prediction under controlled metadata settings.
- Run ablations.
- Compare within-section and within-scribe distinctions where enough data exist.
- Avoid interpreting Currier as a language distinction without additional evidence.

---

# T19 — Semantic Overreach

## Threat

Structural regularity may be mistaken for linguistic meaning.

A text-generation mechanism can be highly structured without conveying natural-language semantics.

## Mitigation

Maintain explicit evidence levels:

- structural prediction;
- mechanism identification;
- latent linguistic structure;
- partial decipherment;
- full decipherment.

Never describe structural prediction alone as translation.

---

# T20 — LLM Hallucinated Decipherment

## Threat

Modern language models can generate fluent interpretations from arbitrary patterns.

Fluency may be psychologically persuasive without being evidential.

## Mitigation

General-purpose LLMs may not supply semantic ground truth.

Every claimed mapping must originate from:

- explicit model parameters;
- formal channel rules;
- reproducible statistical relationships;
- or independent historical-linguistic evidence.

---

# T21 — Researcher Expectation Bias

## Threat

Once the researcher favors a cipher, language, or historical theory, ambiguous evidence may be interpreted in its favor.

## Mitigation

- Keep decision logs.
- Freeze hypotheses before test evaluation.
- Use automated evaluation.
- Report negative results.
- Seek blinded expert review for semantic outputs.

---

# T22 — Reproducibility Failure

## Threat

A result may depend on undocumented preprocessing, random seeds, or environment details.

## Mitigation

Every experiment records:

```text
experiment_id
git_commit
config_hash
dataset_manifest_hash
split_version
random_seed
model_name
model_parameters
hardware
validation_metrics
test_metrics
```

Pin dependency versions and preserve model checkpoints.

---

# T23 — Data Version Drift

## Threat

A transcription or source file may be updated online during the project.

## Mitigation

- Store exact raw copies.
- Record version/date.
- Record SHA-256.
- Never silently replace a source file.
- Treat updated corpora as new dataset versions.

---

# T24 — Hidden Human Intervention

## Threat

Manual corrections to model outputs may be inadvertently counted as automatic decipherment.

## Mitigation

Every decoded unit must be marked as one of:

- automatic;
- UNKNOWN;
- manually inspected;
- manually corrected.

Only automatic output counts toward primary decipherment metrics.

---

# T25 — Non-Independence of Expert Review

## Threat

Experts may be biased if told which output is the proposed solution.

## Mitigation

Where feasible, use blinded evaluation containing:

- proposed decoding;
- shuffled or alternative decoding;
- control outputs.

Experts should evaluate grammaticality, historical plausibility, and semantic coherence without knowing the favored condition.

---

# Risk Register

| Threat | Severity | Primary Mitigation |
|---|---|---|
| T1 Transcription error | High | Multiple transcriptions |
| T2 Glyph segmentation | High | Alternative representations |
| T3 Space interpretation | High | Boundary sensitivity analysis |
| T4 Folio leakage | Critical | Leaf-level splits |
| T5 Metadata leakage | High | Ablation and controls |
| T6 Neural memorization | Critical | Small models + held-out folios |
| T7 Illustration bias | Critical | Text-first semantics |
| T8 Language fishing | Critical | Frozen screening protocol |
| T9 Post-hoc rules | Critical | Frozen decoder |
| T10 Cherry-picking | Critical | Coverage + contiguous test |
| T11 Multiple testing | High | Confirmatory/exploratory split |
| T12 Anachronism | High | Historical review |
| T13 Benchmark overfitting | High | Locked test + stress tests |
| T14 Weak controls | Critical | Strong competing generators |
| T15 Control mismatch | High | Matched controls |
| T16 Section imbalance | Medium | Balanced reporting |
| T17 Scribe imbalance | Medium | Leave-one-scribe-out |
| T18 Currier confounding | High | Controlled analyses |
| T19 Semantic overreach | Critical | Evidence-level hierarchy |
| T20 LLM hallucination | Critical | Formal decoder only |
| T21 Expectation bias | High | Preregistration + blinding |
| T22 Reproducibility failure | High | Full experiment provenance |
| T23 Version drift | Medium | Checksums + versioning |
| T24 Hidden intervention | Critical | Automatic/manual labeling |
| T25 Expert-review bias | Medium | Blinded evaluation |

---

# Governing Safety Principle

> A proposed decipherment must become harder to sustain as more independent constraints are added, unless it is genuinely explaining the manuscript.

The project should therefore reward models that survive independent tests and penalize models that require increasing interpretive flexibility.
