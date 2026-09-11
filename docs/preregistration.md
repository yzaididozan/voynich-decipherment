# Preregistration

## Status

**Draft v0.1**

This preregistration becomes **v1.0** after:

1. the primary corpus has been acquired and checksummed;
2. the IVTFF parser has passed validation;
3. primary corpus counts have been reproduced;
4. the deterministic train/validation/test split has been generated;
5. no model has been evaluated on the locked test set.

Once v1.0 is frozen, confirmatory hypotheses and thresholds must not be altered because of test-set performance.

---

# 1. Primary Research Question

> **Which generative mechanism or family of mechanisms best explains the Voynich Manuscript's textual structure under out-of-sample testing across transcription uncertainty, scribal hands, Currier variation, manuscript sections, positional effects, and historically plausible controls?**

# 2. Secondary Research Question

> **Can the surviving structural model be combined with a historically plausible source-language and encoding model to recover reproducible semantic or plaintext information from previously unseen Voynich folios?**

The secondary question becomes confirmatory only if a model survives the first-stage structural benchmark.

# 3. Primary Hypothesis

### H1

Voynichese contains a reproducible hierarchical generative structure that enables a model trained on one subset of the manuscript to predict unseen folios significantly better than matched natural-language, historical-cipher, structured-pseudo-text, and randomized control models.

### H0

Once manuscript structure, transcription uncertainty, and appropriate control generators are considered, Voynichese cannot be modeled significantly better by a coherent shared generative system than by competing control mechanisms.

# 4. Secondary Hypotheses

### H2 — Shared-System Hypothesis

A shared core model with limited scribe-, Currier-, and section-specific variation will generalize better than completely independent manuscript-subgroup models.

### H3 — Nontrivial-Unit Hypothesis

Learned multi-glyph or hierarchical units will provide better held-out prediction than a naïve one-glyph-one-letter representation.

### H4 — Segmentation Hypothesis

Conventional Voynich spaces do not fully specify the statistically optimal sequence boundaries.

### H5 — Mechanism-Discrimination Hypothesis

Some candidate mechanism families will reproduce individual Voynich properties but fail the combined multiscale constraint suite.

### H6 — Semantic-Structure Hypothesis

Text-only latent representations will predict independently defined manuscript categories above matched null baselines if the text carries content-related information.

### H7 — Decipherment Hypothesis

If a recoverable source language exists, a fixed historically plausible language-plus-channel model will outperform competing language/channel combinations on unseen Voynich material and produce stable candidate plaintext.

# 5. Primary Corpus

Primary manuscript:

**Beinecke MS 408**

Primary transcription:

**Zandbergen–Landini ZL v3b**

Primary format:

**IVTFF 2.0**

Primary processing tool:

**IVTT v2.4**

Robustness transcriptions may include:

- RF v1b;
- GC/v101 v2a;
- IT v2a.

All source files will receive SHA-256 hashes and be recorded in `data/manifest.yaml`.

# 6. Inclusion Criteria

The primary structural corpus will include original Voynich-script text.

Where confidently identified, later marginal additions, Latin-alphabet material, signatures, catalog marks, and non-Voynich text will be excluded from primary modeling but preserved with exclusion metadata.

# 7. Data Partition

Atomic split unit:

**physical folio/leaf**

Recto and verso must remain together.

Target split:

- 70% training;
- 15% validation;
- 15% locked test.

Target stratification variables:

- Currier class;
- scribal hand;
- manuscript section;
- locus composition.

The final split files will be committed as:

```text
data/splits/v1/train.txt
data/splits/v1/validation.txt
data/splits/v1/test_LOCKED.txt
```

# 8. Deterministic Split Seed

Primary project split seed:

```text
40814041438
```

The exact split-generation code will be committed and hashed.

# 9. Locked-Test Rule

The locked test set may not be used for:

- architecture selection;
- hyperparameter selection;
- segmentation selection;
- cipher selection;
- source-language selection;
- threshold tuning;
- error-driven model redesign;
- debugging model behavior.

If a software defect invalidates test evaluation, the incident must be documented.

# 10. Confirmatory Constraint Benchmark

Each mechanism is evaluated against ten constraints.

## C1 — Historical Plausibility

The mechanism must not require obviously anachronistic materials, tools, or computational procedures.

Classification:

- Compatible
- Questionable
- Anachronistic

## C2 — Multiple-Scribe Compatibility

A shared mechanism should generalize across scribal hands or require only limited scribe-specific adaptation.

## C3 — Currier Structure

The model should account for or recover A/B-like variation.

## C4 — Conditional Entropy and Positional Restrictions

The model should reproduce major glyph-level predictability and within-token restrictions.

## C5 — Local and Long-Range Structure

The model should reproduce both local similarity and section-scale distributions.

## C6 — Positional Effects

Where supported by the corpus, the model should reproduce line-, paragraph-, locus-, and related positional effects.

## C7 — Segmentation Robustness

Conclusions should survive reasonable alternative treatments of glyph composition and uncertain spaces.

## C8 — Coverage

The mechanism should operate across large predefined corpus portions rather than selected examples.

## C9 — Held-Out Prediction

Mappings and model rules should predict withheld folios without post-test rule changes.

## C10 — Control Comparison

Performance should be compared with:

- natural-language corpora;
- historical-cipher corpora;
- structured pseudo-text;
- shuffled/randomized baselines.

# 11. Primary Structural Metrics

At minimum:

- unigram entropy;
- first-order conditional entropy;
- higher-order conditional entropy;
- bits per glyph;
- perplexity;
- bigram/trigram distributions;
- branching entropy;
- token-length distribution;
- token-initial/medial/final distributions;
- mutual information by distance;
- local edit-distance neighborhoods;
- section-specific distribution differences;
- Currier classification/recovery;
- scribe generalization.

# 12. Segmentation Conditions

At minimum:

### S0

Conventional ZL/EVA segmentation.

### S1

Uncertain spaces interpreted as boundaries.

### S2

Uncertain spaces interpreted as non-boundaries.

### S3

All spaces removed.

### S4

Automatically learned segmentation.

### S5

Alternative transcription representation.

A result that exists only under one fragile representation will be labeled accordingly.

# 13. Baseline Models

Before neural modeling:

- unigram;
- bigram;
- trigram;
- 4-gram;
- 5-gram;
- token-aware n-gram;
- space-free n-gram;
- HMM or equivalent latent-state baseline.

# 14. Neural Model Scope

Initial neural models should remain small.

Target raw-glyph Transformer search space:

```text
Layers: 4 or 6
Hidden dimension: 128 or 256
Attention heads: 4 or 8
Context length: 256 or 512
Dropout: 0.1–0.3
Target total parameters: <10 million
```

Large models are discouraged because corpus size makes memorization a major validity threat.

# 15. Training Protocol

Initial optimizer:

`AdamW`

Candidate learning rates:

```text
1e-4
3e-4
1e-3
```

Candidate weight decay:

```text
0
0.01
0.1
```

Use:

- early stopping on validation data;
- gradient clipping;
- fixed random seeds;
- no test-set tuning.

# 16. Repeated Runs

Final candidate neural models are evaluated over five predetermined random seeds:

```text
40801
40802
40803
40804
40805
```

Report:

- mean;
- standard deviation;
- individual-run values.

# 17. Structural Prediction Threshold

A candidate structural model will be considered meaningfully predictive only if:

1. it reduces held-out negative log-likelihood by at least **5%** relative to the strongest preregistered simple baseline;
2. improvement is positive for at least **4 of 5 seeds**;
3. a 95% bootstrap confidence interval excludes zero improvement;
4. the qualitative improvement survives at least one independent transcription robustness analysis.

This is evidence of predictive structure, not decipherment.

# 18. Currier Recovery Threshold

For supervised Currier prediction:

- balanced accuracy ≥ **0.80**
- macro-F1 ≥ **0.80**

For unsupervised recovery:

- adjusted Rand index ≥ **0.50**
- permutation-test \(p < 0.01\)

# 19. Segmentation Success Criterion

A learned segmentation may replace conventional segmentation only if it:

1. improves validation predictive likelihood;
2. improves locked-test predictive likelihood;
3. is not dependent on a single transcription;
4. produces reusable units rather than memorized lines or rare whole sequences.

# 20. Mechanism Survival Rule

### Rejected

Fails at least one critical requirement:

- held-out prediction;
- adequate corpus coverage;
- control comparison;

or fails catastrophically across multiple independent structural constraints.

### Weak Survivor

Passes predictive evaluation but explains fewer than 7 of 10 benchmark constraints.

### Strong Survivor

Passes at least 8 of 10 constraints and must include passes on:

- C8 Coverage;
- C9 Prediction;
- C10 Controls.

C1 must be at least historically `Compatible` or `Questionable`.

# 21. Multi-Scribe Criterion

A shared-system model receives strong support if:

- it generalizes to held-out scribal material;
- held-out predictive loss is no more than **15% worse** than an equivalent model trained with that scribe available;
- core learned units remain statistically similar.

# 22. Robustness Criterion

A primary conclusion is considered robust only if its qualitative direction survives:

- the primary ZL representation;
- at least one alternative transcription representation;
- at least two spacing/segmentation treatments.

# 23. Semantic-Signal Gate

Semantic decipherment does not begin unless a text-only representation predicts independent manuscript categories above chance.

Minimum confirmatory criterion:

- macro-F1 improvement ≥ **0.15 absolute** over the strongest simple baseline;
- permutation \(p < 0.01\);
- replication across at least **4 of 5 seeds**.

This demonstrates content-related structure, not word meaning.

# 24. Candidate-Language Gate

A source-language hypothesis advances only if a fixed language-plus-channel model:

1. exceeds language-independent structural baselines on held-out Voynich;
2. exceeds competing candidate languages under comparable transformation complexity;
3. reproduces major structural benchmark properties;
4. maintains stable mappings across folios;
5. does not require passage-specific transformation rules.

# 25. Partial Decipherment Threshold

A system may be called a **candidate partial decipherment** only if:

- at least 80% of eligible test units receive automatic output or a formally defined `UNKNOWN` state;
- at least 70% receive non-UNKNOWN decoding;
- at least 95% of decoded units obey the frozen transformation rules without manual exception;
- decoded morphology exhibits statistically recurring structure;
- performance exceeds shuffled-mapping controls;
- at least one blinded expert evaluation favors the proposed decoding over matched false decodings.

These thresholds may be recalibrated only **before** locked Voynich semantic testing, using synthetic known-cipher calibration.

# 26. Full Decipherment Threshold

The term **decipherment** is reserved for a system satisfying all of:

- rule consistency;
- large-scale coverage;
- recurring grammar;
- recurring vocabulary;
- historical plausibility;
- multi-scribe compatibility;
- Currier compatibility;
- blind held-out generalization;
- independent semantic validation;
- independent replication.

# 27. Confirmatory vs Exploratory Analysis

## Confirmatory

- frozen hypotheses;
- frozen primary corpus;
- frozen dataset split;
- preregistered benchmark;
- preregistered metrics;
- preregistered model families;
- preregistered thresholds.

## Exploratory

Examples:

- additional languages;
- additional ciphers;
- newly proposed segmentation algorithms;
- visualization-driven hypotheses;
- post hoc model extensions.

Exploratory findings must be labeled explicitly and cannot be retroactively promoted to confirmatory evidence on the same test set.

# 28. Translation Safeguards

The following are prohibited as confirmatory decipherment evidence:

- arbitrary anagramming;
- changing glyph values between passages;
- choosing whichever historical spelling fits;
- adding unexplained nulls after viewing output;
- switching source language by passage;
- interpreting a word from an illustration and then using that meaning as proof;
- hand-correcting model output while counting it as automatic decoding.

# 29. Synthetic Calibration

Before semantic Voynich decoding, the full decipherment pipeline should be tested on known historical-language texts encrypted under concealed cipher systems.

The system should attempt to recover:

- source-language family;
- mechanism family;
- symbol mappings;
- partial plaintext.

Failure on controlled problems weakens confidence in any Voynich interpretation.

# 30. Paper I Activation

Paper I begins when:

- corpus version is frozen;
- benchmark is frozen;
- baselines are complete;
- major mechanism families have been tested;
- held-out structural evaluation is complete.

Translation is not required.

# 31. Paper II Activation

Paper II begins only if at least one mechanism reaches **Strong Survivor** status.

# 32. Governing Principle

> **Prediction precedes interpretation.**

Any proposed solution that relies primarily on retrospective visual or semantic fitting will be treated as exploratory rather than confirmatory.
