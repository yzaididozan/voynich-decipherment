# Frozen historical-language controls and cipher benchmark — v1

## Purpose

This stage creates a reproducible natural-language/cipher control block for
VOYAGER without using Voynich validation or locked-test material.

Four source-language corpora are frozen:

| Source ID | Language | Work / edition | Role |
|---|---|---|---|
| `latin_imitatione_christi` | Latin | *De imitatione Christi*, Latin Wikisource | late-medieval Latin control; dating/authorship caveat recorded |
| `italian_decameron_massera` | Italian | Boccaccio, *Decameron*, Massèra 1927 edition | medieval Italian control |
| `german_narrenschiff_1499` | German | Brant, *Narrenschiff*, Basel 1499 edition | near-period German control |
| `french_roland_gautier_1872` | Old French | *Chanson de Roland*, Gautier critical edition 1872 | diachronic medieval French control |

Each source is truncated deterministically to the first **20,000 normalized
alphabetic tokens**. The source file, page revision IDs, extraction hashes,
source SHA-256, retrieval date, rights/terms note, and limitations are saved.

The French control is explicitly not period-matched to the fifteenth century.
It is retained as a medieval Old French structural control. It must not be
described as evidence about fifteenth-century French without a later,
separately frozen near-period French source.

## Freeze rule

`language-controls-v1` is write-once. Once
`data/controls/languages/language_controls_v1_manifest.json` exists, the fetch
script verifies hashes and refuses to re-fetch current Wikisource revisions.
Changing any source or extraction policy requires a new version.

## Cipher ensemble

For every frozen source:

- 7 historical cipher-control families;
- 5 frozen seeds;
- 35 synthetic ciphertext corpora.

Across four source languages this produces **140 control corpora**.

The cipher families and parameters remain those already frozen in
`configs/cipher/historical_controls_v1.json`.

## Structural benchmark

Every control is passed through the applicable VOYAGER Tier-0 algorithms:

- glyph unigram entropy;
- conditional entropy orders 1–4;
- branching entropy;
- token-length distributions;
- within-token glyph mutual information distances 1–8;
- raw token mutual information distances 1–5;
- token-internal initial/medial/final positional JSD;
- edit-distance neighborhoods over the frozen top-vocabulary limit;
- token-frequency and vocabulary diagnostics.

The benchmark imports the existing Tier-0 metric functions and the existing
`configs/tier0/train_v1.json` parameters rather than reimplementing alternative
definitions.

Voynich comparison reads **TRAIN Tier-0 only**:
`results/tier0/train/v1`.

It does not read validation membership or `test_LOCKED.txt`.

## Deliberate omissions

Generic language/cipher controls have no legitimate Currier, scribal-hand,
section, quire, illustration, or physical-manuscript position metadata.
Those constraints are marked unavailable rather than simulated.

No one-dimensional "Voynich similarity score" is computed. Raw metric deltas
are retained in `mechanism_matrix.csv` because arbitrary weighting would create
another post-hoc degree of freedom.

## Run order

1. Freeze source corpora and provenance.
2. Commit the source freeze.
3. Generate the 140 cipher controls.
4. Run the structural benchmark.
5. Inspect `mechanism_matrix.csv` and `family_summary.csv`.
6. Only then define any next-stage confirmatory mechanism-survival analysis.

The all-in-one orchestrator exists for reproducibility, but the recommended
research workflow commits the source freeze between steps 1 and 3.
