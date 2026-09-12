# Manual manuscript-image validation — v1

## Scope

This checkpoint validates a deterministic **TRAIN-ONLY** sample of ZL3b
transcriptions against high-resolution images of Yale Beinecke MS 408.

It does **not** inspect the frozen validation membership or `test_LOCKED.txt`.

Primary image source:
https://beinecke.library.yale.edu/beinecke/collections/beinecke-cipher-voynich-manuscript

## Review procedure

For every row in `pages.csv`:

1. Open the corresponding manuscript folio in Yale's MS 408 image viewer.
2. Confirm that the sampled transcription page corresponds to the correct
   manuscript page.
3. Review all associated rows in `loci.csv`.
4. Compare the parsed transcription with the visible manuscript text.
5. Distinguish ordinary transcription ambiguity from parser/alignment failure.

### `pages.csv` allowed `review_status`

- `PASS`
- `TRANSCRIPTION_DISAGREEMENT`
- `PARSER_OR_ALIGNMENT_ISSUE`
- `UNRESOLVED`

### `loci.csv` allowed `check_status`

- `PASS`
- `TRANSCRIPTION_DISAGREEMENT`
- `PARSER_OR_ALIGNMENT_ISSUE`
- `UNRESOLVED`

Suggested alignment fields use `YES`, `NO`, or `UNCERTAIN`.

## Interpretation

A `TRANSCRIPTION_DISAGREEMENT` does not automatically invalidate the corpus.
It means the image plausibly supports a different glyph/space reading and
should be documented as transcription uncertainty.

A `PARSER_OR_ALIGNMENT_ISSUE` is more serious: it indicates the analytical
record may have been assigned to the wrong locus/page or transformed
incorrectly. Resolve such issues before additional model-development choices.

Do not alter the sampled page list after looking at manuscript images.
