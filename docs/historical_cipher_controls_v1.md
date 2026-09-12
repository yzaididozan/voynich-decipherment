# Historical Cipher Controls v1

This control block implements seven frozen mechanism families:

1. monoalphabetic substitution;
2. homophonic substitution;
3. verbose substitution;
4. monoalphabetic substitution with null insertion;
5. nomenclator-like frequent-word codes with alphabetic fallback;
6. line-level columnar transposition;
7. frequent-bigram ("syllabic") hybrid encoding.

These are calibration controls, not reconstructions of the Voynich Manuscript.

## Design rules

- Cipher symbols are abstract atomic labels rather than Voynich-looking glyphs.
- The same family parameters are used for every source-language corpus.
- Five deterministic seeds are generated for every family.
- Every synthetic corpus must mechanically decrypt to the normalized plaintext.
- No parameter is tuned against Voynich validation or locked-test text.
- Punctuation, digits, and layout beyond line boundaries are not encoded in v1.
- Plaintext spelling is never modernized or language-normalized by the generator.

## Historical-plausibility interpretation

The families are deliberately simple mechanism classes. Fifteenth-century
European diplomatic ciphers provide precedent for substitution with multiple
alternatives and for code groups representing larger units; the 2025 Naibbe
study is a modern proof-of-concept showing that a verbose homophonic
substitution system executable with fifteenth-century materials can reproduce
several Voynich-like statistics. This v1 suite does not attempt to reproduce
Naibbe itself and does not claim that every parameter choice is attested in one
specific historical key.

The nomenclator, null, transposition, and syllabic/hybrid controls should
therefore be interpreted as falsification/calibration families. Stronger
historical reconstruction can be added later as a separately frozen v2 rather
than altering these controls after seeing results.

## Source corpus requirement

The generator requires an independently acquired UTF-8 plaintext source.
Historical orthography should be frozen before generation. A source manifest
should record provenance, license, language, date, normalization decisions,
and SHA-256 before any cipher control is generated.

Example:

```bash
python scripts/generate_historical_cipher_controls.py \
  --source data/controls/languages/latin/source_01.txt \
  --source-id latin_source_01
```

Outputs are written beneath:

```text
data/controls/ciphers/historical_v1/<source-id>/<family>/seed_<seed>/
```

Each run contains machine-readable ciphertext, a human-readable view, the
complete reversible key, summary counts, and checksums.
