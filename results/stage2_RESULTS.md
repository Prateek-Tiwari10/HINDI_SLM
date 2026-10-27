# Stage 2 — Normalization, filters, audit (dry run on the Stage 1 samples)

**Status:** done.
- **Kaggle run:** `prateek99491/hindi-lm-runner` version 7, started with Save Version in the web UI. API-pushed versions can't read Kaggle Secrets; see "What failed".
- **Kaggle username verified:** `prateek99491` (`kaggle/push.sh` checks it before each push).
- **Tests:** 29/29 passed on Kaggle (`tests/test_normalize.py`, `tests/test_filters.py`, `tests/test_dedup_split.py`).

## What was run

`python -m data.pipeline stage2` with `configs/data.yaml` (hash `3d5c2c56ce14`), applied to the four 10k-doc samples:
- `normalize()`;
- then the filters, cheapest first;
- the first failing filter is the recorded reason.

Audits: 50 reservoir-sampled removed docs per (filter, source) are in `results/filter_audit/<filter>/<source>.md`.

### Normalization: one deviation from the decision log, same output on real text

- **Order:** control characters (C*) are removed *before* the ZWJ/ZWNJ and pipe→danda steps, not after.
- **Final NFC:** NFC is applied once more at the end.
- **`html.unescape`:** it repeats until the text stops changing (at most 3 rounds).
- **Why:** with the literal order, inputs like `क\x01|` or `&amp;lt;` need two passes to settle, which breaks the required `normalize(normalize(x)) == normalize(x)`.
- **Evidence:** the idempotence test covers these cases.

## % removed per filter per source (first failing filter)

| source | docs | min_words | devanagari_ratio | unique_lines | symbol_ratio | mean_word_len | boilerplate | **surviving** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikipedia | 10,000 | 32.4 | 2.4 | 0.1 | 1.9 | 0.0 | 0.0 | **63.2** |
| manual | 10,000 | 0.5 | 3.1 | 11.4 | 7.6 | 0.0 | 0.6 | **76.8** |
| sangraha_verified | 10,000 | 0.5 | 2.1 | 0.2 | 1.0 | 0.0 | 0.0 | **96.2** |
| fineweb2 | 10,000 | 0.0 | 4.5 | 0.0 | 0.0 | 0.0 | 0.1 | **95.3** |

**Top-5 removal reasons (all sources):**
1. `min_words`: 3,341
2. `devanagari_ratio`: 1,206
3. `unique_lines`: 1,171
4. `symbol_ratio`: 1,054
5. `boilerplate`: 74

## STOP-rule check

| Rule | Value | Result |
| --- | --- | --- |
| Filters remove > 60% of Wikipedia | 36.8% | pass |
| Filters remove > 90% of any source | max 36.8% | pass |

These checks are now enforced in code (`check_removal_stop_rules`, limits in `configs/data.yaml` `stop_rules`).

## Audit reading

- **wikipedia / min_words (32%):** year-index pages ("944 ग्रेगोरी कैलंडर का एक अधिवर्ष है" followed by empty month headings) and one-sentence MLA stubs. Removing them is correct.
  - It's a large share, but these docs hold almost no text. Wikipedia is also only ~0.06B tokens, and it is upsampled 2×.
- **wikipedia / devanagari_ratio (2.4%):** mostly raw wikitable markup (`|-style=...`, `!colspan=...`). Correct. A few Hindi articles with long English sections are lost; that's acceptable at this rate.
- **manual / unique_lines (11.4%) and symbol_ratio (7.6%):** almost all Amar Ujala pages where the crawler stored the page's JSON metadata plus the "ऐप डाउनलोड करें" paywall stub instead of the article. Correct.
  - One BBC Hindi article (author/byline bullets) is a borderline false positive.
- **sangraha / devanagari_ratio (2.1%):** English-headline clickbait ("Admit Card 2023 Direct Link") and paywall promos ("Try Punjab Kesari E-Paper"). Correct.
- **boilerplate:** rare (≤0.6%). The phrase list is light, as intended.

## Decision at the STOP gate

**No threshold changes.**
- Every rule passed.
- The removed samples are overwhelmingly junk.
- The one false-positive pattern (BBC byline bullets → symbol_ratio) is too rare to justify loosening a threshold.
- Prateek can still override this by changing `configs/data.yaml` before Stage 3 finishes.

## What failed / notes

- **Kaggle secret:** versions 4–6 (pushed with `kaggle kernels push`) failed at `get_secret("HF_TOKEN")` with HTTP 400. API-pushed versions can't read Kaggle Secrets; versions started in the UI can.
  - **New workflow:** `kaggle/push.sh` pushes the code and stage command, then Prateek clicks Save Version.
  - Stages 3a–5 are chained in one resumable run to keep the clicks few.
- **Normalization whitespace:** lines that contain only spaces survive `normalize()`, because the spec collapses only 3+ consecutive `\n`. They're harmless: the filters ignore empty lines, and the tokenizer treats them as whitespace.
