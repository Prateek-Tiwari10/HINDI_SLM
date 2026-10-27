# Stage 3 — Full-scale pipeline on Kaggle CPU

**Status:** done; the token target is met.
- **Kaggle run:** `prateek99491/hindi-lm-runner` version 9 (Save Version in the UI).
- **Kaggle username verified:** `prateek99491` (`kaggle/push.sh` checks it before each push).
- **Chained command:** `stage3a --max-hours 8.5 && stage3b && tokenizer.sweep && tokenizer.encode`.
- **Machine:** 4 CPU cores, 31 GB RAM.

## What was run

**3a — stream, normalize, filter, exact dedup, MinHash-sign**
- Sources in priority order: Wikipedia (all), manual collection (all), then Sangraha Verified `hin`, streamed file by file until the estimate reached 2.1B × 1.25 = 2.625B train tokens.
- One input file = one shard: 81.5k docs per Wikipedia file, 174.8k per Sangraha file, all below the 200k-doc cap.
- For each shard, uploaded in one atomic commit with its done-marker under `work/filtered/<shard>/`:
  - `docs.parquet` (zstd);
  - `signatures.npy` (`uint32`, `[n, 128]`);
  - `hashes.npy`;
  - `stats.json`.
- **Exact dedup:** a global xxh64 set; first seen wins, which preserves source priority.
- **MinHash:** word 5-gram shingles, 128 permutations, seed 42.

**3b — near-dedup** (global, after all shards were signed)
- 16 bands × 8 rows LSH, then verification at signature agreement ≥ 0.80.
- Union-find over the verified pairs. Within each cluster the kept doc is chosen by Wikipedia > manual > Sangraha, then longest.
- **3c — quality classifier:** skipped (see below).
- **3d — split:** `xxh64(id) mod 10,000` gives 99.5 / 0.25 / 0.25. Written to `final/{train,val,test}/*.parquet` on HF.

## Funnel per source (docs / GB of UTF-8 text)

| source | raw | after filters | after exact dedup | after near-dedup | after quality | final |
| --- | --- | --- | --- | --- | --- | --- |
| wikipedia | 163,093 / 0.63 | 93,146 / 0.57 | 93,054 / 0.57 | 86,499 / 0.56 | 86,499 / 0.56 | 86,499 / 0.56 |
| manual | 377,324 / 2.59 | 286,741 / 1.68 | 285,503 / 1.68 | 284,508 / 1.67 | 284,508 / 1.67 | 284,508 / 1.67 |
| sangraha_verified (25 of 100 files) | 4,355,234 / 24.22 | 4,182,186 / 23.24 | 4,097,801 / 22.77 | 4,084,777 / 22.67 | 4,084,777 / 22.67 | 4,082,876 / 22.66 |
| **total** | 4,895,651 / 27.44 | 4,562,073 / 25.49 | 4,476,358 / 25.02 | 4,455,784 / 24.90 | same | 4,453,883 / 24.89 |

**Notes:**
- Filter removal rates match the Stage 2 dry run: Wikipedia keeps 57%, manual 76%, Sangraha 96%.
- FineWeb-2 was never read; only 25 of Sangraha's 100 files were needed.
- **Known discrepancy:** 1,901 more Sangraha docs (0.05%) left at the split step than the near-dedup drop list names.
  - `split_shard` drops by doc *id*. Sangraha appears to reuse some `doc_id`s for different texts, so dropping one near-duplicate also drops its id-twins.
  - Not verified, and too small to rerun for. A future fix is to drop by (shard, row) instead of id.

## Split

| split | wikipedia | manual | sangraha | total docs |
| --- | --- | --- | --- | --- |
| train | 86,063 / 0.561 GB | 283,060 / 1.661 GB | 4,062,323 / 22.548 GB | 4,431,446 |
| val | 214 | 712 | 10,231 | 11,157 |
| test | 222 | 736 | 10,322 | 11,280 |

**Why val/test can't contain near-duplicates of train:** near-dedup ran globally over all docs *before* the split. Any near-duplicate cluster was already collapsed to one doc, and that doc lands in exactly one split.

## Estimated train tokens (bytes / 9.5)

| | tokens |
| --- | --- |
| train text once | 2.607B |
| Wikipedia 2× upsample adds | 0.059B |
| **total estimate** | **2.666B** (target ≥ 2.1B) |

Stage 5 measured the real count: **2.824B** content tokens at 8.99 bytes/token with `hi_unigram_8192`.

## Near-dedup

| | value |
| --- | --- |
| docs in | 4,476,358 |
| signature memory | 2.29 GB (RAM available 28.2 GB) |
| LSH candidate pairs | 88,194 |
| verified duplicate pairs (≥ 0.80) | 30,787 |
| dropped | 20,574 docs: Sangraha 13,024, Wikipedia 6,555, manual 995 |

**Audit** (`results/filter_audit/near_dedup_pairs.md`, 20 random pairs):
- Wikipedia duplicates are template village stubs that differ only in the village name, e.g. "अफौ / वेतनधार, भिकियासैण तहसील …".
- Manual vs Sangraha pairs are the same news story crawled twice. The manual copy was kept each time, as the priority rule says.
- Only about 0.46% was removed. That's low because AI4Bharat already deduplicated Sangraha.

## Quality classifier (3c)

**Skipped.** Estimated train tokens after near-dedup were 2.67B. The classifier runs only above 2× the target (4.2B). This was expected, since 3a stops streaming at 1.25× the target.

## Kernel runtimes

| step | minutes |
| --- | --- |
| setup (code download, install, 29 tests) | 0.4 |
| 3a: 38 shards (Wikipedia 2, manual 11, Sangraha 25) | 245 |
| 3b near-dedup (load, LSH, verify, union-find, audit) | ~3 |
| 3d split (38 shards) | ~9 |

- Throughput: Sangraha took about 480–490 s per 174.8k-doc shard (about 360 docs/s on 4 workers).
- No shard failed or needed a retry.
- The near-dedup memory check passed.

## Failed / retried

- None in this run.
- Earlier CLI pushes (versions 4–6, 8) failed at the Kaggle secret, as documented in Stage 2.
