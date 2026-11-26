# Progress — Hindi SLM

_Last updated: 2026-11-26_

Goal: a ~60M-parameter decoder-only Hindi LM trained from scratch (data, tokenizer, model, training), fine-tuned into a
Hinglish → Devanagari transliterator and served as a live demo. Decisions D1–D13 are in DECISIONS.md (local-only).

## Summary

| Area | Status |
| --- | --- |
| Stage 0 — environment, Kaggle/HF setup, scaffold | ✅ done |
| Stage 1 — source inventory | ✅ done |
| Stage 2 — normalization + filters (dry run, audited) | ✅ done |
| Stage 3 — full pipeline v1 (filter, dedup, split) | ✅ done |
| Stage 4 — tokenizer sweep v1 | ✅ done (`hi_unigram_8192`) |
| Stage 5 — tokenized dataset v1 | ✅ done (2.824B train tokens) |
| **Data v2** — 10B raw → ~2B rigorously cleaned | ✅ `final_v2/` built on HF |
| **Tokenizer v2** — ladder + 8-tokenizer grid on `final_v2` | ✅ done, accepted 2026-10-09 (`hi_unigram_10240`) |
| Tokenize `final_v2` → `tokenized/v2/` | ✅ done 2026-10-21 (`hindi-lm-runner` v22): 1.937B train tokens |
| D5 model — LLaMA-style decoder, 61,128,320 params | ✅ Stage 1 correctness passed 2026-10-21 ([results/d5d6/stage1_RESULTS.md](results/d5d6/stage1_RESULTS.md)) |
| D6 Stage 2 — GPU smoke test + benchmark | ✅ passed 2026-10-21: 51.8k tok/s (micro-batch 8 + compile), 20.8 GPU-h projected ([stage2](results/d5d6/stage2_RESULTS.md)) |
| D6 Stage 3 — kill-and-resume test | ✅ passed 2026-10-21: resumed run bit-identical (max loss diff 0.0, batches equal) ([stage3](results/d5d6/stage3_RESULTS.md)) |
| D6 Stage 4 — LR sweep | ✅ done 2026-10-21: peak LR **1e-3** (val 3.605 vs 3.741 @6e-4, 3.969 @3e-4; no spikes) ([stage4](results/d5d6/stage4_RESULTS.md)) |
| D6 Stage 5 — full pretraining (7,405 steps, 1.94B tokens) | ✅ done 2026-10-23 (2 sessions): best val 2.783 (subset, step 7,250), final all-window val 2.805 / BPB 0.435 / PPL 16.5; no halts ([stage5](results/d5d6/stage5_RESULTS.md)) |
| D6 Stage 6 — evaluation (val/test all windows, samples, repetition/Distinct) | ✅ done 2026-10-27: final `step_007405` val 2.805 / PPL 16.53 / BPB 0.435, test 2.816 / PPL 16.70 / BPB 0.438 (beats `best/` by 0.002) ([stage6](results/d5d6/stage6_RESULTS.md)) |
| D7 Stage 7a — Hinglish tokenizer extension (D8) | ✅ done 2026-10-27: char-level Latin + `<rom>`/`<dev>`, Devanagari-only output mask ([stage7a](results/d7_hinglish/stage7a_RESULTS.md)) |
| D7 Stage 7b — model surgery on 61M checkpoint (D9) | ✅ passed 2026-11-09: 61,169,280 params, base logits exact, KL 2.7e-5 with `mean` init, KV cache identical ([stage7b](results/d7_hinglish/stage7b_RESULTS.md)) |
| D7 Stage 7c — Hinglish pair dataset (D10) | ✅ 2026-11-17 (`romanize-v2`), rebuilt 2026-11-22 as `romanize-v2.1`: 600k synthetic train pairs, coverage 0.863, realism 0.643, 0 leakage ([stage7c](results/d7_hinglish/stage7c_RESULTS.md)) |
| D7 Stage 7d — SFT harness + gates G1–G6 | ✅ 2026-11-17: all six gates pass ([stage7d](results/d7_hinglish/stage7d_RESULTS.md)) |
| D7 Stage 7e — training (Phase A, LR bracket, main, ablations) (D11) | ✅ 2026-11-22: LR 6e-4; main `best/` clean dev CER **0.084**, WER 0.212, EM 0.148. Phase A adds nothing (A2 0.108 vs 0.110 at epoch 1); random init 0.126. Hindi LM loss 2.79 → 8.75 (forgotten) ([stage7e](results/d7_hinglish/stage7e_RESULTS.md)) |
| D7 Step 6 — full evaluation (`test_real`, `test_chat`, word-lookup baseline, ambiguous-word slices) (D12) | ✅ 2026-11-26, run once: normalized CER `test_real` **0.058** (clean 0.062) vs lookup baseline 0.213; `test_chat` **0.040** vs 0.126; ambiguity 0.843 vs 0.785, so no retrain ([step6](results/d7_hinglish/step6_RESULTS.md)) |
| Public weights — `Prateek-Tiwari10/hinglish-to-devanagari-61m` | ✅ 2026-11-26: `@7cf31e65`, safetensors + tokenizer + model card; logits identical to the checkpoint |
| Serve — FastAPI + Docker → GHCR → Azure Container Apps via GitHub Actions (D13) | 🟡 code, Docker and CI done; local gate passed 2026-11-26 (744 MB image, 7.1 s cold start, 302 ms p50 per chat sentence at 0.5 CPU). Waiting on Prateek's one-time Azure setup ([DEPLOY.md](DEPLOY.md)), then the live load test |
| Git: commit and push all work | ✅ through Step 6 + serve (2026-11-26) |

## v1 pipeline (Stages 0–5): done

**Stage 0 — setup** ([results/stage0_RESULTS.md](results/stage0_RESULTS.md))
- Kaggle account pinned to `prateek99491` via `kaggle/k.sh`; private HF repos `hindi-lm-data` and `hindi-lm-tokenizer` created.
- Rule: no compute on the laptop; everything runs as Kaggle kernels.

**Stage 1 — sources** ([results/stage1_RESULTS.md](results/stage1_RESULTS.md))
- Sampled Wikipedia, own news crawl, Sangraha Verified and FineWeb-2.
- Sangraha alone ≈ 10.4B tokens, so FineWeb-2 was excluded; Sangraha Synthetic excluded by design.

**Stage 2 — normalization + filters** ([results/stage2_RESULTS.md](results/stage2_RESULTS.md))
- Idempotent `normalize()` (NFC, ZWJ/ZWNJ rules, `|`→`।`) and six heuristic filters.
- Survival: Wikipedia 63%, news 77%, Sangraha 96%. 50 removed docs per filter audited — all junk.

**Stage 3 — full pipeline** ([results/stage3_RESULTS.md](results/stage3_RESULTS.md))
- Stream → filter → exact dedup → MinHash near-dedup → 99.5/0.25/0.25 doc split.
- 4.90M raw docs (27.4 GB) → 4.45M final docs (24.9 GB); only 25 of 100 Sangraha files needed.

**Stage 4 — tokenizer sweep** ([results/stage4_RESULTS.md](results/stage4_RESULTS.md))
- {BPE, Unigram} × {8K, 10K, 12K, 16K}; rule chose `hi_unigram_8192` (fertility 1.447, morphology 47.4%).

**Stage 5 — tokenized dataset** ([results/stage5_RESULTS.md](results/stage5_RESULTS.md))
- `tokenized/hi_unigram_8192/` on HF: train 2.824B tokens, val 6.84M, test 7.02M; 8.99 bytes/token.

## v2 rework: in progress

On 2026-10-05 the v1 dataset was judged not clean enough. New target: collect ~10B raw tokens and keep the best 1.8–2B.
v2 lives under `work/v2/`, `final_v2/`, `tokenized/v2/` on HF; v1 is kept for comparison.

**Data v2: done**
- All 100 Sangraha Verified files + Wikipedia + news crawl.
- New line-level cleaning ([data/lines.py](data/lines.py)): cross-document boilerplate lines and pattern lines ("यह भी पढ़ें", "ऐप डाउनलोड करें", …).
- Gopher/FineWeb-style doc filters added ([data/filters.py](data/filters.py)), global dedup.
- Web-doc selection: a fastText quality ranker was trained but **rejected** (audit showed it learned source/spelling style, not quality); outputs kept in `work/v2/quality/` as evidence. Now `quality.method: random` — a seeded random draw of web docs up to 2.0B train tokens, keeping all Wikipedia/news.
- Result: `final_v2/` train ≈ 17.5 GB of text, 7,873 val docs.

**Tokenizer v2: done** ([results/tokenizer_v2/stage4_selection.md](results/tokenizer_v2/stage4_selection.md))
- Ran on the TPU-VM kernel `hindi-lm-tpu` (for RAM).
- Sample-size ladder (5% / 15% / 30%): differences within tolerance, so 5% is enough.
- 8-tokenizer grid at 5%: chosen **`hi_unigram_10240`** — fertility 1.393, morphology 44.2%, embedding share 8.7%.
- Recorded in [DECISIONS.md](DECISIONS.md) D4; model in HF `hindi-lm-tokenizer/sweep_v2/`.

**Tokenized dataset v2: done** (`tokenized/v2/hi_unigram_10240/` on HF)

| split | docs | tokens incl. bos/eos | content tokens |
| --- | --- | --- | --- |
| train (Wikipedia ×2) | 3,212,257 | 1,943,118,661 | **1.937B** |
| val | 7,873 | 4,575,239 | 4.56M |
| test | 7,831 | 4,668,557 | 4.65M |

- Shards: `train_000.bin … train_003.bin` (500M tokens each, last 443M), `val.bin`, `test.bin`, `meta.json`; `uint16`, bos=2, eos=3, seed 1337.
- Train mix: Sangraha 1.672B (86.4%), news crawl 0.142B (7.3%), Wikipedia 2 × 61.2M (6.3%).
- Measured **9.31 bytes/token** (v1: 8.99 with the 8K vocab).
- Within the 1.8–2B target; ≈ 1.08 epochs for the 1.8B-token D1 budget.

## D5/D6: model and pretraining (in progress)

Instructions: [CLAUDE_CODE_INSTRUCTIONS_D5_D6.md](CLAUDE_CODE_INSTRUCTIONS_D5_D6.md). Reports go in [results/d5d6/](results/d5d6/).

- **Architecture (fixed):** d_model 640, 12 layers, 10 query heads / 5 K/V heads (GQA, head_dim 64), SwiGLU 1728, RoPE θ=10k, RMSNorm pre-norm, no biases, tied embeddings, context 1024.
- **Stage 1 — correctness: ✅ passed** (Kaggle CPU `hindi-lm-d6-tests` v1). 74/74 tests pass; exactly 61,128,320 params (54,574,080 non-embedding). Causality, RoPE, manual = SDPA, GQA = MHA, KV-cache = no-cache, init std and init loss all checked.
- **Stage 2 — GPU smoke test + benchmark: ✅ passed** (`hindi-lm-d6-smoke` v2). Overfit loss 9.36 → 0.0002; best setting micro-batch 8 + `torch.compile` = 51,779 tok/s on 2×T4 (18.4% MFU); one epoch = 7,405 steps ≈ 10.4 h wall / 20.8 GPU-h.
- **Stages 3–5** run in one kernel, `hindi-lm-d6-pretrain` (`train/d6_pipeline.py`), with gates enforced in code; results land in HF `hindi-lm-checkpoints/d6_results/`. Done in 2 sessions; final checkpoint `step_007405`, `best/` = step 7,250.
- **Stage 6 — evaluation: ✅ done** (`hindi-lm-d6-eval`, 2026-10-27, ~6 min on one T4). Scored both checkpoints on all 4,468 val and 4,559 test windows; the final `step_007405` is slightly better than `best/` (val 2.8050 vs 2.8071), so it is the pretrained model. Greedy decoding loops (4-gram repetition 0.78); T=0.7 top-k 50 is fluent and on topic (repetition 0.12, Distinct-2 0.70); facts are unreliable, as expected at 61M with no instruction tuning. Samples in [results/d5d6/samples.md](results/d5d6/samples.md).

## Next steps

1. **Prateek:** one-time Azure setup in [DEPLOY.md](DEPLOY.md):
   - push, then make the GHCR package public;
   - `az` commands for the resource group, environment and app;
   - the OIDC app;
   - the repo variables;
   - run **deploy-demo**.
2. **Then:**
   - run **loadtest-demo** against the live URL, with the rate limit temporarily raised;
   - put p50/p95 and requests per second into the README and step6_RESULTS.md;
   - put the live URL at the top of the README.
3. **Optional:** pick a licence for the public model card.
4. **Future work:** plain-Hindi replay during fine-tuning to undo the forgetting (D11/D12).

## Open notes

- Stage 5 session 1 (2026-10-21) stopped at step 6,800/7,405 on "HF private storage limit". Pruned checkpoints still used storage via git history (repo showed 4.75 GB but used 20.7 GB). On 2026-10-22: deleted `sweep/` (logs kept in `d6_results/`), squashed history, removed unreferenced checkpoint blobs; repo now 1.74 GB. Session 2 resumed from `step_006500` and finished step 7,405 on 2026-10-23.

- v1 split dropped 1,901 extra Sangraha docs (0.05%) because some Sangraha `doc_id`s repeat; fix is to drop by (shard, row).
- Kaggle runs that need `HF_TOKEN` must be started with **Save Version** in the Kaggle UI (API pushes can't read secrets).
- Optional later upgrade: LLM-labelled quality classifier (FineWeb-Edu style) to replace the random web draw.
- HF storage counts git history: squash and delete orphaned LFS objects after pruning checkpoints (Stage 7e v2 stopped on the cap at main step 18,750; the 7e pipeline now does this after every run).
