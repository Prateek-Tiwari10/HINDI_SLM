# Stage 2 — GPU smoke test + benchmark (D6), Kaggle T4×2

**Status:** done. **Gate 2: PASS.**
- **Kaggle run:** `prateek99491/hindi-lm-d6-smoke`, version 2. Machine: 2 × Tesla T4, torch 2.11.0+cu128, Python 3.13.
- **Started by API push:** the code is embedded and the run uses random tokens, so it needs no HF access.
- **Command:** `python -m train.stage2` (after `pytest`: 80/80 passed).
- **Raw outputs:** [stage2.json](stage2.json), [bench.jsonl](bench.jsonl).

**Version 1 failed (code bug, fixed):** `HindiLlama.forward` called `targets.view(-1)` on the shifted slice `x[:, 1:]`, which is not contiguous. The Stage 1 tests only passed contiguous targets, so they missed it.
- **Fix:** `targets.reshape(-1)`, plus a new test `test_loss_accepts_shifted_slices`.
- **Re-run:** version 2 ran the same stage unchanged otherwise.

## 1. Overfit one batch

Fixed random batch 4 × 1024 (uniform random tokens, the hardest case), LR 1e-3 constant, AdamW, fp16 autocast + GradScaler, clip 1.0, 300 steps on one T4.

| | value |
| --- | --- |
| loss at step 1 | 9.361 (≈ ln 10240 + 0.13, as expected at init) |
| first step below 0.5 | **116** |
| loss at step 300 | **0.00021** |
| NaN / inf | none |
| time | 68 s |

Gate (< 0.5 and finite): **PASS**. Plot: [plots/stage2_overfit.png](plots/stage2_overfit.png).

## 2. Benchmark (2 × T4, DDP, fp16, fused AdamW, grad_accum 4, random tokens)

Each setting ran in its own `torchrun`, so a crash loses only that setting. There were 3 warm-up optimizer steps (compilation happens there), then 6 timed steps.

| micro-batch | compile | tokens/s (both GPUs) | s per 262,144-token step | MFU | peak mem / GPU (GB) | status |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | off | 38,947 | 6.73 | 13.8% | 5.73 | ok |
| 16 | off | 35,317 | 7.42 | 12.5% | 10.10 | ok |
| 24 | off | — | — | — | — | failed after 9 s (OOM, see note) |
| **8** | **on** | **51,779** | **5.06** | **18.4%** | **4.01** | **ok — chosen** |
| 16 | on | 47,897 | 5.47 | 17.0% | 6.67 | ok |
| 24 | on | 48,668 | 5.39 | 17.3% | 9.40 | ok |

MFU uses 6N + 12·L·d·T FLOPs/token (N = 61,128,320) against 65 TFLOPs fp16 per T4. Plot: [plots/stage2_throughput.png](plots/stage2_throughput.png).

**Choice: micro-batch 8, `torch.compile` on.**
- **Compile speedup:** 1.33× the best eager setting (51,779 vs 38,947 tokens/s). That clears the ≥ 15% rule, and the losses were finite.
- **`configs/pretrain_61m.yaml` now has:**
  - `micro_batch: 8`, down from the planned 16;
  - `grad_accum: 16`, so 2 · 8 · 16 · 1024 = **262,144 tokens/step, unchanged**;
  - `compile: true`.

**Notes:**
- **Smaller micro-batches are faster.** The fp32 upcast of the (B, 1024, 10240) logits for the loss is memory-bound and grows with B. Larger batches add no compute efficiency on T4 at this model size.
- **Micro-batch 24 without compile crashed after 9 s.** The traceback tail captured in the log stops at torchrun's `ChildFailedError`, so the OOM message itself was not captured. Peak memory goes 5.7 → 10.1 GB from micro-batch 8 to 16, which extrapolates to about 14.5 GB at 24, the limit of a 15 GB T4. Either way, this setting is slower than the chosen one, so the choice is the same.

## 3. Projection for one epoch

| | value |
| --- | --- |
| steps | floor(1,895,725 windows / 256) = **7,405** |
| tokens | 1,941,176,320 |
| throughput | 51,779 tokens/s |
| wall-clock | **10.41 h** |
| GPU-hours (2 × wall) | **20.83** (gate: ≤ 30) |
| 11h15m sessions | 1 by throughput alone. **Expect 2** once data download, compile warm-up, evals and checkpoints are added, and Stages 3–4 run in the same first session (see below). |

Gate (projection computed, ≤ 30 GPU-hours): **PASS**.

## Changes made for Stages 3–5

- **Checkpoint uploads run in a background thread** (`CheckpointManager.upload_async`). About 15 uploads of 735 MB would otherwise stall both GPUs for minutes each. The log is snapshotted before upload. The next save waits for the previous upload, and the simulated crash in Stage 3 waits for the step-80 upload, so the test still sees step 80 as `latest`.
- **One kernel for Stages 3 → 4 → 5.** Kernels pushed by API cannot read `HF_TOKEN`, and the browser connection I had planned to click Save Version with dropped. So `hindi-lm-d6-pretrain` runs `train/d6_pipeline.py`, which enforces the gates in code:
  - **Stage 3 fails:** stop.
  - **Stage 4:** the LR is picked by the pre-registered rule.
  - **Stage 5:** runs with that LR.
  - **Progress:** results are published to `d6_results/` in the checkpoints repo after each stage, so a re-run skips finished stages.
  - **What it replaces:** the separate `hindi-lm-d6-resume` and `hindi-lm-d6-sweep` kernels in the instructions.
- **Stop rules in `train.py`:** a NaN loss or a GradScaler scale below 1 halts the run without saving. Val loss rising for 3 consecutive evals saves and halts.
