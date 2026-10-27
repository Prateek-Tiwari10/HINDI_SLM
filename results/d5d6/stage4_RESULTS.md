# Stage 4 — Short LR sweep (D6), Kaggle T4×2

**Status:** done. **Gate 4: peak LR = 1e-3**, chosen by the pre-registered rule and written into `configs/pretrain_61m.yaml`.
- **Kaggle run:** `prateek99491/hindi-lm-d6-pretrain`, version 2, second part of `train/d6_pipeline.py`. It was published to HF `d6_results/` at about 13:15 UTC.
- **Driver:** `python -m train.stage4`. Raw outputs: [stage4.json](stage4.json), [sweep_lr_3e-4.jsonl](sweep_lr_3e-4.jsonl), [sweep_lr_6e-4.jsonl](sweep_lr_6e-4.jsonl), [sweep_lr_1e-3.jsonl](sweep_lr_1e-3.jsonl).
- **Who confirmed:** Prateek asked for the stages to run through Stage 5 without waiting for him. So the rule's pick was applied automatically, with no manual confirmation. If you want a different LR, change it before Stage 5 resumes.

## What ran

- **Config:** the real one (micro-batch 8 × grad_accum 16 × 2 GPUs = 262,144 tokens/step, `torch.compile`, fp16).
- **Schedule:** shortened to 600 steps, ≈ 157M tokens: warmup 200, cosine to 10% of peak at step 600.
- **Evals:** val on the 512-window subset every 100 steps; all 4,468 val windows at step 600.
- **Storage:** each run is in its own HF folder `sweep/lr_X/`, holding a final checkpoint, `best/` and the log.

## Results

| peak LR | final val loss (all 4,468 windows) | final val BPB | val PPL | final train loss | spikes | GradScaler decreases (min scale) | max grad norm | tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3e-4 | 3.9689 | 0.6160 | 52.9 | 3.9993 | 0 | 0 (65,536) | 3.27 | 57,720 |
| 6e-4 | 3.7405 | 0.5805 | 42.1 | 3.7689 | 0 | 0 (65,536) | 2.64 | 57,701 |
| **1e-3** | **3.6048** | **0.5594** | **36.8** | **3.6339** | **0** | **0 (65,536)** | **1.68** | **57,706** |

Val loss on the 512-window subset during the runs:

| step | 3e-4 | 6e-4 | 1e-3 |
| --- | --- | --- | --- |
| 100 | 5.908 | 5.526 | 5.323 |
| 200 | 4.948 | 4.689 | 4.790 |
| 300 | 4.425 | 4.207 | 4.158 |
| 400 | 4.150 | 3.941 | 3.846 |
| 500 | 4.000 | 3.780 | 3.656 |
| 600 (all windows) | 3.969 | 3.741 | **3.605** |

Plot: [plots/stage4_lr_sweep.png](plots/stage4_lr_sweep.png) (train loss thin, val loss with markers, one colour per LR).

## Rule applied

1. Ranking by final val loss: **1e-3 (3.6048)**, then 6e-4 (3.7405), then 3e-4 (3.9689).
2. 1e-3 won, so its spike check applies: a train-loss point more than 0.5 above the mean of the previous 20 logged points (200 steps). It had **0 spikes**, no GradScaler underflow and no halt. So 1e-3 stands.

**Pick: peak LR 1e-3.** The gap to 6e-4 is 0.136 nats (−3.6%), well beyond noise. The order was stable from step 300 on.

## Observations

- **At step 200 (end of warmup), 1e-3 was briefly behind 6e-4** (val 4.790 vs 4.689). The higher peak LR costs a little during the ramp, then wins clearly once the cosine decay starts.
- **Nothing went unstable at any LR.** There were no overflow-driven GradScaler decreases: the scale stayed at its initial 65,536, since 600 steps is under the 2,000-step growth interval. Grad norms stayed small, and the largest pre-clip norm came at the first step.
- **Real-data throughput was 57.7k tokens/s (20.5% MFU),** faster than the Stage 2 benchmark's 51.8k. grad_accum 16 amortizes the DDP all-reduce better than the benchmark's grad_accum 4.
  - Each 600-step run took about 47 min, including compile and the final checkpoint upload.
  - At this speed one epoch is 7,405 × 4.54 s ≈ **9.3 h**.
- **Caveat:** a 600-step sweep favours a higher LR more than a full 7,405-step run would. The rule was fixed in advance, and 1e-3 is a common peak for models of this size, so it is used as is.
