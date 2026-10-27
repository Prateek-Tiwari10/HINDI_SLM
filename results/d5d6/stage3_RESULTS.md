# Stage 3 — Kill-and-resume test (D6), Kaggle T4×2

**Status:** done. **Gate 3: PASS.** The resumed run is bit-identical to the uninterrupted one.
- **Kaggle run:** `prateek99491/hindi-lm-d6-pretrain`, version 2 (Save Version, HF_TOKEN attached), first part of `train/d6_pipeline.py`. Started 10:45 UTC; Stage 3 was published at about 10:56 UTC.
- **Driver:** `python -m train.stage3`. Raw outputs: [stage3.json](stage3.json), [resume_A.jsonl](resume_A.jsonl), [resume_B.jsonl](resume_B.jsonl), [resume_B_before_kill.jsonl](resume_B_before_kill.jsonl).

## What ran

- **Tiny schedule:**
  - 120 steps, checkpoint every 40, warmup 20, peak LR 6e-4;
  - micro-batch 8 × grad_accum 2 × 2 GPUs = 32,768 tokens/step;
  - real data, real model;
  - a train record every step, with a SHA-256 of each step's token ids gathered from both ranks.
- **Deterministic mode (`--deterministic`):**
  - `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - the SDPA math backend, because the memory-efficient T4 kernel uses atomics in backward;
  - `torch.compile` off.
- **Run A:** 120 steps straight through, to `resume_test/A/`.
- **Run B:** the same config, to `resume_test/B/`.
  - It is killed with `os._exit(1)` right after step 95, with no final save.
  - The local checkpoint folder is then deleted, so the resume must come from HF.
  - A fresh `torchrun` process resumes from `latest.txt` and runs to 120.

## Results

| check | value | result |
| --- | --- | --- |
| exit codes A / B killed / B resumed | 0 / 1 / 0 | as intended |
| `latest.txt` on HF after the kill | `step_000080` | correct (the step-95 state was never saved) |
| resume event | `resume from step_000080` at step 80 | correct |
| max \|loss A − loss B resumed\|, steps 81–120 (40 steps) | **0.0** | **PASS** (tolerance 1e-4) |
| token batches, steps 81–120 | all 40 hashes equal | **PASS** |
| max \|loss A − loss B before kill\|, steps 1–95 | 0.0 | deterministic baseline |
| max \|loss B before kill − loss B resumed\|, steps 81–95 | 0.0 | the resumed process replays exactly what the killed one had done |
| `resume_test/` deleted from HF | yes | clean |

Sample points (identical in A and resumed B):

| step | loss | batch hash |
| --- | --- | --- |
| 81 | 6.268281 | c230862821da6611 |
| 95 | 6.046062 | 0680fa0038d68591 |
| 120 | 5.989858 | 156d1067e71d91cf |

Plot: [plots/stage3_resume.png](plots/stage3_resume.png). The top panel shows the three loss curves, which overlap. The bottom panel shows |A − B resumed|, which is 0 at every step.

**What this proves:** the checkpoint restores model, optimizer, LR scheduler, GradScaler, step counter, tokens seen and every rank's RNG state completely. The data order is recomputed from `(step, micro, rank)` alone. The 15 steps lost to the crash (81–95) are replayed exactly.

## Val bytes for BPB (computed once in this run, cached)

[val_bytes.json](val_bytes.json) gives exact UTF-8 bytes of the scored target tokens. It is built from a per-token byte table, with bos/eos and control tokens at 0 bytes.

| | windows | target tokens | bytes | bytes / token |
| --- | --- | --- | --- | --- |
| subset used during training | 512 | 523,776 | 4,898,930 | 9.353 |
| all of val | 4,468 | 4,570,764 | 42,489,737 | 9.296 |

## Notes

- **Normal training is not bitwise reproducible.** Stage 5 runs `torch.compile` with the memory-efficient SDPA kernel, so a resumed session there continues from the exact saved state. Later steps can differ from a hypothetical uninterrupted run only by fp16 rounding noise. The state restore itself, which this stage tests, is exact.
- **Throughput here was 18k tokens/s,** because of deterministic mode, the math attention backend, no compile and grad_accum 2. It says nothing about Stage 5 speed.
- **Document masking:** windows cross document boundaries (bos/eos mark them). Document-level attention masking is a possible later change.
