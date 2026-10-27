"""Stage 2 driver (Kaggle T4×2): overfit one batch, benchmark settings, project the full run.

Writes everything to --out (default /kaggle/working/out/d5d6/): stage2.json, bench.jsonl,
overfit_losses.json and the plots. Needs no HF access (random tokens), so the kernel can
start from an API push.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from eval.plots import plot_bench, plot_overfit
from model.config import ModelConfig
from model.model import HindiLlama
from train.optim import build_optimizer

TRAIN_TOKENS = 1_943_118_661          # tokenized/v2 meta.json, train incl. bos/eos
SEQ, WINDOW = 1024, 1025
GLOBAL_WINDOWS = 2 * 16 * 8           # world × micro_batch × grad_accum of the D6 config
SESSION_H = 11.25


def overfit(steps: int = 300, lr: float = 1e-3) -> list[float]:
    """Train on one fixed random 4×1024 batch (single GPU, fp16 + GradScaler); return the losses."""
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    mcfg = ModelConfig.from_yaml("configs/model_61m.yaml")
    model = HindiLlama(mcfg).to(device).train()
    opt = build_optimizer(model, {"weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8}, lr)
    scaler = torch.amp.GradScaler("cuda")
    batch = torch.randint(0, mcfg.vocab_size, (4, SEQ + 1), device=device)
    losses = []
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    del model, opt
    torch.cuda.empty_cache()
    return losses


def main() -> None:
    """Run the three Stage 2 parts and write the summary."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d5d6")
    args = p.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    summary: dict = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                     "torch": torch.__version__}
    print(f"[stage2] {summary}", flush=True)

    # 1. overfit one batch
    t0 = time.time()
    losses = overfit()
    (out / "overfit_losses.json").write_text(json.dumps(losses))
    plot_overfit(losses, out / "plots" / "stage2_overfit.png")
    summary["overfit"] = {"first": losses[0], "min": min(losses), "last": losses[-1],
                          "steps_to_below_0.5": next((i + 1 for i, v in enumerate(losses) if v < 0.5), None),
                          "finite": all(math.isfinite(v) for v in losses), "seconds": round(time.time() - t0, 1),
                          "pass": losses[-1] < 0.5 and all(math.isfinite(v) for v in losses)}
    print(f"[stage2] overfit: {summary['overfit']}", flush=True)

    # 2. benchmark, one torchrun per setting
    bench_path = out / "bench.jsonl"
    bench_path.unlink(missing_ok=True)
    attempts = []
    for comp in (0, 1):
        for mb in (8, 16, 24):
            cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "-m", "train.bench",
                   "--micro-batch", str(mb), "--compile", str(comp), "--out", str(bench_path)]
            print(f"[stage2] $ {' '.join(cmd)}", flush=True)
            t = time.time()
            try:
                res = subprocess.run(cmd, timeout=1500, capture_output=True, text=True)
                ok, tail = res.returncode == 0, (res.stdout + res.stderr)[-1500:]
            except subprocess.TimeoutExpired:
                ok, tail = False, "timeout after 1500 s"
            reason = "ok" if ok else ("OOM" if "OutOfMemory" in tail or "out of memory" in tail else "failed")
            attempts.append({"micro_batch": mb, "compile": bool(comp), "status": reason,
                             "seconds": round(time.time() - t, 1)})
            print(f"[stage2] -> {reason} in {time.time() - t:.0f}s\n{tail if not ok else ''}", flush=True)
    records = [json.loads(ln) for ln in bench_path.read_text().splitlines()] if bench_path.exists() else []
    summary["bench_attempts"] = attempts
    summary["bench"] = records
    plot_bench(records, out / "plots" / "stage2_throughput.png")

    # pick: fastest finite run; compile only if >= 15% faster than the best non-compiled run
    stable = [r for r in records if r["finite"]]
    best_eager = max((r for r in stable if not r["compile"]), key=lambda r: r["tok_per_s"], default=None)
    best_comp = max((r for r in stable if r["compile"]), key=lambda r: r["tok_per_s"], default=None)
    choice = best_eager
    if best_comp and best_eager and best_comp["tok_per_s"] >= 1.15 * best_eager["tok_per_s"]:
        choice = best_comp
    summary["choice"] = choice
    summary["compile_speedup"] = (round(best_comp["tok_per_s"] / best_eager["tok_per_s"], 3)
                                  if best_comp and best_eager else None)

    # 3. projection for one epoch of the real config
    steps = (TRAIN_TOKENS // WINDOW) // GLOBAL_WINDOWS
    tokens = steps * GLOBAL_WINDOWS * SEQ
    if choice:
        wall_h = tokens / choice["tok_per_s"] / 3600
        summary["projection"] = {"steps": steps, "tokens": tokens, "tok_per_s": choice["tok_per_s"],
                                 "wall_hours": round(wall_h, 2), "gpu_hours": round(2 * wall_h, 2),
                                 "sessions_11h15m": math.ceil(wall_h / SESSION_H),
                                 "gate_over_30_gpu_hours": 2 * wall_h > 30}
    print(f"[stage2] choice: {choice}\n[stage2] projection: {summary.get('projection')}", flush=True)
    (out / "stage2.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
