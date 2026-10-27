"""Throughput / memory benchmark of one setting: torchrun --nproc_per_node=2 -m train.bench --micro-batch 16 --compile 0

Random tokens (speed does not depend on the data), the real model, DDP, fp16 autocast +
GradScaler, fused AdamW, gradient accumulation with no_sync. A few warm-up optimizer steps
(compilation happens there), then timed steps. Rank 0 appends one JSON line to --out.
One setting per process, so an out-of-memory crash only loses that setting.
"""

import argparse
import contextlib
import json
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.config import ModelConfig
from model.model import HindiLlama
from train.optim import build_optimizer
from train.train import flops_per_token


def main() -> None:
    """Benchmark one (micro_batch, compile) setting and append the result to --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--micro-batch", type=int, required=True)
    p.add_argument("--compile", type=int, choices=[0, 1], default=0)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--model-config", default="configs/model_61m.yaml")
    p.add_argument("--peak-tflops", type=float, default=65.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    mcfg = ModelConfig.from_yaml(args.model_config)
    model = HindiLlama(mcfg).to(device)
    n_params = model.num_params()
    opt = build_optimizer(model, {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8}, 3e-4)
    scaler = torch.amp.GradScaler("cuda")
    ddp = DDP(model, device_ids=[device.index])
    run = torch.compile(ddp) if args.compile else ddp
    B, T, accum = args.micro_batch, args.seq_len, args.grad_accum
    gen = torch.Generator(device=device).manual_seed(rank)
    x = torch.randint(0, mcfg.vocab_size, (B, T + 1), device=device, generator=gen)

    def one_step() -> float:
        total = 0.0
        for m in range(accum):
            with (contextlib.nullcontext() if m == accum - 1 else ddp.no_sync()):
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = run(x[:, :-1], x[:, 1:])
                scaler.scale(loss / accum).backward()
            total += loss.item() / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return total

    t_compile = time.time()
    losses = [one_step() for _ in range(args.warmup)]
    torch.cuda.synchronize()
    warmup_s = time.time() - t_compile
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    losses += [one_step() for _ in range(args.steps)]
    torch.cuda.synchronize()
    dt = time.time() - t0
    tokens = args.steps * accum * B * T * world
    tps = tokens / dt
    mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        rec = {
            "micro_batch": B, "compile": bool(args.compile), "grad_accum": accum, "world": world,
            "tok_per_s": round(tps), "s_per_262k_step": round(262_144 / tps, 3),
            "mfu": round(tps * flops_per_token(n_params, mcfg, T) / (args.peak_tflops * 1e12 * world), 4),
            "peak_mem_gb": round(mem.item(), 2), "warmup_s": round(warmup_s, 1),
            "finite": all(math.isfinite(v) for v in losses), "last_loss": round(losses[-1], 4),
            "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        }
        print(json.dumps(rec), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
