"""D6 pretraining loop. Launch with: torchrun --nproc_per_node=2 -m train.train [--overrides].

fp16 autocast + GradScaler, DDP over 2 T4s, gradient accumulation, warmup + cosine LR,
deterministic resumable data order, checkpoints to HF every N steps and on a wall-clock
budget, and resume from latest.txt on start. Rank 0 does all writing and uploading.

Log records (one JSON object per line, logs/pretrain_61m.jsonl), `step` = completed steps:
    {"type": "train", step, tokens, loss, lr, grad_norm, scale, tok_per_s, mfu, mem_gb}
    {"type": "val", step, tokens, val_loss, val_ppl, val_bpb, windows}
    {"type": "event", step, event, ...}   (fresh start, resume, checkpoint, stop)
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from model.config import ModelConfig
from model.model import HindiLlama
from train.checkpoint import CheckpointManager, rng_state, set_rng_state, step_name
from train.data import TrainBatches, ValWindows, download_split, target_bytes, token_byte_lengths
from train.optim import build_optimizer, build_scheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI overrides of configs/pretrain_61m.yaml (used by the resume test and the LR sweep)."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain_61m.yaml")
    p.add_argument("--peak-lr", type=float)
    p.add_argument("--total-steps", type=int, help="default: one epoch")
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--hf-prefix", help="checkpoint folder in the HF repo, e.g. sweep/lr_3e-4/")
    p.add_argument("--save-every", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--log-path", help="local jsonl path")
    p.add_argument("--compile", type=int, choices=[0, 1])
    p.add_argument("--budget-s", type=float, help="wall-clock budget in seconds")
    p.add_argument("--kill-at-step", type=int, help="os._exit(1) right after this step (resume test)")
    p.add_argument("--log-batch-hashes", action="store_true", help="log a hash of each step's token ids")
    p.add_argument("--deterministic", action="store_true", help="torch deterministic algorithms (resume test)")
    p.add_argument("--final-full-eval", type=int, choices=[0, 1], default=1,
                   help="evaluate all val windows when the schedule ends")
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    """Pretrain config with CLI overrides applied."""
    cfg = yaml.safe_load(Path(args.config).read_text())
    overrides = {
        ("optim", "peak_lr"): args.peak_lr, ("optim", "warmup_steps"): args.warmup_steps,
        ("total_steps",): args.total_steps, ("micro_batch",): args.micro_batch,
        ("grad_accum",): args.grad_accum, ("checkpoint", "hf_prefix"): args.hf_prefix,
        ("checkpoint", "every_steps"): args.save_every, ("logging", "eval_every_steps"): args.eval_every,
        ("logging", "every_steps"): args.log_every, ("logging", "jsonl"): args.log_path,
        ("compile",): None if args.compile is None else bool(args.compile),
        ("checkpoint", "wall_clock_budget_s"): args.budget_s,
    }
    for keys, value in overrides.items():
        if value is not None:
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
    assert cfg["optim"]["peak_lr"] is not None, "peak_lr is unset (chosen in Stage 4); pass --peak-lr"
    return cfg


def flops_per_token(n_params: int, mcfg: ModelConfig, seq_len: int) -> float:
    """Training FLOPs per token: 6N + 12·L·d·T (the second term is attention scores)."""
    return 6 * n_params + 12 * mcfg.n_layers * mcfg.d_model * seq_len


class Logger:
    """Append-only JSONL log on rank 0, also echoed to stdout."""

    def __init__(self, path: Path, rank: int) -> None:
        self.path, self.rank = path, rank

    def __call__(self, record: dict) -> None:
        """Write one record (rank 0 only)."""
        if self.rank != 0:
            return
        line = json.dumps(record)
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")

    def truncate_after(self, step: int) -> None:
        """Drop records past `step`: a killed run may have logged beyond its last checkpoint."""
        if self.rank != 0 or not self.path.exists():
            return
        keep = [ln for ln in self.path.read_text().splitlines() if ln and json.loads(ln).get("step", 0) <= step]
        self.path.write_text("".join(ln + "\n" for ln in keep))


@torch.no_grad()
def evaluate(model: HindiLlama, val: ValWindows, ids: list[int], rank: int, world: int,
             device: torch.device, batch: int = 16) -> float:
    """Mean val loss (nats/token) over the given windows; each rank takes every world-th window."""
    was_training = model.training
    model.eval()
    mine = ids[rank::world]
    total = torch.zeros(2, device=device, dtype=torch.float64)          # (loss sum, target count)
    for i in range(0, len(mine), batch):
        x, y = val.batch(mine[i:i + batch], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        total[0] += loss.double() * y.numel()
        total[1] += y.numel()
    if world > 1:
        dist.all_reduce(total)
    model.train(was_training)
    return (total[0] / total[1]).item()


def main(argv: list[str] | None = None) -> None:
    """Train (or resume) one run as configured; see the module docstring."""
    args = parse_args(argv)
    cfg = load_config(args)
    t_start = float(os.environ.get("RUN_STARTED_AT", time.time()))   # set by the Kaggle runner
    if args.deterministic:                                 # before any CUDA work creates cuBLAS handles
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        # SDPA's memory-efficient backward (the T4 kernel) uses atomics; the math backend does not
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    def barrier() -> None:
        if distributed:
            dist.barrier()

    # ---------- data ----------
    if rank == 0:
        download_split(cfg)
    barrier()
    meta, paths = download_split(cfg)                      # no-op downloads now; returns paths
    seq_len, B, accum = cfg["data"]["seq_len"], cfg["micro_batch"], cfg["grad_accum"]
    batches = TrainBatches(paths["train"], seq_len, B, accum, world, rank, cfg["data"]["order_seed"])
    total_steps = cfg["total_steps"] or batches.steps_per_epoch
    tokens_per_step = batches.global_batch_windows * seq_len
    val = ValWindows(paths["val"][0], seq_len)
    val_subset = list(range(min(cfg["data"]["val_subset_windows"], val.n_windows)))
    val_all = list(range(val.n_windows))

    import sentencepiece as spm                            # noqa: PLC0415 - only needed here
    from data import hub                                   # noqa: PLC0415
    tok_path = hub.download(cfg["tokenizer"]["hf_repo"], cfg["tokenizer"]["path"],
                            Path(cfg["data"]["local_dir"]) / "tokenizer", repo_type="model")
    byte_len = token_byte_lengths(spm.SentencePieceProcessor(model_file=str(tok_path)))
    bytes_subset = target_bytes(val, val_subset, byte_len)
    bytes_all = target_bytes(val, val_all, byte_len)

    # ---------- model / optimizer ----------
    mcfg = ModelConfig.from_yaml(cfg["model_config"])
    mcfg.attn_impl = cfg["attn_impl"]
    torch.manual_seed(cfg["data"]["order_seed"])           # same init on every rank
    model = HindiLlama(mcfg).to(device)
    n_params = model.num_params()
    opt = build_optimizer(model, cfg["optim"], cfg["optim"]["peak_lr"])
    sched = build_scheduler(opt, cfg["optim"]["warmup_steps"], total_steps, cfg["optim"]["min_lr_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ck = cfg["checkpoint"]
    log_path = Path(cfg["logging"]["jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(ck["hf_repo"], ck["hf_prefix"], ck["local_dir"], ck["keep_last"], log_path.name)
    log = Logger(log_path, rank)

    # ---------- resume or fresh ----------
    latest, ckpt_path, remote_log = None, None, None
    if rank == 0:
        latest = manager.latest()
        if latest:
            ckpt_path, remote_log = manager.download(latest)
    if distributed:
        obj = [latest, str(ckpt_path) if ckpt_path else None]
        dist.broadcast_object_list(obj, src=0)
        latest, ckpt_path = obj[0], Path(obj[1]) if obj[1] else None
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    if latest:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step, tokens_seen, best_val = state["step"], state["tokens_seen"], state["best_val_loss"]
        assert state["order_seed"] == cfg["data"]["order_seed"], "data-order seed changed between sessions"
        set_rng_state(state["rng"][rank])
        if rank == 0:
            if remote_log is not None:
                log_path.write_bytes(Path(remote_log).read_bytes())
            log.truncate_after(start_step)
        del state
    elif rank == 0 and log_path.exists():
        log_path.unlink()
    barrier()
    log({"type": "event", "step": start_step, "event": "resume" if latest else "fresh_start",
         "from": latest, "total_steps": total_steps, "train_windows": batches.n_windows,
         "global_batch_windows": batches.global_batch_windows, "tokens_per_step": tokens_per_step,
         "micro_batch": B, "grad_accum": accum, "world": world, "peak_lr": cfg["optim"]["peak_lr"],
         "n_params": n_params, "val_bytes_subset": bytes_subset, "val_bytes_all": bytes_all,
         "torch": torch.__version__})
    if rank == 0:
        print(f"[train] total steps = floor({batches.n_windows:,} windows / {batches.global_batch_windows} "
              f"per step) = {batches.steps_per_epoch:,} (running {total_steps:,})", flush=True)

    train_model: torch.nn.Module = model
    ddp = None
    if distributed:
        ddp = DDP(model, device_ids=[local_rank])
        train_model = ddp
    if cfg["compile"]:
        train_model = torch.compile(train_model)

    fpt = flops_per_token(n_params, mcfg, seq_len)
    peak_flops = cfg["logging"]["peak_tflops_per_gpu"] * 1e12 * world
    log_every, eval_every = cfg["logging"]["every_steps"], cfg["logging"]["eval_every_steps"]
    save_every, budget = ck["every_steps"], ck["wall_clock_budget_s"]
    best_state, best_info, best_uploaded = None, None, True

    def save(step: int) -> None:
        """All ranks: gather RNG; rank 0: write, upload (with best/ if it improved)."""
        nonlocal best_uploaded
        rngs = [None] * world
        if distributed:
            dist.all_gather_object(rngs, rng_state())
        else:
            rngs = [rng_state()]
        if rank == 0:
            state = {
                "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step, "tokens_seen": tokens_seen,
                "best_val_loss": best_val, "rng": rngs, "order_seed": cfg["data"]["order_seed"],
                "model_config": mcfg.to_dict(), "train_config": cfg,
            }
            path = manager.save_local(step, state)
            del state
            log({"type": "event", "step": step, "event": "checkpoint", "name": step_name(step),
                 "elapsed_s": round(time.time() - t_start)})
            best = None
            if not best_uploaded and best_state is not None:
                best = (best_state, best_info)
                best_uploaded = True
            manager.upload_async(step, path, log_path, best)
        barrier()

    # ---------- loop ----------
    model.train()
    t_log, steps_since_log = time.time(), 0
    step_times: list[float] = []
    saved_at = start_step
    step = start_step
    halted = rising = False
    val_history: list[float] = []
    while step < total_steps:
        t0 = time.time()
        lr = sched.get_last_lr()[0]
        loss_sum = torch.zeros((), device=device)
        hasher = hashlib.sha256() if args.log_batch_hashes else None
        for m in range(accum):
            x, y = batches.get(step, m, device)
            if hasher is not None:
                hasher.update(x.cpu().numpy().tobytes())
            sync = ddp is None or m == accum - 1
            with (contextlib.nullcontext() if sync else ddp.no_sync()):
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    _, loss = train_model(x, y)
                scaler.scale(loss / accum).backward()
            loss_sum += loss.detach()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        tokens_seen += tokens_per_step
        steps_since_log += 1

        if step % log_every == 0 or step == total_steps or hasher is not None:
            stats = torch.stack([loss_sum / accum, grad_norm.float()])
            if distributed:
                dist.all_reduce(stats)
                stats /= world
            batch_hash = None
            if hasher is not None:
                digests = [None] * world
                if distributed:
                    dist.all_gather_object(digests, hasher.hexdigest())
                else:
                    digests = [hasher.hexdigest()]
                batch_hash = hashlib.sha256("".join(digests).encode()).hexdigest()[:16]
            dt = time.time() - t_log
            tps = steps_since_log * tokens_per_step / dt
            rec = {"type": "train", "step": step, "tokens": tokens_seen, "loss": round(stats[0].item(), 6),
                   "lr": lr, "grad_norm": round(stats[1].item(), 4), "scale": scaler.get_scale(),
                   "tok_per_s": round(tps), "mfu": round(tps * fpt / peak_flops, 4),
                   "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else 0}
            if batch_hash:
                rec["batch_hash"] = batch_hash
            log(rec)
            t_log, steps_since_log = time.time(), 0
            # stop rules: the last good checkpoint stays on HF, nothing broken is saved
            if not math.isfinite(rec["loss"]) or rec["scale"] < 1:
                log({"type": "event", "step": step, "event": "halt", "reason":
                     "nan_loss" if not math.isfinite(rec["loss"]) else "grad_scaler_below_1"})
                halted = True
                break

        if step % eval_every == 0 or step == total_steps:
            final = step == total_steps and args.final_full_eval
            ids, nbytes = (val_all, bytes_all) if final else (val_subset, bytes_subset)
            vloss = evaluate(model, val, ids, rank, world, device)
            n_tok, n_b = nbytes
            log({"type": "val", "step": step, "tokens": tokens_seen, "val_loss": round(vloss, 6),
                 "val_ppl": round(math.exp(vloss), 3), "val_bpb": round(vloss / math.log(2) * n_tok / n_b, 6),
                 "windows": len(ids)})
            val_history.append(vloss)
            if len(val_history) >= 4 and all(a < b for a, b in zip(val_history[-4:], val_history[-3:])):
                log({"type": "event", "step": step, "event": "halt", "reason": "val_loss_rose_3_evals",
                     "last_vals": val_history[-4:]})
                rising = True
            if vloss < best_val:
                best_val = vloss
                if rank == 0:
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                    best_info = {"step": step, "tokens": tokens_seen, "val_loss": vloss,
                                 "windows": len(ids), "model_config": mcfg.to_dict()}
                    best_uploaded = False

        step_times.append(time.time() - t0)
        over_budget = torch.zeros((), device=device)
        if rank == 0:
            avg = float(np.mean(step_times[-20:]))
            over_budget += float(time.time() - t_start + 2 * avg > budget)
        if distributed:
            dist.all_reduce(over_budget)

        if step % save_every == 0 or step == total_steps or over_budget.item() > 0 or rising:
            save(step)
            saved_at = step
        if rising:
            break
        if args.kill_at_step is not None and step == args.kill_at_step:
            if rank == 0:
                manager.wait()        # the simulated crash comes after the step-80 upload has landed
            barrier()
            print(f"[train] rank {rank}: simulated crash at step {step} (os._exit(1), no save)", flush=True)
            os._exit(1)
        if over_budget.item() > 0:
            log({"type": "event", "step": step, "event": "stop_wall_clock", "elapsed_s": round(time.time() - t_start)})
            break

    if saved_at != step and not halted:
        save(step)
    log({"type": "event", "step": step, "event": "session_end", "done": step >= total_steps, "halted": halted or rising,
         "elapsed_s": round(time.time() - t_start)})
    if rank == 0:
        manager.wait()
        manager.upload_log(log_path)
    barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
