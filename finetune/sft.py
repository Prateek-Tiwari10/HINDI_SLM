"""D7 Stages 7d/7e: supervised fine-tuning loop, Hinglish -> Devanagari.

    torchrun --nproc_per_node=2 -m finetune.sft --run phaseA --phase A --init base
    torchrun --nproc_per_node=2 -m finetune.sft --run bracket_3e-4 --phase B --init run:phaseA --peak-lr 3e-4
    torchrun --nproc_per_node=2 -m finetune.sft --run main --phase B --init run:phaseA --peak-lr <best> \
        --epochs 2 --warmup-frac 0.03 --eval-every 1000
    ... --init random (A1: no pretraining)   ... --init base --phase B (A2: no phase A)

--init: base = pretrained 10,240 model resized with mean init (D9); random = fresh model at the
extended vocab; run:<name> = final weights of an earlier run (d7/<name>/final/model.pt).
Phase A trains only the 64 new embedding rows (freeze_for_phase_a + phase_a_param_groups,
wd 0); phase B trains everything with the pretraining param groups.

Loss: fp32 cross-entropy summed over target positions and divided by the number of targets
in the whole global batch (token-level mean over all ranks and micro-steps). Each rank
scales by world because DDP averages gradients.

HF layout under d7/<run>/ in the checkpoints repo:
    step_XXXXXX/ckpt.pt   resume state (model + optimizer + scheduler + scaler + RNG), latest only
    best/model.pt         weights of the lowest selection dev CER (save_finetuned format, no optimizer)

Dev evaluation (spec 7d, D10): masked greedy decoding with the KV cache; each prompt's budget is
decode_budget(prompt_len, meta.json decode_cap). dev_real records are tagged clean / contaminated
(seen in pretraining). If the contaminated share exceeds eval.clean_selection_above, the eval set is
dev_subset clean pairs + dev_subset contaminated pairs and model selection uses the clean CER;
otherwise one uniform subset and selection uses all of it. Both parts are always logged.
    epoch<k>/model.pt     weights at each epoch boundary before the last
    final/model.pt        weights at the last step; then step_* folders are deleted (spec 1.4)
    logs/<run>.jsonl      the log;  done.json  written last (a rerun then exits at once)
"""

import argparse
import contextlib
import gzip
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import hub
from data.config import load_config
from eval.metrics import transliteration_scores
from finetune.data import BatchPlan, PairArrays, collate, decode_budget, masked_loss_sum
from model.config import ModelConfig
from model.model import HindiLlama
from model.surgery import (build_output_mask, finetune_meta, freeze_for_phase_a, load_base_for_finetune,
                           load_finetuned, phase_a_param_groups, save_finetuned, unfreeze_all)
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
from train.checkpoint import CheckpointManager, rng_state, set_rng_state, step_name
from train.optim import build_scheduler, param_groups
from train.train import Logger

CONFIG = "configs/sft.yaml"
SPLIT_FILES = ("ids.npy", "offsets.npy", "prompt_len.npy")


@dataclass
class Hparams:
    run: str
    phase: str
    init: str
    peak_lr: float
    warmup_steps: int
    total_steps: int
    global_pairs: int
    micro_batch: int
    accum: int
    world: int
    weight_decay: float
    seed: int
    eval_every: int
    save_every: int
    max_train_pairs: int | None
    compile: bool
    stop_step: int | None = None     # train until here; the LR schedule still runs to total_steps


# ---------------------------------------------------------------- data
def fetch_pairs(cfg: dict, splits: tuple[str, ...], revision: str | None) -> Path:
    """Download the arrays (and dev_real/val_syn JSONL) of the given splits; returns the local dir."""
    d = cfg["data"]
    local = Path(d["local_dir"])
    hub.download(d["hf_repo"], f"{d['path']}meta.json", local, revision=revision)
    for split in splits:
        names = [f"{split}.{ext}" for ext in SPLIT_FILES] + ([f"{split}.jsonl.gz"] if split != "train" else [])
        for name in names:
            hub.download(d["hf_repo"], f"{d['path']}{name}", local, revision=revision)
    return local / d["path"]


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dev_subset(arrays: PairArrays, n: int, seed: int, max_tokens: int, pool=None) -> list[int]:
    """Fixed, sorted subset of n dev_real pairs whose built length fits max_tokens (from pool if given)."""
    fits = np.nonzero(arrays.lengths <= max_tokens)[0]
    if pool is not None:
        fits = np.intersect1d(fits, np.asarray(pool, dtype=np.int64))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(fits, size=min(n, len(fits)), replace=False).tolist())


# ---------------------------------------------------------------- model / optimizer
def build_model(init: str, phase: str, tok, cfg: dict, local_dir: Path, seed: int) -> tuple[HindiLlama, str]:
    """(model at the extended vocab, id of the starting weights) for --init base | random | run:<name>."""
    mcfg = ModelConfig.from_yaml(cfg["base"]["model_config"])
    local_dir = Path(cfg["checkpoint"]["local_dir"]) / "_models"     # shared by runs in one session
    base_id = f"{cfg['base']['hf_repo']}/{cfg['base']['checkpoint']}"
    if init == "base":
        path = hub.download(cfg["base"]["hf_repo"], cfg["base"]["checkpoint"], local_dir / "base", repo_type="model")
        model = load_base_for_finetune(path, tok, mcfg, init=cfg["base"]["init"], seed=cfg["base"]["init_seed"])
        start_id = base_id
    elif init == "random":
        torch.manual_seed(seed)
        model = HindiLlama(ModelConfig(**{**mcfg.to_dict(), "vocab_size": tok.vocab_size, "expected_params": None}))
        start_id = "random_init"
    elif init.startswith("run:"):
        remote = f"{cfg['checkpoint']['prefix_root']}{init[4:]}/final/model.pt"
        path = hub.download(cfg["checkpoint"]["hf_repo"], remote, local_dir / "init", repo_type="model")
        model, _ = load_finetuned(path, tok)
        start_id = f"{cfg['checkpoint']['hf_repo']}/{remote}"
    else:
        raise ValueError(f"unknown --init {init!r}")
    model.cfg.attn_impl = "sdpa"
    if phase == "A":
        freeze_for_phase_a(model, tok.base_vocab_size)
    else:
        unfreeze_all(model)
    return model, start_id


def build_optimizer(model: HindiLlama, phase: str, hp: Hparams, ocfg: dict) -> torch.optim.AdamW:
    """Phase A: the embedding alone, wd 0 (D9). Phase B: decay on 2-D weights, none on gains."""
    groups = phase_a_param_groups(model) if phase == "A" else param_groups(model, hp.weight_decay)
    fused = torch.cuda.is_available() and next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=hp.peak_lr, betas=tuple(ocfg["betas"]), eps=ocfg["eps"], fused=fused)


# ---------------------------------------------------------------- one optimizer step
def train_step(train_model, model, opt, sched, scaler, arrays: PairArrays, plan: BatchPlan, step: int, hp: Hparams,
               buckets: list[int], rank: int, device, ddp: DDP | None, grad_clip: float) -> tuple[float, int, float]:
    """One global batch. Returns (sum of target losses on this rank, global target count, grad norm)."""
    gidx = plan.global_indices(step)
    n_targets = int(arrays.n_targets[gidx].sum())
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    for m in range(hp.accum):
        x, y = collate(arrays, plan.micro_indices(step, m, rank, hp.world, hp.micro_batch), buckets, device)
        sync = ddp is None or m == hp.accum - 1
        with (contextlib.nullcontext() if sync else ddp.no_sync()):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = train_model(x)
            part = masked_loss_sum(logits, y)
            scaler.scale(part * (hp.world / n_targets)).backward()
        loss_sum += part.detach().double()
    scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    sched.step()
    return loss_sum, n_targets, float(grad_norm)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_loss(model, arrays: PairArrays, indices, buckets, rank: int, world: int, device, batch: int) -> float:
    """Token-level mean target loss over the given pairs (each rank takes every world-th pair)."""
    was = model.training
    model.eval()
    mine = sorted(list(indices)[rank::world], key=lambda i: arrays.lengths[i])
    tot = torch.zeros(2, device=device, dtype=torch.float64)
    for s in range(0, len(mine), batch):
        x, y = collate(arrays, mine[s:s + batch], buckets, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        tot[0] += masked_loss_sum(logits, y).double()
        tot[1] += (y != -100).sum()
    if world > 1:
        dist.all_reduce(tot)
    model.train(was)
    return (tot[0] / tot[1]).item()


@torch.no_grad()
def greedy_decode(model, prompts: list[list[int]], tok, mask: torch.Tensor, device, max_new: int,
                  cap: dict | None = None) -> list[str]:
    """Masked greedy decoding with the KV cache, one prompt at a time (no padding: exact).

    cap: meta.json decode_cap; each prompt then gets min(max_new, decode_budget(len(prompt), cap)) tokens.
    """
    out = []
    for prompt in prompts:
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        budget = min(max_new, decode_budget(len(prompt), cap)) if cap else max_new
        n_new = min(budget, model.cfg.max_seq_len - len(prompt))
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            y = model.generate(x, n_new, temperature=0, use_cache=True, eos_id=tok.eos_id, logits_mask=mask)
        gen = y[0, len(prompt):].tolist()
        if tok.eos_id in gen:
            gen = gen[: gen.index(tok.eos_id)]
        out.append(tok.decode_dev(gen))
    return out


def eval_dev(model, arrays: PairArrays, refs: list[str], indices: list[int], tok, mask, rank: int, world: int,
             device, max_new: int, cap: dict | None = None) -> tuple[dict, dict[int, str]]:
    """CER / WER / exact match on the dev subset (rank 0 gets the scores and all hypotheses)."""
    mine = indices[rank::world]
    hyps = dict(zip(mine, greedy_decode(model, [arrays.prompt(i) for i in mine], tok, mask, device, max_new, cap)))
    if world > 1:
        parts = [None] * world
        dist.all_gather_object(parts, hyps)
        hyps = {k: v for p in parts for k, v in p.items()}
    scores = transliteration_scores([hyps[i] for i in indices], [refs[i] for i in indices])
    return scores, hyps


def dev_eval_plan(arrays: PairArrays, records: list[dict], ecfg: dict, max_tokens: int) -> dict:
    """The fixed dev eval set, its clean / contaminated parts and the part used for model selection."""
    contaminated = np.array([r.get("contamination", "clean") != "clean" for r in records])
    share = float(contaminated.mean())
    n, seed = ecfg["dev_subset"], ecfg["subset_seed"]
    if share > ecfg["clean_selection_above"]:
        parts = {"clean": dev_subset(arrays, n, seed, max_tokens, np.nonzero(~contaminated)[0]),
                 "contaminated": dev_subset(arrays, n, seed, max_tokens, np.nonzero(contaminated)[0])}
        indices, select = sorted(parts["clean"] + parts["contaminated"]), "clean"
    else:
        indices, select = dev_subset(arrays, n, seed, max_tokens), "all"
        parts = {"clean": [i for i in indices if not contaminated[i]],
                 "contaminated": [i for i in indices if contaminated[i]]}
    return {"indices": indices, "parts": {"all": indices, **parts}, "select": select,
            "contaminated_share": round(share, 4)}


def part_scores(hyps: dict[int, str], refs: list[str], parts: dict[str, list[int]]) -> dict[str, dict]:
    return {name: transliteration_scores([hyps[i] for i in idx], [refs[i] for i in idx])
            for name, idx in parts.items() if idx}


# ---------------------------------------------------------------- resume state
def make_state(model, opt, sched, scaler, step: int, rngs: list, best: dict, meta: dict, hp: Hparams) -> dict:
    return {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "rng": rngs, "best": best, "finetune": meta,
            "model_config": model.cfg.to_dict(), "hparams": asdict(hp)}


def restore_state(state: dict, model, opt, sched, scaler, rank: int) -> tuple[int, dict]:
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    set_rng_state(state["rng"][rank])
    return state["step"], state["best"]


# ---------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--run", required=True, help="HF folder d7/<run>/ and log name")
    p.add_argument("--phase", choices=["A", "B"], required=True)
    p.add_argument("--init", required=True, help="base | random | run:<name>")
    p.add_argument("--peak-lr", type=float)
    p.add_argument("--total-steps", type=int)
    p.add_argument("--epochs", type=float, help="instead of --total-steps")
    p.add_argument("--stop-after-epochs", type=float,
                   help="stop (and finish the run) after this many epochs; the schedule still spans --epochs "
                        "(7e ablations: same schedule as the main run, compared at its epoch-1 checkpoint)")
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--warmup-frac", type=float)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--max-train-pairs", type=int, help="train on the first N pairs only (overfit gate)")
    p.add_argument("--data-revision", help="pin the pairs repo revision (Stage 7c upload sha)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", type=int, choices=[0, 1])
    p.add_argument("--budget-s", type=float)
    p.add_argument("--skip-step0-eval", action="store_true",
                   help="no step-0 evaluation (7e bracket: identical for every LR, already logged by phaseA)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    t_start = float(os.environ.get("RUN_STARTED_AT", time.time()))
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=1))
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    def barrier():
        if distributed:
            dist.barrier()

    ck = cfg["checkpoint"]
    prefix = f"{ck['prefix_root']}{args.run}/"
    local = Path(ck["local_dir"]) / args.run
    log_path = local / f"{args.run}.jsonl"
    manager = CheckpointManager(ck["hf_repo"], prefix, local, keep_last=1, log_name=log_path.name)
    if rank == 0:
        local.mkdir(parents=True, exist_ok=True)
        done = f"{prefix}done.json" in hub.list_files(ck["hf_repo"], prefix, repo_type="model")
    else:
        done = None
    if distributed:
        obj = [done]
        dist.broadcast_object_list(obj, src=0)
        done = obj[0]
    if done:
        if rank == 0:
            print(f"[sft] {prefix} already finished (done.json); nothing to do", flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    # ---------- data ----------
    revision = args.data_revision or cfg["data"]["revision"]
    if rank == 0:
        fetch_pairs(cfg, ("train", "val_syn", "dev_real"), revision)
    barrier()
    pair_dir = fetch_pairs(cfg, ("train", "val_syn", "dev_real"), revision)
    train = PairArrays(pair_dir, "train")
    val, dev = PairArrays(pair_dir, "val_syn"), PairArrays(pair_dir, "dev_real")
    dev_records = read_jsonl_gz(pair_dir / "dev_real.jsonl.gz")
    dev_refs = [r["dev"] for r in dev_records]
    decode_cap = json.loads((pair_dir / "meta.json").read_text())["decode_cap"]
    buckets = cfg["data"]["buckets"]
    lengths = train.lengths
    if args.max_train_pairs:
        lengths = lengths[: args.max_train_pairs]

    # ---------- hyperparameters ----------
    phase_cfg = cfg["phases"][args.phase]
    micro = args.micro_batch or cfg["batch"]["micro_batch"]
    global_pairs = min(cfg["batch"]["global_pairs"], len(lengths))
    if global_pairs % (micro * world):
        micro = global_pairs // world
    accum = global_pairs // (micro * world)
    plan = BatchPlan(lengths, global_pairs, cfg["data"]["chunk_batches"], cfg["data"]["order_seed"])
    total = (round(args.epochs * plan.steps_per_epoch) if args.epochs else
             args.total_steps or phase_cfg["total_steps"])
    warmup = args.warmup_steps if args.warmup_steps is not None else (
        round(args.warmup_frac * total) if args.warmup_frac else phase_cfg["warmup_steps"])
    peak_lr = args.peak_lr or phase_cfg["peak_lr"]
    assert peak_lr, "phase B needs --peak-lr (chosen by the Stage 7e bracket)"
    hp = Hparams(run=args.run, phase=args.phase, init=args.init, peak_lr=peak_lr, warmup_steps=warmup,
                 total_steps=total, global_pairs=global_pairs, micro_batch=micro, accum=accum, world=world,
                 weight_decay=0.0 if args.phase == "A" else cfg["optim"]["weight_decay"], seed=args.seed,
                 eval_every=args.eval_every or cfg["eval"]["every_steps"],
                 save_every=args.save_every or ck["every_steps"], max_train_pairs=args.max_train_pairs,
                 compile=bool(cfg["compile"] if args.compile is None else args.compile),
                 stop_step=min(total, round(args.stop_after_epochs * plan.steps_per_epoch))
                 if args.stop_after_epochs else total)

    # ---------- model ----------
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    mask = build_output_mask(tok).to(device)
    latest, ckpt_path, remote_log = None, None, None
    if rank == 0:
        latest = manager.latest()
        if latest:
            ckpt_path, remote_log = manager.download(latest)
    if distributed:
        obj = [latest, str(ckpt_path) if ckpt_path else None]
        dist.broadcast_object_list(obj, src=0)
        latest, ckpt_path = obj[0], Path(obj[1]) if obj[1] else None
    if latest:                                              # resume: architecture only, weights from state
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = HindiLlama(ModelConfig(**state["model_config"]))
        start_id = state["finetune"]["base_checkpoint"]
        if args.phase == "A":
            freeze_for_phase_a(model, tok.base_vocab_size)
    else:
        if rank == 0:
            build_model(args.init, args.phase, tok, cfg, local, args.seed)   # downloads once
        barrier()
        model, start_id = build_model(args.init, args.phase, tok, cfg, local, args.seed)
    model.to(device)
    torch.manual_seed(args.seed + rank)
    opt = build_optimizer(model, args.phase, hp, cfg["optim"])
    sched = build_scheduler(opt, hp.warmup_steps, hp.total_steps, cfg["optim"]["min_lr_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    meta = {**finetune_meta(tok, start_id, args.phase), "init": args.init,
            "data_repo": cfg["data"]["hf_repo"], "data_path": cfg["data"]["path"], "data_revision": revision,
            "hparams": asdict(hp)}
    log = Logger(log_path, rank)
    step, best = 0, {"dev_cer": None, "step": None}
    if latest:
        step, best = restore_state(state, model, opt, sched, scaler, rank)
        del state
        if rank == 0 and remote_log is not None:
            log_path.write_bytes(Path(remote_log).read_bytes())
            log.truncate_after(step)
    elif rank == 0 and log_path.exists():
        log_path.unlink()
    barrier()
    log({"type": "event", "step": step, "event": "resume" if latest else "fresh_start", "from": latest or start_id,
         **asdict(hp), "steps_per_epoch": plan.steps_per_epoch, "train_pairs": len(lengths),
         "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
         "torch": torch.__version__})

    ddp = DDP(model, device_ids=[local_rank]) if distributed else None
    train_model = ddp or model
    if hp.compile:
        train_model = torch.compile(train_model, dynamic=False)

    dev_plan = dev_eval_plan(dev, dev_records, cfg["eval"], buckets[-1])
    sub = dev_plan["indices"]
    log({"type": "event", "step": step, "event": "dev_eval_plan", "select": dev_plan["select"],
         "contaminated_share": dev_plan["contaminated_share"], "decode_cap": decode_cap,
         **{f"n_{k}": len(v) for k, v in dev_plan["parts"].items()}})
    val_ids = list(range(len(val)))

    def export(folder: str, weights: dict | None, info: dict) -> None:
        """Rank 0: weights-only save_finetuned file + info.json to d7/<run>/<folder>/."""
        if rank != 0:
            return
        out = local / folder
        out.mkdir(parents=True, exist_ok=True)
        snapshot = HindiLlama(model.cfg)
        snapshot.load_state_dict(weights if weights is not None else
                                 {k: v.detach().cpu() for k, v in model.state_dict().items()})
        save_finetuned(out / "model.pt", snapshot, {**meta, "best_dev_cer": best["dev_cer"]}, info=info)
        (out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
        hub.upload_folder(ck["hf_repo"], out, f"{prefix}{folder}", repo_type="model", message=f"{prefix}{folder}")

    def evaluate(at: int) -> dict:
        vloss = eval_loss(model, val, val_ids, buckets, rank, world, device, cfg["eval"]["val_batch"])
        _, hyps = eval_dev(model, dev, dev_refs, sub, tok, mask, rank, world, device, cfg["eval"]["max_new_tokens"],
                           decode_cap)
        parts = part_scores(hyps, dev_refs, dev_plan["parts"])
        scores = parts[dev_plan["select"]]
        rec = {"type": "val", "step": at, "val_syn_loss": round(vloss, 6), "select": dev_plan["select"],
               "dev_cer": round(scores["cer"], 6), "dev_wer": round(scores["wer"], 6),
               "dev_em": round(scores["exact_match"], 6), "dev_n": scores["n"],
               **{f"dev_{m}_{name}": round(sc[k], 6) for name, sc in parts.items()
                  for m, k in (("cer", "cer"), ("wer", "wer"), ("em", "exact_match"))},
               "samples": [{"i": i, "hyp": hyps[i], "ref": dev_refs[i]} for i in sub[:3]]}
        log(rec)
        if best["dev_cer"] is None or scores["cer"] < best["dev_cer"]:
            best.update(dev_cer=scores["cer"], step=at, val_syn_loss=vloss, dev_wer=scores["wer"],
                        dev_em=scores["exact_match"], select=dev_plan["select"],
                        dev_parts={name: {k: round(v, 6) for k, v in sc.items()} for name, sc in parts.items()})
            export("best", None, {"step": at, **best})      # uploaded now: survives a session restart
        return rec

    def save(at: int) -> None:
        rngs = [None] * world
        if distributed:
            dist.all_gather_object(rngs, rng_state())
        else:
            rngs = [rng_state()]
        if rank == 0:
            path = manager.save_local(at, make_state(model, opt, sched, scaler, at, rngs, best, meta, hp))
            log({"type": "event", "step": at, "event": "checkpoint", "name": step_name(at),
                 "elapsed_s": round(time.time() - t_start)})
            manager.upload_async(at, path, log_path)
        barrier()

    if step == 0 and not args.skip_step0_eval:
        evaluate(0)                                          # step-0 baseline
    model.train()
    t_log, since = time.time(), 0
    step_times = []
    saved_at = step
    while step < hp.stop_step:
        t0 = time.time()
        lr = sched.get_last_lr()[0]
        loss_sum, n_targets, gnorm = train_step(train_model, model, opt, sched, scaler, train, plan, step, hp,
                                                buckets, rank, device, ddp, cfg["optim"]["grad_clip"])
        step += 1
        since += 1
        if step % cfg["logging"]["every_steps"] == 0 or step == hp.total_steps:
            if distributed:
                dist.all_reduce(loss_sum)
            loss = loss_sum.item() / n_targets
            log({"type": "train", "step": step, "loss": round(loss, 6), "lr": lr, "grad_norm": round(gnorm, 4),
                 "scale": scaler.get_scale(), "steps_per_s": round(since / (time.time() - t_log), 3),
                 "mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else 0})
            t_log, since = time.time(), 0
            if not math.isfinite(loss):
                log({"type": "event", "step": step, "event": "halt", "reason": "nan_loss"})
                raise SystemExit("non-finite loss; the last good checkpoint stays on HF")
        epoch_end = step % plan.steps_per_epoch == 0 and step < hp.total_steps
        if step % hp.eval_every == 0 or step in (hp.stop_step, hp.total_steps) or epoch_end:
            evaluate(step)
        if epoch_end:
            export(f"epoch{step // plan.steps_per_epoch}", None, {"step": step, **best, "epoch_end": True})
            barrier()
        step_times.append(time.time() - t0)
        over = torch.zeros((), device=device)
        if rank == 0:
            budget = args.budget_s or ck["wall_clock_budget_s"]
            over += float(time.time() - t_start + 2 * float(np.mean(step_times[-20:])) > budget)
        if distributed:
            dist.all_reduce(over)
        if step % hp.save_every == 0 or over.item() > 0:
            save(step)
            saved_at = step
        if over.item() > 0 and step < hp.stop_step:
            log({"type": "event", "step": step, "event": "stop_wall_clock", "elapsed_s": round(time.time() - t_start)})
            break

    if step >= hp.stop_step:
        if saved_at != step:
            save(step)
        if rank == 0:
            manager.wait()
            final_info = {"step": step, "best": best, "hparams": asdict(hp)}
            export("final", None, final_info)
            manager.keep_last = 0
            manager.prune()                                  # resume state no longer needed (spec 1.4)
            manager.upload_log(log_path)
            hub.upload_file(ck["hf_repo"], _write(local / "done.json", final_info), f"{prefix}done.json",
                            repo_type="model", message=f"{prefix} done")
        log({"type": "event", "step": step, "event": "run_done", "best": best, "elapsed_s": round(time.time() - t_start)})
    elif rank == 0:
        manager.wait()
    if rank == 0 and log_path.exists():
        out = Path(ck["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / log_path.name).write_bytes(log_path.read_bytes())
    barrier()
    if distributed:
        dist.destroy_process_group()


def _write(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    return path


if __name__ == "__main__":
    main()
