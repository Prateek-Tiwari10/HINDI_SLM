"""Stage 7d gates on the real 61M model (one GPU). Writes OUT_DIR/stage7d.json.

    python -m finetune.stage7d

G4 overfit: 32 train pairs, full fine-tune from the resized base, every step sees all 32;
    pass = train loss < 0.05 and 32/32 exact match (masked greedy decoding) within 300 steps.
G5 phase-A freeze: 50 phase-A steps on real batches; base embedding rows and every
    non-embedding tensor bit-identical, the 64 new rows changed.
G6 eval sanity: step-0 val_syn loss and dev_real CER/WER/EM on the fixed 500-pair subset.
Compile probe: eager vs torch.compile steps/s and compiled-graph count on 40 phase-B steps.
The training code is finetune.sft's (build_model, build_optimizer, train_step, eval_*).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data.config import load_config
from finetune.data import BatchPlan, PairArrays, collate, masked_loss_sum
from finetune.sft import (CONFIG, Hparams, build_model, build_optimizer, dev_subset, eval_dev, eval_loss,
                          fetch_pairs, greedy_decode, read_jsonl_gz, train_step)
from model.surgery import build_output_mask
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
from train.optim import build_scheduler

OUT_DIR = Path(os.environ.get("STAGE7D_OUT", "/kaggle/working/out/stage7d"))
STAGE7C_JSON = Path("/kaggle/working/out/stage7c/stage7c.json")


def log(msg):
    print(f"[stage7d {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hparams(phase, lr, warmup, total, pairs, micro, compile_=False):
    return Hparams(run="gates", phase=phase, init="base", peak_lr=lr, warmup_steps=warmup, total_steps=total,
                   global_pairs=pairs, micro_batch=micro, accum=pairs // micro, world=1,
                   weight_decay=0.0 if phase == "A" else 0.1, seed=1337, eval_every=10**9, save_every=10**9,
                   max_train_pairs=None, compile=compile_)


def setup(phase, hp, cfg, tok, local, device):
    model, start_id = build_model("base", phase, tok, cfg, local, hp.seed)
    model.to(device)
    opt = build_optimizer(model, phase, hp, cfg["optim"])
    sched = build_scheduler(opt, hp.warmup_steps, hp.total_steps, cfg["optim"]["min_lr_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    return model, opt, sched, scaler, start_id


def g4_overfit(cfg, tok, mask, train, local, device):
    n, max_steps = 32, 300
    hp = hparams("B", 3e-4, 20, max_steps, n, 32)
    model, opt, sched, scaler, _ = setup("B", hp, cfg, tok, local, device)
    plan = BatchPlan(train.lengths[:n], n, 1, 1337)
    refs = [tok.decode_dev(train.example(i)[0][train.prompt_len[i]:-1].tolist()) for i in range(n)]
    prompts = [train.prompt(i) for i in range(n)]
    curve, passed_at, em = [], None, 0
    for step in range(max_steps):
        loss_sum, n_t, _ = train_step(model, model, opt, sched, scaler, train, plan, step, hp,
                                      cfg["data"]["buckets"], 0, device, None, cfg["optim"]["grad_clip"])
        loss = loss_sum.item() / n_t
        curve.append(round(loss, 5))
        if (step + 1) % 25 == 0 or loss < 0.05:
            hyps = greedy_decode(model, prompts, tok, mask, device, cfg["eval"]["max_new_tokens"])
            em = sum(h == r for h, r in zip(hyps, refs))
            model.train()
            log(f"G4 step {step + 1}: loss {loss:.4f}, exact {em}/{n}")
            if loss < 0.05 and em == n:
                passed_at = step + 1
                break
    return {"pass": passed_at is not None, "steps": passed_at or max_steps, "final_loss": curve[-1],
            "exact_match": f"{em}/{n}", "loss_curve_every_10": curve[::10], "lr": hp.peak_lr}


def g5_phase_a(cfg, tok, train, local, device):
    hp = hparams("A", cfg["phases"]["A"]["peak_lr"], 5, 50, 64, 32)
    model, opt, sched, scaler, _ = setup("A", hp, cfg, tok, local, device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    plan = BatchPlan(train.lengths, 64, cfg["data"]["chunk_batches"], cfg["data"]["order_seed"])
    for step in range(50):
        train_step(model, model, opt, sched, scaler, train, plan, step, hp, cfg["data"]["buckets"], 0, device,
                   None, cfg["optim"]["grad_clip"])
    base = tok.base_vocab_size
    changed_frozen = []
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        if k in ("tok_emb.weight", "lm_head.weight"):
            if not torch.equal(v[:base], before[k][:base]):
                changed_frozen.append(f"{k}[:base]")
        elif not torch.equal(v, before[k]):
            changed_frozen.append(k)
    w = model.tok_emb.weight.detach().cpu()
    new_delta = (w[base:] - before["tok_emb.weight"][base:]).norm(dim=1)
    return {"pass": not changed_frozen and bool((new_delta[:50] > 0).all()), "changed_frozen": changed_frozen,
            "new_rows_changed": int((new_delta > 0).sum()), "new_rows_used_by_data": 50,
            "new_row_delta_norm_mean": round(float(new_delta[:50].mean()), 6)}


def g6_eval(cfg, tok, mask, val, dev, dev_refs, local, device):
    model, _ = build_model("base", "B", tok, cfg, local, 1337)
    model.to(device)
    t0 = time.time()
    vloss = eval_loss(model, val, list(range(len(val))), cfg["data"]["buckets"], 0, 1, device, cfg["eval"]["val_batch"])
    t1 = time.time()
    sub = dev_subset(dev, cfg["eval"]["dev_subset"], cfg["eval"]["subset_seed"], cfg["data"]["buckets"][-1])
    scores, hyps = eval_dev(model, dev, dev_refs, sub, tok, mask, 0, 1, device, cfg["eval"]["max_new_tokens"])
    t2 = time.time()
    return {"pass": np.isfinite(vloss) and scores["n"] == len(sub), "val_syn_loss": round(vloss, 5),
            "dev_cer": round(scores["cer"], 5), "dev_wer": round(scores["wer"], 5),
            "dev_em": round(scores["exact_match"], 5), "dev_n": scores["n"],
            "seconds": {"val_syn_loss": round(t1 - t0, 1), "dev_decode": round(t2 - t1, 1)},
            "samples": [{"rom_prompt_len": len(dev.prompt(i)), "hyp": hyps[i], "ref": dev_refs[i]} for i in sub[:5]]}


def compile_probe(cfg, tok, train, local, device):
    out = {}
    for flag in (False, True):
        hp = hparams("B", 1e-4, 10, 40, 64, 32, compile_=flag)
        model, opt, sched, scaler, _ = setup("B", hp, cfg, tok, local, device)
        tm = torch.compile(model, dynamic=False) if flag else model
        plan = BatchPlan(train.lengths, 64, cfg["data"]["chunk_batches"], cfg["data"]["order_seed"])
        torch._dynamo.reset()
        times, shapes = [], set()
        for step in range(40):
            for m in range(hp.accum):
                x, _ = collate(train, plan.micro_indices(step, m, 0, 1, hp.micro_batch), cfg["data"]["buckets"])
                shapes.add(tuple(x.shape))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            train_step(tm, model, opt, sched, scaler, train, plan, step, hp, cfg["data"]["buckets"], 0, device,
                       None, cfg["optim"]["grad_clip"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        from torch._dynamo.utils import counters
        out["compiled" if flag else "eager"] = {
            "s_per_step_last20": round(float(np.mean(times[20:])), 4), "first_step_s": round(times[0], 2),
            "shapes_seen": sorted(shapes), "unique_graphs": int(counters["stats"]["unique_graphs"]) if flag else 0}
        del model, opt, tm
        torch.cuda.empty_cache()
    return out


def main():
    cfg = load_config(CONFIG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    revision = json.loads(STAGE7C_JSON.read_text())["upload"]["revision"] if STAGE7C_JSON.exists() else None
    local = Path(cfg["checkpoint"]["local_dir"]) / "gates"
    local.mkdir(parents=True, exist_ok=True)
    pair_dir = fetch_pairs(cfg, ("train", "val_syn", "dev_real"), revision)
    train, val, dev = (PairArrays(pair_dir, s) for s in ("train", "val_syn", "dev_real"))
    dev_refs = [r["dev"] for r in read_jsonl_gz(pair_dir / "dev_real.jsonl.gz")]
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    mask = build_output_mask(tok).to(device)
    res = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "torch": torch.__version__, "data_revision": revision, "train_pairs": len(train)}
    path = OUT_DIR / "stage7d.json"
    for name, fn in (("G6_eval_sanity", lambda: g6_eval(cfg, tok, mask, val, dev, dev_refs, local, device)),
                     ("G5_phase_a_freeze", lambda: g5_phase_a(cfg, tok, train, local, device)),
                     ("G4_overfit", lambda: g4_overfit(cfg, tok, mask, train, local, device)),
                     ("compile_probe", lambda: compile_probe(cfg, tok, train, local, device))):
        t0 = time.time()
        log(f"{name} ...")
        try:
            res[name] = fn()
        except Exception as err:  # noqa: BLE001 - record and continue with the other gates
            res[name] = {"pass": False, "error": f"{type(err).__name__}: {err}"}
        res[name]["wall_s"] = round(time.time() - t0, 1)
        log(f"{name}: {json.dumps({k: v for k, v in res[name].items() if k not in ('samples', 'loss_curve_every_10')}, ensure_ascii=False)}")
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()
    res["all_gates_pass"] = all(res[g].get("pass") for g in ("G4_overfit", "G5_phase_a_freeze", "G6_eval_sanity"))
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    log(f"all gates pass: {res['all_gates_pass']}")


if __name__ == "__main__":
    main()
