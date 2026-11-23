"""Stage 7e in one Kaggle kernel (hindi-lm-d7e-train, 2x T4), with the choices made in code.

    python -m finetune.stage7e                 # the whole pipeline; a re-run continues where it stopped
    python -m finetune.stage7e forgetting --model base10240|resized|<run>/<folder> --tag <tag>
    python -m finetune.stage7e g5check
    python -m finetune.stage7e samples
    python -m finetune.stage7e report

Every HF run needs a manual Save Version (API pushes cannot read HF_TOKEN), so the whole of
Stage 7e is chained into one kernel to keep that to one click per ~11 h session. Each step is
skipped when its result is already on HF; each finetune.sft run resumes from its own checkpoint.

 0. Data: romanize-v2.1 pairs (python -m data.hinglish.stage7c all) unless <path>meta.json is
    already in the pairs repo. The pairs revision is pinned in pipeline.json at first use.
 1. Forgetting baseline: Hindi LM loss / BPB on 200 v2 val windows, base model (10,240) and resized.
 2. Phase A (run phaseA: 64 new rows, LR 1e-3, 1,000 steps), then the G5 property on its final weights.
 3. LR bracket 1e-4 / 3e-4 / 6e-4: 1,500 steps each from phaseA, warmup 100. If the best is at an
    edge, one more point beyond it (3e-5 or 1e-3). Pick: lowest selection dev CER at step 1,500.
    The bracket weights are then deleted from HF (logs and done.json stay).
 4. Main run: best LR, 2 epochs, warmup 3%, eval every 1,000; best/, epoch1/, final/.
 5. Forgetting after main (best/ and final/), 30 dev_real examples from main best/, report.
 6. Ablations, 1 epoch each on the main run's 2-epoch schedule (compared with main at epoch 1):
    A2 = phase B straight from the resized base; A1 = random init at the main LR and at 1e-3.
 7. Final report.
After each step OUT is uploaded to d7/stage7e_results/ in the checkpoints repo, so progress can
be read from the laptop while the kernel runs.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data import hub
from data.config import load_config
from finetune.sft import (CONFIG, build_model, dev_eval_plan, fetch_pairs, greedy_decode, read_jsonl_gz)
from finetune.data import PairArrays
from eval.metrics import canonical_dev, transliteration_scores

OUT = Path(os.environ.get("STAGE7E_OUT", "/kaggle/working/out/d7e"))
DL = Path("/kaggle/tmp/d7e_dl")
BRACKET = [1e-4, 3e-4, 6e-4]
EDGE_EXTRA = {1e-4: 3e-5, 6e-4: 1e-3}
A1_ALT_LR = 1e-3                     # the pretraining peak (spec 7e.4)
FORGET_WINDOWS = 200
STAGE7C_OUT = Path("/kaggle/working/out/stage7c")


def log(msg: str) -> None:
    print(f"[d7e {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr_name(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")          # 1e-4, 3e-5, 1e-3


cfg = load_config(CONFIG)
REPO = cfg["checkpoint"]["hf_repo"]
ROOT = cfg["checkpoint"]["prefix_root"]               # d7/
RESULTS = f"{ROOT}stage7e_results"
SFT_OUT = Path(cfg["checkpoint"]["out_dir"])


# ---------------------------------------------------------------- HF state
def remote_files(prefix: str) -> list[str]:
    return hub.list_files(REPO, prefix, repo_type="model")


def remote_json(path: str) -> dict | None:
    if path not in remote_files(path.rsplit("/", 1)[0] + "/"):
        return None
    return json.loads(Path(hub.download(REPO, path, DL / "json", repo_type="model")).read_text())


def run_done(run: str) -> dict | None:
    return remote_json(f"{ROOT}{run}/done.json")


def run_log(run: str) -> list[dict]:
    path = f"{ROOT}{run}/logs/{run}.jsonl"
    if path not in remote_files(f"{ROOT}{run}/logs/"):
        return []
    text = Path(hub.download(REPO, path, DL / "logs", repo_type="model")).read_text()
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def publish(what: str) -> None:
    for f in SFT_OUT.glob("*.jsonl") if SFT_OUT.exists() else []:
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        (OUT / "logs" / f.name).write_bytes(f.read_bytes())
    if OUT.exists():
        hub.upload_folder(REPO, OUT, RESULTS, repo_type="model", message=f"d7e results after {what}")


class State:
    """pipeline.json: pinned data revision, bracket results, picked LR, wall time per step."""

    def __init__(self) -> None:
        self.path = OUT / "pipeline.json"
        if self.path.exists():                            # subprocesses: the parent's latest state
            self.d = json.loads(self.path.read_text())
        else:                                             # a new kernel session: what was published
            self.d = remote_json(f"{RESULTS}/pipeline.json") or {"wall_s": {}}

    def __getitem__(self, k):
        return self.d.get(k)

    def set(self, **kw) -> None:
        self.d.update(kw)
        self.path.write_text(json.dumps(self.d, indent=1, ensure_ascii=False) + "\n")

    def add_wall(self, name: str, seconds: float) -> None:
        self.d["wall_s"][name] = round(self.d["wall_s"].get(name, 0) + seconds)
        self.set()


def sh(args: list[str]) -> int:
    log("$ " + " ".join(args))
    return subprocess.run(args).returncode


def sub(state: State, name: str, *args: str) -> None:
    """python -m finetune.stage7e <args> in a fresh process (frees the GPU afterwards)."""
    t0 = time.time()
    rc = sh([sys.executable, "-m", "finetune.stage7e", *args])
    state.add_wall(name, time.time() - t0)
    if rc != 0:
        publish(name)
        sys.exit(f"[d7e] {name} failed with exit code {rc}")


def sft(state: State, run: str, *args: str) -> dict:
    """One finetune.sft run over all GPUs; returns its done.json (exits if it did not finish)."""
    d = run_done(run)
    if d:
        log(f"{run}: already done (best dev CER {d['best']['dev_cer']:.4f} at step {d['best']['step']})")
        return d
    ngpu = max(1, torch.cuda.device_count())
    t0 = time.time()
    rc = sh([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}",
             "-m", "finetune.sft", "--run", run, "--data-revision", state["data_revision"], *args])
    state.add_wall(run, time.time() - t0)
    state.d.setdefault("ngpu", ngpu)
    state.set()
    publish(run)
    d = run_done(run)
    if d is None:
        sys.exit(f"[d7e] {run} stopped before the end (exit code {rc}); re-run the kernel to resume it")
    squash_history(run)
    return d


def squash_history(after: str) -> None:
    """Squash the checkpoints repo's git history (Prateek, 2026-11-22).

    HF counts every LFS version in history toward the 100 GB private limit, and each run leaves
    ~1-3 GB of pruned resume checkpoints and overwritten best/ weights there (the main run hit the
    limit). Current files are unchanged; only old versions of deleted/overwritten files (LFS objects no current
    file refers to) are dropped.
    """
    try:
        api = hub.get_api()
        api.super_squash_history(repo_id=REPO, repo_type="model",
                                 commit_message=f"Squash history after {after} (free storage)")
        # squashing alone frees nothing: the LFS objects stay stored until deleted explicitly
        live = {s.lfs.sha256 for s in api.model_info(REPO, files_metadata=True).siblings if s.lfs}
        orphan = [f for f in api.list_lfs_files(REPO, repo_type="model") if f.file_oid not in live]
        if orphan:
            api.permanently_delete_lfs_files(REPO, orphan, repo_type="model")
        log(f"squashed {REPO} history after {after}; deleted {len(orphan)} orphaned LFS objects "
            f"({sum(f.size for f in orphan) / 1e9:.2f} GB)")
    except Exception as e:  # noqa: BLE001 - storage housekeeping must not stop the pipeline
        log(f"squash after {after} failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- step 0: data
def ensure_data(state: State) -> None:
    from data.hinglish.romanize import GENERATOR_VERSION
    want = cfg["data"]["path"].rstrip("/")
    if want != GENERATOR_VERSION:
        sys.exit(f"[d7e] configs/sft.yaml data.path {want!r} != romanize.GENERATOR_VERSION {GENERATOR_VERSION!r}")
    if state["data_revision"]:
        log(f"data: {want} pinned at {state['data_revision'][:10]}")
        return
    data_repo = cfg["data"]["hf_repo"]
    present = f"{want}/meta.json" in hub.list_files(data_repo, f"{want}/", repo_type="dataset")
    if not present:
        t0 = time.time()
        rc = sh([sys.executable, "-m", "data.hinglish.stage7c", "all"])
        state.add_wall("stage7c_data", time.time() - t0)
        if (STAGE7C_OUT / "stage7c.json").exists():
            (OUT / "stage7c.json").write_bytes((STAGE7C_OUT / "stage7c.json").read_bytes())
            for png in STAGE7C_OUT.glob("*.png"):
                (OUT / png.name).write_bytes(png.read_bytes())
        if rc != 0:
            publish("stage7c")
            sys.exit(f"[d7e] Stage 7c ({want}) failed its gate or crashed (exit {rc}); see stage7c.json")
        report = json.loads((OUT / "stage7c.json").read_text())
        rev = report["upload"]["revision"]
    else:
        rev = hub.get_api().dataset_info(data_repo).sha
    state.set(data_revision=rev, data_path=want)
    publish("data")
    log(f"data: {want} at {rev[:10]}")


# ---------------------------------------------------------------- subcommands (own process)
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_model(spec: str, tok):
    """base10240 (pretrained, original vocab) | resized (mean init, D9) | <run>/<folder> (fine-tuned)."""
    from model.config import ModelConfig
    from model.model import HindiLlama
    from model.surgery import load_finetuned
    if spec == "base10240":
        path = hub.download(cfg["base"]["hf_repo"], cfg["base"]["checkpoint"], DL / "base", repo_type="model")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = HindiLlama(ModelConfig.from_yaml(cfg["base"]["model_config"]))
        model.load_state_dict(state.get("model", state), strict=True)
    elif spec == "resized":
        model, _ = build_model("base", "B", tok, cfg, DL, 0)
    else:
        path = hub.download(REPO, f"{ROOT}{spec}/model.pt", DL / spec.replace("/", "_"), repo_type="model")
        model, _ = load_finetuned(path, tok)
    model.cfg.attn_impl = "sdpa"
    return model.to(device()).eval()


@torch.no_grad()
def cmd_forgetting(spec: str, tag: str) -> None:
    """Hindi LM loss / BPB on FORGET_WINDOWS evenly spaced v2 val windows (plain Devanagari, no <rom>).

    loss_full: softmax over the model's whole vocab (the model as it is);
    loss_base_vocab: softmax over the 10,240 base IDs only (ignores mass on the new Latin IDs).
    """
    import sentencepiece as spm
    import torch.nn.functional as F
    from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
    from train.data import ValWindows, token_byte_lengths
    pcfg = yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())
    d = pcfg["data"]
    meta = json.loads(Path(hub.download(d["hf_repo"], f"{d['prefix']}meta.json", DL / "v2")).read_text())
    (name, _), = meta["splits"]["val"]["files"]
    windows = ValWindows(Path(hub.download(d["hf_repo"], f"{d['prefix']}{name}", DL / "v2")), d["seq_len"])
    ids = sorted(set(np.linspace(0, windows.n_windows - 1, FORGET_WINDOWS).round().astype(int).tolist()))
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    byte_len = token_byte_lengths(spm.SentencePieceProcessor(model_file=str(tok.base_model)))
    model = load_eval_model(spec, tok)
    base_v = tok.base_vocab_size
    tot = {"full": 0.0, "base": 0.0, "n": 0, "bytes": 0}
    for s in range(0, len(ids), 8):
        x, y = windows.batch(ids[s:s + 8], device())
        with torch.autocast("cuda", dtype=torch.float16, enabled=device().type == "cuda"):
            logits, _ = model(x)
        logits = logits.float()
        tot["full"] += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        tot["base"] += F.cross_entropy(logits[..., :base_v].reshape(-1, base_v), y.reshape(-1), reduction="sum").item()
        tot["n"] += y.numel()
        tot["bytes"] += int(byte_len[y.cpu().numpy()].sum())
    res = {"model": spec, "windows": len(ids), "seq_len": d["seq_len"], "target_tokens": tot["n"],
           "loss_full": round(tot["full"] / tot["n"], 6), "loss_base_vocab": round(tot["base"] / tot["n"], 6),
           "bpb_full": round(tot["full"] / math.log(2) / tot["bytes"], 6),
           "ppl_full": round(math.exp(tot["full"] / tot["n"]), 3)}
    log(f"forgetting {tag}: {res}")
    (OUT / f"forgetting_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")


def cmd_g5check() -> None:
    """G5 property after the whole of Phase A: phaseA/final vs the resized base it started from."""
    from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    before = load_eval_model("resized", tok).state_dict()
    after = load_eval_model("phaseA/final", tok).state_dict()
    base = tok.base_vocab_size
    changed_frozen = []
    for k, v in after.items():
        if k in ("tok_emb.weight", "lm_head.weight"):
            if not torch.equal(v[:base], before[k][:base]):
                changed_frozen.append(f"{k}[:base]")
        elif not torch.equal(v, before[k]):
            changed_frozen.append(k)
    delta = (after["tok_emb.weight"][base:] - before["tok_emb.weight"][base:]).float().norm(dim=1)
    res = {"pass": not changed_frozen, "changed_frozen": changed_frozen,
           "new_rows_changed": int((delta > 0).sum()), "new_rows": int(delta.numel()),
           "new_row_delta_norm_mean": round(float(delta.mean()), 6)}
    log(f"G5 after phase A: {res}")
    (OUT / "g5_after_phaseA.json").write_text(json.dumps(res, indent=1) + "\n")
    if not res["pass"]:
        sys.exit("[d7e] phase A changed frozen weights")


def cmd_samples(state: State) -> None:
    """30 dev_real examples from main best/ (at least 5 failures), from the fixed eval subset."""
    from model.surgery import build_output_mask
    from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    pair_dir = fetch_pairs(cfg, ("dev_real",), state["data_revision"])
    dev = PairArrays(pair_dir, "dev_real")
    records = read_jsonl_gz(pair_dir / "dev_real.jsonl.gz")
    cap = json.loads((pair_dir / "meta.json").read_text())["decode_cap"]
    plan = dev_eval_plan(dev, records, cfg["eval"], cfg["data"]["buckets"][-1])
    rng = np.random.default_rng(7)
    pick = rng.permutation(plan["indices"])[:120].tolist()
    model = load_eval_model("main/best", tok)
    mask = build_output_mask(tok).to(device())
    hyps = greedy_decode(model, [dev.prompt(i) for i in pick], tok, mask, device(), cfg["eval"]["max_new_tokens"], cap)
    rows = []
    for i, h in zip(pick, hyps):
        r = records[i]
        sc = transliteration_scores([h], [r["dev"]])
        rows.append({"i": i, "contamination": r.get("contamination", "clean"), "rom": r["rom"], "ref": r["dev"],
                     "hyp": h, "exact": canonical_dev(h) == canonical_dev(r["dev"]), "cer": round(sc["cer"], 4)})
    fails = [x for x in rows if not x["exact"]]
    first = rows[:30]
    first_fails = [x for x in first if not x["exact"]]
    extra = [x for x in rows[30:] if not x["exact"]][: max(0, 5 - len(first_fails))]
    chosen = [x for x in first if x["exact"]][: 30 - len(first_fails) - len(extra)] + first_fails + extra
    out = {"model": "main/best", "n_decoded": len(rows), "exact_rate_of_decoded": round(1 - len(fails) / len(rows), 4),
           "examples": sorted(chosen, key=lambda x: x["i"])}
    (OUT / "samples_dev_real.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    log(f"samples: {len(chosen)} examples, {sum(not x['exact'] for x in chosen)} failures")


def _val(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("type") == "val"]


def cmd_report(state: State) -> None:
    """stage7e.json + plots from the run logs and done.json files on HF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = ["phaseA", *[f"bracket_{lr_name(lr)}" for lr in (state["bracket_lrs"] or [])], "main",
            *(state["ablation_runs"] or [])]
    report = {"pipeline": state.d, "runs": {}}
    logs = {}
    for run in runs:
        recs = run_log(run)
        if not recs:
            continue
        logs[run] = recs
        start = next((r for r in recs if r.get("event") in ("fresh_start", "resume")), {})
        report["runs"][run] = {"done": run_done(run), "val": [{k: v for k, v in r.items() if k != "samples"}
                                                              for r in _val(recs)],
                               "steps_per_epoch": start.get("steps_per_epoch"),
                               "hparams": {k: start.get(k) for k in ("phase", "init", "peak_lr", "warmup_steps",
                                                                      "total_steps", "stop_step", "micro_batch",
                                                                      "accum", "world")}}
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    for run in ("phaseA", "main"):
        recs = logs.get(run)
        if not recs:
            continue
        tr = [r for r in recs if r.get("type") == "train"]
        va = _val(recs)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot([r["step"] for r in tr], [r["loss"] for r in tr], lw=0.8, label="train loss (synthetic pairs)")
        axes[0].plot([r["step"] for r in va], [r["val_syn_loss"] for r in va], "o-", label="val_syn loss")
        axes[0].set(xlabel="step", ylabel="target-token cross-entropy (nats)", title=f"{run}: loss", yscale="log")
        axes[0].legend()
        for part, style in (("clean", "o-"), ("contaminated", "s--")):
            key = f"dev_cer_{part}"
            if va and key in va[0]:
                axes[1].plot([r["step"] for r in va], [r[key] for r in va], style, label=f"dev_real CER ({part})")
                axes[1].plot([r["step"] for r in va], [r[f"dev_wer_{part}"] for r in va], style, alpha=0.5,
                             label=f"dev_real WER ({part})")
        axes[1].set(xlabel="step", ylabel="error rate", title=f"{run}: dev_real CER / WER (masked greedy)",
                    yscale="log")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "plots" / f"stage7e_{run}.png", dpi=130)
        plt.close(fig)
    if state["bracket"]:
        b = sorted(state["bracket"].items(), key=lambda kv: float(kv[0]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([float(k) for k, _ in b], [v["dev_cer"] for _, v in b], "o-", label="selection dev CER at step 1,500")
        ax.set(xscale="log", xlabel="peak LR (phase B)", ylabel="dev_real CER (clean part)",
               title="Stage 7e LR bracket")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "plots" / "stage7e_bracket.png", dpi=130)
        plt.close(fig)
    for f in sorted(OUT.glob("forgetting_*.json")):
        report.setdefault("forgetting", {})[f.stem[len("forgetting_"):]] = json.loads(f.read_text())
    for name in ("g5_after_phaseA.json", "samples_dev_real.json"):
        if (OUT / name).exists():
            report[name[:-5]] = json.loads((OUT / name).read_text())
    (OUT / "stage7e.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    log(f"report: {len(report['runs'])} runs")


# ---------------------------------------------------------------- the pipeline
def pipeline() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = State()
    state.set()
    ensure_data(state)

    if not (OUT / "forgetting_base10240.json").exists() and not state["forgetting_base_done"]:
        sub(state, "forgetting_base", "forgetting", "--model", "base10240", "--tag", "base10240")
        sub(state, "forgetting_resized", "forgetting", "--model", "resized", "--tag", "resized")
        state.set(forgetting_base_done=True)
        publish("forgetting baseline")

    sft(state, "phaseA", "--phase", "A", "--init", "base", "--eval-every", "250", "--save-every", "500")
    if not state["g5_after_phaseA"]:
        sub(state, "g5check", "g5check")
        state.set(g5_after_phaseA=True)
        publish("phaseA")

    if not state["lr"]:
        lrs = list(BRACKET)
        results = {}

        def bracket_run(lr: float) -> None:
            d = sft(state, f"bracket_{lr_name(lr)}", "--phase", "B", "--init", "run:phaseA", "--peak-lr", str(lr),
                    "--total-steps", "1500", "--warmup-steps", "100", "--eval-every", "1500",
                    "--save-every", "1500", "--skip-step0-eval")
            assert d["best"]["step"] == 1500, d["best"]
            results[lr_name(lr)] = {"lr": lr, "dev_cer": d["best"]["dev_cer"], "dev_wer": d["best"]["dev_wer"],
                                    "dev_em": d["best"]["dev_em"], "val_syn_loss": d["best"]["val_syn_loss"],
                                    "dev_parts": d["best"].get("dev_parts")}
            state.set(bracket_lrs=lrs, bracket=results)

        for lr in BRACKET:
            bracket_run(lr)
        best = min(BRACKET, key=lambda lr: results[lr_name(lr)]["dev_cer"])
        if best in EDGE_EXTRA:
            extra = EDGE_EXTRA[best]
            log(f"bracket best {lr_name(best)} is at the edge; adding {lr_name(extra)}")
            lrs.append(extra)
            bracket_run(extra)
        pick = min(lrs, key=lambda lr: results[lr_name(lr)]["dev_cer"])
        state.set(lr=pick, lr_rule="lowest selection (clean) dev_real CER at step 1,500")
        log(f"bracket pick: {lr_name(pick)} ({ {k: round(v['dev_cer'], 4) for k, v in results.items()} })")
        publish("bracket")
        api = hub.get_api()
        for lr in lrs:                                    # spec 1.4: only main/ablations keep weights
            for folder in ("best", "final"):
                try:
                    api.delete_folder(f"{ROOT}bracket_{lr_name(lr)}/{folder}", repo_id=REPO, repo_type="model",
                                      commit_message=f"delete bracket weights {lr_name(lr)}/{folder}")
                except Exception as e:  # noqa: BLE001 - already deleted on a re-run
                    log(f"delete bracket_{lr_name(lr)}/{folder}: {type(e).__name__}")
    lr = state["lr"]

    sft(state, "main", "--phase", "B", "--init", "run:phaseA", "--peak-lr", str(lr), "--epochs", "2",
        "--warmup-frac", "0.03", "--eval-every", "1000", "--save-every", "2000")
    if not state["main_post_done"]:
        sub(state, "forgetting_main_best", "forgetting", "--model", "main/best", "--tag", "main_best")
        sub(state, "forgetting_main_final", "forgetting", "--model", "main/final", "--tag", "main_final")
        sub(state, "samples", "samples")
        state.set(main_post_done=True)
        sub(state, "report", "report")
        publish("main")

    common = ["--phase", "B", "--epochs", "2", "--stop-after-epochs", "1", "--warmup-frac", "0.03",
              "--eval-every", "3125", "--save-every", "5000", "--skip-step0-eval"]
    ablations = [("A2_onephase", "base", lr)] + [(f"A1_scratch_{lr_name(x)}", "random", x)
                                                 for x in dict.fromkeys([lr, A1_ALT_LR])]
    state.set(ablation_runs=[name for name, _, _ in ablations])
    for name, init, a_lr in ablations:
        sft(state, name, "--init", init, "--peak-lr", str(a_lr), *common)
        publish(name)
    sub(state, "report", "report")
    state.set(finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    publish("all")
    log("Stage 7e pipeline finished")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="pipeline", choices=["pipeline", "forgetting", "g5check", "samples",
                                                                  "report"])
    p.add_argument("--model")
    p.add_argument("--tag")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "pipeline":
        pipeline()
    elif args.cmd == "forgetting":
        cmd_forgetting(args.model, args.tag)
    elif args.cmd == "g5check":
        cmd_g5check()
    elif args.cmd == "samples":
        cmd_samples(State())
    elif args.cmd == "report":
        cmd_report(State())


if __name__ == "__main__":
    main()
