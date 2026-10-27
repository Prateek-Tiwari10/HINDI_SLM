"""Stage 6 driver (Kaggle T4, needs HF_TOKEN): evaluate the pretrained 61M model.

1. Loss, PPL and BPB over ALL val and test windows, for best/ and for the final step
   checkpoint. During training, best/ was picked by comparing 512-window subset losses
   with the full-window loss of the last step, so both are scored here on the same sets.
2. Plots of the whole training log: train/val loss vs tokens, LR vs step, grad norm vs step.
3. Samples from best/: the 10 prompts of eval/prompts.txt × {greedy, T=0.7 top-k 50, T=1.0},
   150 new tokens each, written to samples.md.
4. Repetition rate (repeated 4-grams) and Distinct-1/2 of those samples (eval/metrics.py).

Outputs go to --out and are uploaded to d6_results/ in the checkpoints repo.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import yaml

from data import hub
from data.normalize import normalize
from eval.metrics import distinct_n, repetition_rate
from eval.plots import plot_loss_vs_tokens, plot_series, read_jsonl
from model.config import ModelConfig
from model.model import HindiLlama
from train.data import ValWindows, target_bytes, token_byte_lengths
from train.train import evaluate

SETTINGS = [  # (name, temperature, top_k)
    ("greedy", 0.0, None),
    ("T=0.7, top-k 50", 0.7, 50),
    ("T=1.0", 1.0, None),
]
NEW_TOKENS = 150
SEED = 1337


def load_model(path: Path, key: str | None, mcfg: ModelConfig, device: torch.device) -> HindiLlama:
    """HindiLlama with weights from a best/model.pt (key None) or a step ckpt.pt (key 'model')."""
    state = torch.load(path, map_location="cpu", weights_only=key is None)
    model = HindiLlama(mcfg)
    model.load_state_dict(state[key] if key else state)
    return model.to(device).eval()


def score(model: HindiLlama, windows: ValWindows, byte_len: np.ndarray, device: torch.device) -> dict:
    """Loss/PPL/BPB over every window of one split."""
    ids = list(range(windows.n_windows))
    loss = evaluate(model, windows, ids, rank=0, world=1, device=device, batch=32)
    n_tok, n_bytes = target_bytes(windows, range(windows.n_windows), byte_len)
    return {"loss": round(loss, 6), "ppl": round(math.exp(loss), 3),
            "bpb": round(loss / math.log(2) * n_tok / n_bytes, 6),
            "windows": windows.n_windows, "target_tokens": n_tok, "bytes": n_bytes}


def generate_samples(model: HindiLlama, sp, prompts: list[str], bos: int, eos: int,
                     device: torch.device) -> list[dict]:
    """One continuation per (prompt, setting); sampling is seeded per pair."""
    out = []
    for i, prompt in enumerate(prompts):
        ids = [bos] + sp.encode(normalize(prompt))
        x = torch.tensor([ids], device=device)
        for j, (name, temp, top_k) in enumerate(SETTINGS):
            gen = torch.Generator(device=device).manual_seed(SEED + 10 * i + j)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                y = model.generate(x, NEW_TOKENS, temperature=temp, top_k=top_k, eos_id=eos, generator=gen)
            new = y[0, len(ids):].tolist()
            if eos in new:
                new = new[:new.index(eos)]
            text = sp.decode(new)
            out.append({"prompt": prompt, "setting": name, "new_tokens": len(new), "text": text})
    return out


def text_metrics(samples: list[dict]) -> dict:
    """Per setting: mean 4-gram repetition rate and pooled Distinct-1/2 of the continuations."""
    res = {}
    for name, _, _ in SETTINGS:
        words = [s["text"].split() for s in samples if s["setting"] == name]
        res[name] = {"rep4_mean": round(float(np.mean([repetition_rate(w) for w in words])), 4),
                     "distinct_1": round(distinct_n(words, 1), 4),
                     "distinct_2": round(distinct_n(words, 2), 4),
                     "mean_words": round(float(np.mean([len(w) for w in words])), 1)}
    return res


def samples_md(samples: list[dict], metrics: dict, step: int) -> str:
    """samples.md: every continuation, grouped by prompt, then the metrics table."""
    lines = [f"# Stage 6 samples — `best/` (step {step:,})", "",
             f"{NEW_TOKENS} new tokens per sample (stops early at eos). Sampling seed "
             f"{SEED} + 10·prompt + setting. Prompt text is **bold**; the model's continuation follows.", ""]
    for prompt in dict.fromkeys(s["prompt"] for s in samples):
        lines += [f"## {prompt}", ""]
        for s in (s for s in samples if s["prompt"] == prompt):
            lines += [f"**{s['setting']}** ({s['new_tokens']} tokens)", "",
                      f"> **{prompt}**{s['text']}", ""]
    lines += ["## Metrics (whitespace words of the continuations)", "",
              "| setting | 4-gram repetition (mean) | Distinct-1 | Distinct-2 | mean words |",
              "| --- | --- | --- | --- | --- |"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['rep4_mean']} | {m['distinct_1']} | {m['distinct_2']} | {m['mean_words']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run Stage 6 and publish its outputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d5d6")
    args = p.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())
    mcfg = ModelConfig.from_yaml(cfg["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo, prefix = cfg["checkpoint"]["hf_repo"], cfg["checkpoint"]["hf_prefix"]
    dl = Path("/kaggle/tmp/stage6")

    # data: val and test only (train shards are not needed)
    d = cfg["data"]
    meta = json.loads(Path(hub.download(d["hf_repo"], f"{d['prefix']}meta.json", dl)).read_text())
    split_paths = {}
    for split in ("val", "test"):
        (name, _), = meta["splits"][split]["files"]
        split_paths[split] = Path(hub.download(d["hf_repo"], f"{d['prefix']}{name}", dl))
    sp = spm.SentencePieceProcessor(model_file=hub.download(cfg["tokenizer"]["hf_repo"], cfg["tokenizer"]["path"],
                                                            dl / "tokenizer", repo_type="model"))
    byte_len = token_byte_lengths(sp)
    windows = {s: ValWindows(path, d["seq_len"]) for s, path in split_paths.items()}

    # checkpoints: best/ and the final step
    best_info = json.loads(Path(hub.download(repo, f"{prefix}best/info.json", dl, repo_type="model")).read_text())
    latest = Path(hub.download(repo, f"{prefix}latest.txt", dl, repo_type="model")).read_text().strip()
    ckpts = {
        "best": (Path(hub.download(repo, f"{prefix}best/model.pt", dl, repo_type="model")), None, best_info["step"]),
        "final": (Path(hub.download(repo, f"{prefix}{latest}/ckpt.pt", dl, repo_type="model")), "model",
                  int(latest.split("_")[1])),
    }
    results = {"best_info": best_info, "checkpoints": {}}
    models = {}
    for name, (path, key, step) in ckpts.items():
        models[name] = load_model(path, key, mcfg, device)
        results["checkpoints"][name] = {"step": step, **{s: score(models[name], w, byte_len, device)
                                                         for s, w in windows.items()}}
        print(f"[stage6] {name}: {json.dumps(results['checkpoints'][name])}", flush=True)
    del models["final"]

    # plots of the whole training log
    recs = read_jsonl(hub.download(repo, f"{prefix}logs/pretrain_61m.jsonl", dl, repo_type="model"))
    plot_loss_vs_tokens(recs, out / "plots" / "stage6_loss_vs_tokens.png")
    plot_series(recs, "lr", "learning rate", "Learning rate schedule", out / "plots" / "stage6_lr.png")
    plot_series(recs, "grad_norm", "global grad norm (before clipping)", "Gradient norm",
                out / "plots" / "stage6_grad_norm.png", log_y=True)

    # samples + text metrics from best/
    prompts = [ln.strip() for ln in Path("eval/prompts.txt").read_text().splitlines() if ln.strip()]
    assert (sp.bos_id(), sp.eos_id()) == (2, 3), "tokenized data uses bos=2, eos=3"
    samples = generate_samples(models["best"], sp, prompts, sp.bos_id(), sp.eos_id(), device)
    metrics = text_metrics(samples)
    results.update({"text_metrics": metrics, "n_prompts": len(prompts), "new_tokens": NEW_TOKENS,
                    "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__})
    (out / "samples.md").write_text(samples_md(samples, metrics, ckpts["best"][2]))
    (out / "samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples))
    (out / "stage6.json").write_text(json.dumps(results, indent=1))
    print(f"[stage6] {json.dumps(results, indent=1)}", flush=True)
    hub.upload_folder(repo, out, "d6_results", repo_type="model", message="d6 results after stage6")


if __name__ == "__main__":
    main()
