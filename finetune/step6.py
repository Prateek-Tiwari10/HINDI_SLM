"""D7 Step 6: the one-time final evaluation of d7/main/best on test_real + test_chat, and the public release.

    python -m finetune.step6                                  # the whole step (kernel hindi-lm-d7-step6, 2x T4)
    python -m finetune.step6 decode --shard 0 --of 2          # worker: one GPU, every 2nd test pair

Spec: instructions/D7_step6_SPEC.md (rules fixed before the run; repeated in step6_RESULTS.md).
 1. Pins: pairs romanize-v2.1 @ DATA_REV, checkpoint d7/main/best/model.pt (sha256 recorded).
 2. Gate: the inference prompt (finetune/translit.py) equals the stored Stage 7c prompt for every one-chunk pair.
 3. Decode test_real + test_chat once (greedy, mask, KV cache, fp16 on GPU), one worker per GPU.
    Hypotheses go to HF at once; a re-run reuses them and never decodes test again.
 4. Word-lookup baseline from the 600k train pairs; ambiguous words found from the same lexicon.
 5. Raw + normalized CER/WER/EM with bootstrap CIs; ambiguity slices; retrain trigger; 15 examples.
 6. Release: export safetensors + config + tokenizer; gate: identical CPU logits to the .pt checkpoint.
 7. CPU latency of the release (1 thread and all threads); CPU outputs of the 15 examples.
 8. Results to d7/step6_results/, then the release to the public HF repo (model card with these numbers).
step6.json holds aggregate numbers only: no test_chat sentence leaves the private repos.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data import hub
from data.config import load_config
from eval import translit_eval as te
from finetune.data import PairArrays
from finetune.sft import CONFIG, fetch_pairs, read_jsonl_gz
from finetune.translit import (MAX_CHUNK_CHARS, Transliterator, export_release, load_release, prompt_ids,
                               split_input)
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer, normalize_rom

DATA_REV = "74e39da0eb3c1c50a60826b78c1689b08edbb326"   # romanize-v2.1, the pairs Stage 7e trained on
MODEL_PATH = "d7/main/best/model.pt"
PUBLIC_REPO = "Prateek-Tiwari10/hinglish-to-devanagari-61m"
GITHUB = "https://github.com/Prateek-Tiwari10/HINDI_SLM"
SPLITS = ("test_real", "test_chat")
FORMS = {"raw": te.raw_dev, "normalized": te.norm_dev}
SEED = 1337
OUT = Path(os.environ.get("STEP6_OUT", "/kaggle/working/out/step6"))
DL = Path("/kaggle/tmp/step6_dl")
EXAMPLE_MAX_CHARS = 120

cfg = load_config(CONFIG)
REPO = cfg["checkpoint"]["hf_repo"]
RESULTS = "d7/step6_results"


def log(msg: str) -> None:
    print(f"[step6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_split(split: str) -> tuple[list[dict], PairArrays, dict]:
    pair_dir = fetch_pairs(cfg, (split,), DATA_REV)
    return (read_jsonl_gz(pair_dir / f"{split}.jsonl.gz"), PairArrays(pair_dir, split),
            json.loads((pair_dir / "meta.json").read_text()))


def checkpoint_file() -> Path:
    return Path(hub.download(REPO, MODEL_PATH, DL / "ckpt", repo_type="model"))


def load_pt_model(tok, device: str):
    from model.surgery import load_finetuned
    model, ckpt = load_finetuned(checkpoint_file(), tok)
    model.cfg.attn_impl = "sdpa"
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------- worker: decode one shard on one GPU
def cmd_decode(shard: int, of: int) -> None:
    torch.manual_seed(SEED)
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    model, _ = load_pt_model(tok, "cuda")
    for split in SPLITS:
        records, _, meta = load_split(split)
        t = Transliterator(model, tok, meta["decode_cap"], "cuda")
        path = OUT / f"hyps_{split}_{shard}.jsonl"
        with open(path, "w") as f:
            for i in range(shard, len(records), of):
                r = t(records[i]["rom"])
                f.write(json.dumps({"i": i, "id": records[i]["id"], "hyp": r.output, "chunks": r.chunks,
                                    "new_tokens": r.new_tokens, "seconds": round(r.seconds, 4)},
                                   ensure_ascii=False) + "\n")
                if (i // of) % 500 == 0:
                    f.flush()
                    log(f"shard {shard}: {split} {i}/{len(records)}")


def decode_all() -> dict[str, list[dict]]:
    """Hypotheses for both test sets: from HF if a previous attempt uploaded them, else decoded now."""
    have = set(hub.list_files(REPO, f"{RESULTS}/", repo_type="model"))
    if all(f"{RESULTS}/hyps_{s}.jsonl" in have for s in SPLITS):
        log("hypotheses already on HF (earlier attempt): reusing them, test is not decoded again")
        return {s: read_jsonl(Path(hub.download(REPO, f"{RESULTS}/hyps_{s}.jsonl", DL / "hyps", repo_type="model")))
                for s in SPLITS}
    n_gpu = max(1, torch.cuda.device_count())
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, "-m", "finetune.step6", "decode", "--shard", str(k), "--of", str(n_gpu)],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in range(n_gpu)]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(f"[step6] decode workers failed: exit codes {codes}")
    hyps = {}
    for s in SPLITS:
        rows = sorted((r for k in range(n_gpu) for r in read_jsonl(OUT / f"hyps_{s}_{k}.jsonl")), key=lambda r: r["i"])
        assert [r["i"] for r in rows] == list(range(len(rows))), f"{s}: missing or duplicate hypotheses"
        (OUT / f"hyps_{s}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hub.upload_file(REPO, OUT / f"hyps_{s}.jsonl", f"{RESULTS}/hyps_{s}.jsonl", repo_type="model",
                        message=f"step6: {s} hypotheses")
        hyps[s] = rows
    log(f"decoded in {time.time() - t0:.0f} s on {n_gpu} GPU(s); hypotheses uploaded")
    return hyps


# ---------------------------------------------------------------- examples
def pick_examples(records, hyps, base, norm_stats, hits_m, hits_b, clean_idx) -> list[dict]:
    """Spec §6: 5 best, 5 ambiguous, 5 worst from test_real clean, inputs <= 120 chars."""
    rng = np.random.default_rng(SEED)
    info = {}
    for i in clean_idx:
        if len(normalize_rom(records[i]["rom"])) > EXAMPLE_MAX_CHARS:
            continue
        st = norm_stats[i]
        info[i] = {"cer": float(st[0] / max(st[1], 1)), "exact": bool(st[4]), "words": int(st[3])}

    def draw(pool, k):
        pool = sorted(pool)
        return [int(x) for x in rng.choice(pool, size=min(k, len(pool)), replace=False)] if pool else []

    best = draw([i for i, v in info.items() if v["exact"] and 6 <= v["words"] <= 15], 5)
    amb = [i for i in info if hits_m[i]]
    amb_win = [i for i in amb if i not in best and all(ok for _, ok in hits_m[i]) and not all(ok for _, ok in hits_b[i])]
    ambiguous = draw(amb_win, 5)
    if len(ambiguous) < 5:
        rest = [i for i in amb if i not in best and i not in ambiguous and all(ok for _, ok in hits_m[i])]
        ambiguous += draw(rest, 5 - len(ambiguous))
    used = set(best) | set(ambiguous)
    worst = sorted((i for i, v in info.items() if v["words"] >= 4 and i not in used),
                   key=lambda i: (-info[i]["cer"], i))[:5]
    out = []
    for cat, idx in (("best", best), ("ambiguous", ambiguous), ("worst", worst)):
        for i in idx:
            r = records[i]
            out.append({"category": cat, "id": r["id"], "rom": r["rom"], "ref": r["dev"], "hyp": hyps[i],
                        "baseline": base[i], "cer_norm": round(info[i]["cer"], 4), "exact_norm": info[i]["exact"],
                        "ambiguous_words": [{"word": w, "model": ok, "baseline": okb} for (w, ok), (_, okb)
                                            in zip(hits_m[i] or [], hits_b[i] or [])]})
    return out


# ---------------------------------------------------------------- CPU: release gate + latency
def latency(t: Transliterator, texts: list[str], threads: int) -> tuple[dict, list[str]]:
    torch.set_num_threads(threads)
    for x in texts[:5]:
        t(x)
    ms, toks, outs = [], [], []
    for x in texts:
        r = t(x)
        ms.append(r.seconds * 1000)
        toks.append(r.new_tokens)
        outs.append(r.output)
    a = np.asarray(ms)
    return {"threads": threads, "n": len(texts), "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1), "mean_ms": round(float(a.mean()), 1),
            "max_ms": round(float(a.max()), 1), "mean_new_tokens": round(float(np.mean(toks)), 1),
            "tokens_per_s": round(sum(toks) / (sum(ms) / 1000), 1)}, outs


def release_and_latency(tok, cap: dict, examples: list[dict], records: dict, hyps: dict, clean_idx: list[int]) -> dict:
    rel_dir = OUT / "release"
    shutil.rmtree(rel_dir, ignore_errors=True)
    torch.set_num_threads(os.cpu_count() or 1)
    model_pt, ckpt = load_pt_model(tok, "cpu")
    step = ckpt.get("step")
    source = {"checkpoint": f"hindi-lm-checkpoints/{MODEL_PATH}", "sha256": sha256(checkpoint_file()),
              "step": int(step) if isinstance(step, (int, float)) else None, "pairs": f"romanize-v2.1@{DATA_REV[:10]}"}
    export_release(copy.deepcopy(model_pt), tok, cap, rel_dir, source)

    t0 = time.perf_counter()
    rel = load_release(rel_dir, "cpu")
    load_s = time.perf_counter() - t0
    diffs = []
    with torch.inference_mode():
        for i in clean_idx[:3]:
            x = torch.tensor([prompt_ids(tok, normalize_rom(records["test_real"][i]["rom"]))])
            diffs.append(float((model_pt(x)[0] - rel.model(x)[0]).abs().max()))
    if max(diffs) != 0.0:
        sys.exit(f"[step6] release gate failed: max logit diff {max(diffs)} vs the .pt checkpoint")
    log(f"release gate passed (max logit diff 0.0); release load {load_s:.2f} s")
    del model_pt

    rng = np.random.default_rng(SEED)
    chat_i = rng.choice(len(records["test_chat"]), size=min(100, len(records["test_chat"])), replace=False)
    short = [i for i in clean_idx if len(normalize_rom(records["test_real"][i]["rom"])) <= MAX_CHUNK_CHARS]
    real_i = rng.choice(short, size=min(100, len(short)), replace=False)
    sets = {"test_chat": sorted(int(i) for i in chat_i), "test_real_clean": sorted(int(i) for i in real_i)}
    lat, agree = {}, {}
    for threads in sorted({1, os.cpu_count() or 1}):
        for name, idx in sets.items():
            split = "test_chat" if name == "test_chat" else "test_real"
            stats, outs = latency(rel, [records[split][i]["rom"] for i in idx], threads)
            tag = f"cpu_{threads}_thread{'s' if threads > 1 else ''}_{name}"
            lat[tag] = stats
            log(f"latency {tag}: {stats}")
            if threads == 1:
                agree[name] = round(float(np.mean([te.norm_dev(o) == te.norm_dev(hyps[split][i])
                                                   for o, i in zip(outs, idx)])), 4)
    torch.set_num_threads(1)
    for ex in examples:
        r = rel(ex["rom"])
        ex["hyp_cpu"], ex["cpu_ms_1_thread"] = r.output, round(r.seconds * 1000, 1)
    lat["model_load_s"] = round(load_s, 2)
    lat["cpu"] = {"count": os.cpu_count(), "machine": platform.machine(), "processor": platform.processor()}
    lat["cpu_gpu_agreement_normalized_1_thread"] = agree
    return {"dir": rel_dir, "source": source, "latency": lat}


# ---------------------------------------------------------------- model card
def model_card(m: dict, source: dict) -> str:
    def row(name, d):
        x, r = d["normalized"], d["raw"]
        return (f"| {name} | {x['n']:,} | **{x['cer']:.3f}** [{x['cer_ci95'][0]:.3f}, {x['cer_ci95'][1]:.3f}] | "
                f"{x['wer']:.3f} | {x['exact_match']:.3f} | {r['cer']:.3f} |")
    tr, tc = m["model"]["test_real"], m["model"]["test_chat"]
    br, bc = m["baseline"]["test_real"], m["baseline"]["test_chat"]
    amb = m["ambiguity"]["pooled"]
    lat = m["latency"]["cpu_1_thread_test_chat"]
    return f"""---
language:
- hi
tags:
- transliteration
- hinglish
- devanagari
- hindi
---

# Hinglish → Devanagari, 61M (trained from scratch)

Turns Hindi typed in Latin letters into Devanagari:

```
kal meeting cancel ho gayi kya  →  कल मीटिंग कैंसिल हो गई क्या
```

- A 61M-parameter LLaMA-style decoder, **pretrained from scratch** on 1.94B tokens of cleaned Hindi, then
  fine-tuned on 600k synthetic Hinglish → Devanagari pairs.
- The tokenizer's Hindi pieces are unchanged. Latin input is one token per character, and the output is masked to Devanagari pieces only.
- Code, data pipeline, training logs and evaluation: {GITHUB}
- This is not a `transformers` model. Load it with `finetune/translit.py` from the GitHub repo (below).

## Results (held-out test sets, evaluated once)

Normalized scoring (NFC, nukta ignored, a final `।` ignored); 95% bootstrap CI over sentences.

| test set | n | CER [95% CI] | WER | exact | raw CER |
| --- | --- | --- | --- | --- | --- |
{row("Dakshina test, all", tr["all"])}
{row("Dakshina test, clean (not in pretraining)", tr["clean"])}
{row("Chat sentences (private set)", tc["all"])}
{row("Word-lookup baseline, Dakshina all", br["all"])}
{row("Word-lookup baseline, chat", bc["all"])}

- **Ambiguous words** (*ki* → कि/की, *main* → मैं/में and {m["ambiguity"]["n_words"]} Roman words in all, found automatically):
  the model gets **{amb["model"]:.3f}** of {amb["occurrences"]:,} occurrences right, the word-lookup baseline {amb["baseline"]:.3f}.
- **CPU speed:** {lat["p50_ms"]:.0f} ms median per chat sentence on one CPU thread (p95 {lat["p95_ms"]:.0f} ms), greedy with a KV cache.

## Use

```bash
git clone {GITHUB} && cd HINDI_SLM
pip install torch sentencepiece safetensors huggingface_hub pyyaml numpy
```

```python
from huggingface_hub import snapshot_download
from finetune.translit import load_release

t = load_release(snapshot_download("{PUBLIC_REPO}"))
print(t("kal meeting cancel ho gayi kya").output)
```

Inputs longer than {MAX_CHUNK_CHARS} characters are split (at sentence ends, then at word boundaries) and transliterated piece by piece.

## Limitations

- **Transliteration only.** Fine-tuning erased the base model's plain-Hindi ability: language-model loss on held-out Hindi
  text rose from 2.79 to 8.75. The fix is to mix plain-Hindi batches back into fine-tuning; that is future work.
- **Typical errors:** rare names, singular vs plural (है/हैं), badly misspelled input.
- **Training pairs are synthetic.** They are our own Hindi sentences romanized by a generator that uses human-attested spellings. Test inputs are
  human-typed: Dakshina (Wikipedia sentences) and a small private chat set.
- **Contamination:** about 75% of Dakshina test sentences appear in the pretraining text (Devanagari side only), so results are also
  reported on the clean part.

## Files

`model.safetensors` (fp32, tied embeddings), `config.json` (architecture, decode settings, source checkpoint),
`tokenizer_ext.json` + `hi_unigram_10240.model` (tokenizer), `examples.json` (15 test examples: best, ambiguous, worst).
Source: `{source["checkpoint"]}`, sha256 `{source["sha256"][:16]}…`.
"""


# ---------------------------------------------------------------- the step
def pipeline() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if f"{RESULTS}/step6.json" in set(hub.list_files(REPO, f"{RESULTS}/", repo_type="model")):
        log("step6.json already on HF: Step 6 runs once. Nothing to do.")
        return
    wall = {}
    t0 = time.time()
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    torch.manual_seed(SEED)

    records, arrays, meta = {}, {}, None
    for s in SPLITS:
        records[s], arrays[s], meta = load_split(s)
    checkpoint_file()                                  # download once, before the workers start
    gate = {}
    for s in SPLITS:
        one = [i for i, r in enumerate(records[s]) if len(split_input(r["rom"])) == 1]
        bad = [i for i in one if prompt_ids(tok, normalize_rom(records[s][i]["rom"])) != arrays[s].prompt(i)]
        gate[s] = {"one_chunk_pairs": len(one), "split_pairs": len(records[s]) - len(one), "prompt_mismatches": len(bad)}
        if bad:
            sys.exit(f"[step6] prompt-format gate failed on {s}: {len(bad)} pairs, first {bad[:5]}")
    log(f"prompt-format gate passed: {gate}")

    hyp_rows = decode_all()
    wall["decode_s"] = round(time.time() - t0)
    hyps = {s: [r["hyp"] for r in hyp_rows[s]] for s in SPLITS}

    t1 = time.time()
    train_path = hub.download(cfg["data"]["hf_repo"], f"{cfg['data']['path']}train.jsonl.gz", DL / "train",
                              revision=DATA_REV)
    with gzip.open(train_path, "rt", encoding="utf-8") as f:
        lex, lex_stats = te.build_lexicon((r["rom"], r["dev"]) for r in map(json.loads, f))
    table = te.lookup_table(lex)
    amb = te.ambiguous_words(lex)
    base = {s: [te.baseline(r["rom"], table) for r in records[s]] for s in SPLITS}
    wall["baseline_s"] = round(time.time() - t1)
    log(f"lexicon: {lex_stats}; {len(amb)} ambiguous words")

    t1 = time.time()
    refs = {s: [r["dev"] for r in records[s]] for s in SPLITS}
    systems = {"model": hyps, "baseline": base}
    stats = {name: {s: {f: te.sentence_stats(out[s], refs[s], fn) for f, fn in FORMS.items()} for s in SPLITS}
             for name, out in systems.items()}
    tr = records["test_real"]
    parts = {"test_real": {"all": list(range(len(tr))),
                           "clean": [i for i, r in enumerate(tr) if r.get("contamination", "clean") == "clean"],
                           "contaminated": [i for i, r in enumerate(tr) if r.get("contamination", "clean") != "clean"],
                           "fits_one_chunk": [i for i, r in enumerate(hyp_rows["test_real"]) if r["chunks"] == 1],
                           "long_split": [i for i, r in enumerate(hyp_rows["test_real"]) if r["chunks"] > 1]},
             "test_chat": {"all": list(range(len(records["test_chat"])))}}
    metrics = {name: {s: {p: {f: te.summarize(stats[name][s][f][idx]) for f in FORMS}
                          for p, idx in parts[s].items() if idx} for s in SPLITS} for name in systems}
    metrics["model_minus_baseline_ci95"] = {
        s: {p: {f: te.bootstrap(stats["model"][s][f][idx], stats["baseline"][s][f][idx]) for f in FORMS}
            for p, idx in parts[s].items() if p in ("all", "clean")} for s in SPLITS}
    wall["scoring_s"] = round(time.time() - t1)
    log(f"model test_real all normalized: {metrics['model']['test_real']['all']['normalized']}")

    hits_m = {s: [te.ambiguity_hits(r["rom"], r["dev"], h, amb) for r, h in zip(records[s], hyps[s])] for s in SPLITS}
    hits_b = {s: [te.ambiguity_hits(r["rom"], r["dev"], h, amb) for r, h in zip(records[s], base[s])] for s in SPLITS}
    clean = parts["test_real"]["clean"]
    slices = {s: {"model": te.slice_report(hits_m[s]), "baseline": te.slice_report(hits_b[s])} for s in SPLITS}
    slices["test_real_clean"] = {"model": te.slice_report([hits_m["test_real"][i] for i in clean]),
                                 "baseline": te.slice_report([hits_b["test_real"][i] for i in clean])}
    pooled_m = te.slice_report(hits_m["test_real"] + hits_m["test_chat"])
    pooled_b = te.slice_report(hits_b["test_real"] + hits_b["test_chat"])

    def acc(rep, w):
        v = rep["per_word"].get(w)
        return round(v["correct"] / v["n"], 4) if v and v["n"] else None

    shown = list(dict.fromkeys(list(te.AMBIG_ALWAYS) + list(pooled_m["per_word"])[:25]))
    table_words = [{"word": w, "n_test": pooled_m["per_word"].get(w, {}).get("n", 0), "model_acc": acc(pooled_m, w),
                    "baseline_acc": acc(pooled_b, w), "baseline_pick": table.get(w), **amb[w]} for w in shown]
    retrain = pooled_m["accuracy"] is not None and pooled_m["accuracy"] < pooled_b["accuracy"]
    ambiguity = {"rule": {"always": list(te.AMBIG_ALWAYS), "min_train_count": te.AMBIG_MIN_COUNT,
                          "second_share": te.AMBIG_SECOND_SHARE},
                 "n_words": len(amb), "words": {w: amb[w] for w in sorted(amb, key=lambda w: -amb[w]["n_train"])},
                 "per_set": {k: {"model": v["model"]["accuracy"], "baseline": v["baseline"]["accuracy"],
                                 "occurrences": v["model"]["occurrences"],
                                 "aligned_sentences": v["model"]["aligned_sentences"],
                                 "sentences": v["model"]["sentences"]} for k, v in slices.items()},
                 "pooled": {"model": pooled_m["accuracy"], "baseline": pooled_b["accuracy"],
                            "occurrences": pooled_m["occurrences"]},
                 "table": table_words,
                 "retrain_triggered": retrain}
    log(f"ambiguity pooled: model {pooled_m['accuracy']} vs baseline {pooled_b['accuracy']}; retrain={retrain}")

    examples = pick_examples(tr, hyps["test_real"], base["test_real"], stats["model"]["test_real"]["normalized"],
                             hits_m["test_real"], hits_b["test_real"], clean)

    t2 = time.time()
    rel = release_and_latency(tok, meta["decode_cap"], examples, records, hyps, clean)
    wall["cpu_s"] = round(time.time() - t2)

    result = {
        "spec": "instructions/D7_step6_SPEC.md (2026-11-26)",
        "pins": {**rel["source"], "pairs": f"{cfg['data']['hf_repo']}@{DATA_REV}", "path": cfg["data"]["path"]},
        "protocol": {"decoding": "greedy, Devanagari-only mask, KV cache, fp16 autocast on GPU",
                     "max_chunk_chars": MAX_CHUNK_CHARS, "decode_cap": meta["decode_cap"], "seed": SEED,
                     "normalized": "N1 NFC; N2 nukta removed (NFD first); N3 whitespace collapsed; N4 one final danda removed",
                     "raw": "NFC + whitespace collapsed"},
        "gates": {"prompt_format": gate, "release_logits_identical": True},
        "decode": {s: {"chunks_total": int(sum(r["chunks"] for r in hyp_rows[s])),
                       "split_inputs": int(sum(r["chunks"] > 1 for r in hyp_rows[s])),
                       "new_tokens": int(sum(r["new_tokens"] for r in hyp_rows[s])),
                       "gpu_seconds": round(sum(r["seconds"] for r in hyp_rows[s]), 1)} for s in SPLITS},
        "parts_n": {s: {p: len(v) for p, v in parts[s].items()} for s in SPLITS},
        **metrics,
        "lexicon": lex_stats,
        "ambiguity": ambiguity,
        "latency": rel["latency"],
        "examples": examples,
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpus": torch.cuda.device_count()},
        "wall": wall,
        "release": {"repo": PUBLIC_REPO, "commit": None, "public": True, "status": "pending"},
    }
    rel_dir = rel["dir"]
    card_in = {"model": metrics["model"], "baseline": metrics["baseline"], "latency": rel["latency"],
               "ambiguity": {"n_words": len(amb), "pooled": ambiguity["pooled"]}}
    (rel_dir / "README.md").write_text(model_card(card_in, rel["source"]))
    write_json(rel_dir / "examples.json", [{k: e[k] for k in ("category", "rom", "ref", "hyp_cpu", "cer_norm")}
                                           for e in examples])
    write_json(OUT / "step6_examples.json", examples)

    def save_results(what: str) -> None:
        result["wall"]["total_s"] = round(time.time() - t0)
        write_json(OUT / "step6.json", result)
        for name in ("step6.json", "step6_examples.json"):
            hub.upload_file(REPO, OUT / name, f"{RESULTS}/{name}", repo_type="model", message=f"step6: {name} ({what})")

    save_results("before release")                    # the numbers are safe on HF even if publishing fails
    try:
        api = hub.get_api()
        api.create_repo(PUBLIC_REPO, repo_type="model", private=False, exist_ok=True)
        info = hub._retry(lambda: api.upload_folder(folder_path=str(rel_dir), repo_id=PUBLIC_REPO, repo_type="model",
                                                    commit_message="Release d7/main/best with D7 Step 6 results"),
                          "upload release")
        commit = getattr(info, "oid", None) or api.model_info(PUBLIC_REPO).sha
        result["release"] = {"repo": PUBLIC_REPO, "commit": commit, "public": True, "status": "published",
                             "files": sorted(p.name for p in rel_dir.iterdir())}
        log(f"release published: {PUBLIC_REPO}@{commit}")
    except Exception as e:  # noqa: BLE001 - the release folder stays in the kernel output for a manual upload
        result["release"]["status"] = f"failed: {type(e).__name__}: {e}"
        log(f"release upload FAILED ({type(e).__name__}); folder kept in kernel output {rel_dir}")
    save_results("final")
    log(f"Step 6 done in {result['wall']['total_s']} s; results in {REPO}/{RESULTS}/")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="pipeline", choices=["pipeline", "decode"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "decode":
        cmd_decode(args.shard, args.of)
    else:
        pipeline()


if __name__ == "__main__":
    main()
