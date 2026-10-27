"""Stage 4b: train tokenizers with identical settings (the ladder rungs and the 8-tokenizer grid).

Each tokenizer trains in its own child process:
- its peak RAM (ru_maxrss) is measured in isolation;
- an out-of-memory kill loses only that one tokenizer.
Finished tokenizers are uploaded to the tokenizer repo under <sweep_prefix>. A rerun skips them.
"""

import json
import multiprocessing as mp
import os
import resource
import shutil
import time

import sentencepiece as spm

from data import hub
from data.config import REPO_ROOT


def tokenizer_name(model_type, vocab):
    return f"hi_{model_type}_{vocab}"


def resolve_threads(cfg):
    """runtime.threads as a number: 'auto' means every core of the machine, up to max_threads."""
    threads = cfg["runtime"]["threads"]
    return min(os.cpu_count(), cfg["runtime"]["max_threads"]) if threads == "auto" else int(threads)


def trainer_kwargs(cfg, input_files, model_prefix, model_type, vocab, n_chars):
    """The exact SentencePiece settings from the decision log, read from config.

    input_files: the band files of one sample (SentencePiece reads them as one corpus).
    n_chars: characters in the sample; above large_corpus_chars the 64-bit suffix array is used.
    """
    t = cfg["trainer"]
    specials = t["chat_symbols"] + [f"<|reserved_{i}|>" for i in range(t["num_reserved_symbols"])]
    return dict(
        input=",".join(str(f) for f in input_files),
        model_prefix=str(model_prefix),
        model_type=model_type,
        vocab_size=vocab,
        character_coverage=t["character_coverage"],
        byte_fallback=t["byte_fallback"],
        split_digits=t["split_digits"],
        normalization_rule_name=t["normalization_rule_name"],  # identity: normalize() already ran
        pad_id=t["pad_id"], unk_id=t["unk_id"], bos_id=t["bos_id"], eos_id=t["eos_id"],
        user_defined_symbols=specials,
        input_sentence_size=t["input_sentence_size"],
        shuffle_input_sentence=t["shuffle_input_sentence"],
        max_sentence_length=t["max_sentence_length"],
        train_extremely_large_corpus=n_chars > t["large_corpus_chars"],
        num_threads=resolve_threads(cfg),
    )


def _train_child(kwargs, queue):
    """Runs in the child process: train, then report time and peak RSS."""
    start = time.time()
    spm.SentencePieceTrainer.train(**kwargs)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    queue.put({"train_seconds": round(time.time() - start, 1), "peak_ram_gb": round(peak_kb / 1e6, 2)})


def train_one(cfg, input_files, n_chars, name, model_type, vocab):
    """Train one tokenizer in a fresh process. Returns its info dict (with 'error' on failure)."""
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = trainer_kwargs(cfg, input_files, out_dir / name, model_type, vocab, n_chars)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_train_child, args=(kwargs, queue))
    proc.start()
    proc.join()
    info = {"name": name, "model_type": model_type, "vocab_size": vocab, "threads": kwargs["num_threads"],
            "large_corpus": kwargs["train_extremely_large_corpus"]}
    if proc.exitcode != 0:
        info["error"] = f"exit code {proc.exitcode} (negative = killed, e.g. out of memory)"
    else:
        info.update(queue.get())
    return info


def fetch_trained(cfg, name):
    """Download a finished tokenizer from the Hub into out_dir; its info dict, or None if absent."""
    repo, prefix = cfg["hf"]["tokenizer_repo"], cfg["hf"]["sweep_prefix"]
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    if f"{prefix}{name}.json" not in set(hub.list_files(repo, prefix, repo_type="model")):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".model", ".vocab", ".json"):
        src = hub.download(repo, f"{prefix}{name}{ext}", out_dir.parent / "_hub", repo_type="model")
        shutil.copy(src, out_dir / f"{name}{ext}")
    return json.loads((out_dir / f"{name}.json").read_text())


def publish(cfg, info):
    """Save the info next to the model and upload .model/.vocab/.json to <sweep_prefix>."""
    repo, prefix = cfg["hf"]["tokenizer_repo"], cfg["hf"]["sweep_prefix"]
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    name = info["name"]
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1))
    for ext in (".model", ".vocab", ".json"):
        hub.upload_file(repo, out_dir / f"{name}{ext}", f"{prefix}{name}{ext}", repo_type="model")


def train_cached(cfg, input_files, sample_size, name, model_type, vocab, before_train=None):
    """The tokenizer's info: from the Hub if already trained, else trained now and uploaded.

    sample_size: the rung_input() dict of the sample, recorded in the info.
    before_train: called right before a real training (the time-budget check).
    """
    info = fetch_trained(cfg, name)
    if info is not None:
        print(f"[train] {name}: already trained", flush=True)
        return info
    if before_train:
        before_train(name)
    print(f"[train] {name}: training on {sample_size['bytes'] / 1e9:.2f} GB, {sample_size['lines']:,} lines", flush=True)
    info = train_one(cfg, input_files, sample_size["chars"], name, model_type, vocab)
    info["sample"] = sample_size
    print(f"[train] {name}: {info}", flush=True)
    if "error" not in info:
        publish(cfg, info)
    return info


def copy_trained(cfg, src_info, name):
    """Register an already trained model under another name (same sample and settings), uploaded."""
    info = fetch_trained(cfg, name)
    if info is not None:
        return info
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    for ext in (".model", ".vocab"):
        shutil.copy(out_dir / f"{src_info['name']}{ext}", out_dir / f"{name}{ext}")
    info = dict(src_info, name=name, copied_from=src_info["name"])
    publish(cfg, info)
    print(f"[train] {name}: same model as {src_info['name']}", flush=True)
    return info


def train_grid(cfg, input_files, sample_size, reuse=None, before_train=None):
    """Every grid tokenizer on one sample. reuse: {(model_type, vocab): info} of models already trained
    on this exact sample (the chosen ladder rung), copied instead of retrained."""
    infos = []
    for model_type in cfg["sweep"]["model_types"]:
        for vocab in cfg["sweep"]["vocab_sizes"]:
            name = tokenizer_name(model_type, vocab)
            same = (reuse or {}).get((model_type, vocab))
            if same is not None and "error" not in same:
                infos.append(copy_trained(cfg, same, name))
            else:
                infos.append(train_cached(cfg, input_files, sample_size, name, model_type, vocab, before_train))
    return infos
