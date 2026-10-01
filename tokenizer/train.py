"""Stage 4b: train the 8-tokenizer grid ({bpe, unigram} x 4 vocab sizes) with identical settings.

Each tokenizer trains in its own child process:
- its peak RAM (ru_maxrss) is measured in isolation;
- an out-of-memory kill loses only that one tokenizer.
Finished tokenizers are uploaded to the tokenizer repo under sweep/. A rerun skips them.
"""

import json
import multiprocessing as mp
import resource
import time
from pathlib import Path

import sentencepiece as spm

from data import hub
from data.config import REPO_ROOT, load_config


def tokenizer_name(model_type, vocab):
    return f"hi_{model_type}_{vocab}"


def trainer_kwargs(cfg, input_path, model_prefix, model_type, vocab):
    """The exact SentencePiece settings from the decision log, read from config."""
    t = cfg["trainer"]
    specials = t["chat_symbols"] + [f"<|reserved_{i}|>" for i in range(t["num_reserved_symbols"])]
    return dict(
        input=str(input_path),
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
        num_threads=cfg["runtime"]["threads"],
    )


def _train_child(kwargs, queue):
    """Runs in the child process: train, then report time and peak RSS."""
    start = time.time()
    spm.SentencePieceTrainer.train(**kwargs)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    queue.put({"train_seconds": round(time.time() - start, 1), "peak_ram_gb": round(peak_kb / 1e6, 2)})


def train_one(cfg, input_path, model_type, vocab):
    """Train one tokenizer in a fresh process. Returns its info dict (with 'error' on failure)."""
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = tokenizer_name(model_type, vocab)
    kwargs = trainer_kwargs(cfg, input_path, out_dir / name, model_type, vocab)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_train_child, args=(kwargs, queue))
    proc.start()
    proc.join()
    info = {"name": name, "model_type": model_type, "vocab_size": vocab}
    if proc.exitcode != 0:
        info["error"] = f"exit code {proc.exitcode} (negative = killed, e.g. out of memory)"
    else:
        info.update(queue.get())
    return info


def train_grid(cfg, input_path):
    """Train every missing tokenizer; download finished ones. Returns list of info dicts."""
    repo, prefix = cfg["hf"]["tokenizer_repo"], cfg["hf"]["sweep_prefix"]
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    remote = set(hub.list_files(repo, prefix, repo_type="model"))
    infos = []
    for model_type in cfg["sweep"]["model_types"]:
        for vocab in cfg["sweep"]["vocab_sizes"]:
            name = tokenizer_name(model_type, vocab)
            if f"{prefix}{name}.json" in remote:
                for ext in (".model", ".vocab", ".json"):
                    hub.download(repo, f"{prefix}{name}{ext}", out_dir.parent / "_hub", repo_type="model")
                    src = out_dir.parent / "_hub" / prefix / f"{name}{ext}"
                    (out_dir / f"{name}{ext}").write_bytes(src.read_bytes())
                infos.append(json.loads((out_dir / f"{name}.json").read_text()))
                print(f"[train] {name}: already trained", flush=True)
                continue
            print(f"[train] {name}: training", flush=True)
            info = train_one(cfg, input_path, model_type, vocab)
            print(f"[train] {name}: {info}", flush=True)
            infos.append(info)
            if "error" in info:
                continue
            (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1))
            for ext in (".model", ".vocab", ".json"):
                hub.upload_file(repo, out_dir / f"{name}{ext}", f"{prefix}{name}{ext}", repo_type="model")
    return infos


if __name__ == "__main__":
    config = load_config("configs/tokenizer.yaml")
    train_grid(config, REPO_ROOT / config["sample"]["out_path"])
