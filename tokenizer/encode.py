"""Stage 5: encode final/{train,val,test} with the chosen tokenizer into uint16 .bin shards.

    python -m tokenizer.encode [--tokenizer hi_unigram_10240]

1. Encode each final parquet shard: per doc, bos + tokens + eos. Save tokens + doc offsets
   to the Hub with a done-marker (resumable).
2. Assemble each split:
   - build the document list; in train, Wikipedia docs appear twice (D2 upsampling);
   - shuffle the document order with a fixed seed;
   - concatenate into train_000.bin ... (~500M tokens each), val.bin, test.bin.
   Train shards are cut every shard_tokens tokens, so one doc can span two files.
   The training loader reads them as one continuous stream.
3. Write meta.json and upload everything to tokenized/<tokenizer>/.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from data import hub
from data.config import config_hash, load_config

SPLITS = ("train", "val", "test")


def load_tokenizer(cfg, name, local):
    path = hub.download(cfg["hf"]["tokenizer_repo"], f"{cfg['hf']['sweep_prefix']}{name}.model", local, repo_type="model")
    sp = spm.SentencePieceProcessor(model_file=path)
    assert sp.get_piece_size() < 65536, "uint16 storage needs vocab_size < 65536"
    return sp


def encode_docs(sp, texts, threads):
    """List of uint16 arrays, one per doc, each bos + tokens + eos."""
    ids = sp.encode(texts, num_threads=threads)
    bos, eos = sp.bos_id(), sp.eos_id()
    return [np.array([bos, *doc, eos], dtype=np.uint16) for doc in ids]


def encode_shard(cfg, sp, split_name, parquet_remote, out_dir):
    """Encode one final parquet file -> tokens.npy, offsets.npy, sources.json, stats.json."""
    local = Path(cfg["runtime"]["work_dir"], "enc_src")
    path = hub.download(cfg["hf"]["dataset_repo"], parquet_remote, local)
    table = pq.read_table(path, columns=["source", "text", "n_bytes"]).to_pydict()
    Path(path).unlink()
    batch = cfg["encode"]["encode_batch_docs"]
    docs = []
    for start in range(0, len(table["text"]), batch):
        docs.extend(encode_docs(sp, table["text"][start : start + batch], cfg["runtime"]["threads"]))
    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens = np.concatenate(docs) if docs else np.zeros(0, dtype=np.uint16)
    stats = defaultdict(lambda: {"docs": 0, "tokens": 0, "bytes": 0})
    for source, length, n_bytes in zip(table["source"], lengths.tolist(), table["n_bytes"]):
        stats[source]["docs"] += 1
        stats[source]["tokens"] += length - 2  # content tokens, without bos/eos
        stats[source]["bytes"] += n_bytes
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tokens.npy", tokens)
    np.save(out_dir / "offsets.npy", offsets)
    (out_dir / "sources.json").write_text(json.dumps(table["source"]))
    (out_dir / "stats.json").write_text(json.dumps(stats))


def encode_all(cfg, sp, name):
    """Encode every final shard not yet marked done."""
    repo = cfg["hf"]["dataset_repo"]
    enc_prefix = f"work/encoded/{name}/"
    suffix = ".done"
    done = hub.done_markers(repo, enc_prefix, suffix)
    work = Path(cfg["runtime"]["work_dir"], "enc_out")
    for split_name in SPLITS:
        for remote in hub.list_files(repo, f"final/{split_name}/"):
            if not remote.endswith(".parquet"):
                continue
            shard = Path(remote).stem
            key = f"{enc_prefix}{split_name}/{shard}"
            if f"{key}{suffix}" in done:
                continue
            out_dir = work / split_name / shard
            encode_shard(cfg, sp, split_name, remote, out_dir)
            hub.upload_folder(repo, out_dir, key, message=f"Encoded {split_name}/{shard}")
            hub.write_marker(repo, f"{key}{suffix}")
            for f in out_dir.iterdir():
                f.unlink()
            print(f"[encode] {split_name}/{shard}", flush=True)


def load_encoded_split(cfg, name, split_name):
    """Memory-mapped tokens/offsets per shard, plus doc sources and stats."""
    repo = cfg["hf"]["dataset_repo"]
    prefix = f"work/encoded/{name}/{split_name}/"
    local = Path(cfg["runtime"]["work_dir"], "enc_in")
    shards = sorted({p[len(prefix):].split("/")[0] for p in hub.list_files(repo, prefix) if "/" in p[len(prefix):]})
    parts = []
    for shard in shards:
        get = lambda f: hub.download(repo, f"{prefix}{shard}/{f}", local)  # noqa: E731
        parts.append({
            "tokens": np.load(get("tokens.npy"), mmap_mode="r"),
            "offsets": np.load(get("offsets.npy")),
            "sources": json.loads(Path(get("sources.json")).read_text()),
            "stats": json.loads(Path(get("stats.json")).read_text()),
        })
    return parts


def document_order(parts, split_name, cfg, rng):
    """(shard, doc) pairs in shuffled order; Wikipedia repeated in train."""
    repeats = cfg["encode"]["wikipedia_train_repeats"] if split_name == "train" else 1
    shard_idx, doc_idx = [], []
    for s, part in enumerate(parts):
        for d, source in enumerate(part["sources"]):
            copies = repeats if source == "wikipedia" else 1
            shard_idx.extend([s] * copies)
            doc_idx.extend([d] * copies)
    order = rng.permutation(len(shard_idx))
    return np.array(shard_idx, dtype=np.int64)[order], np.array(doc_idx, dtype=np.int64)[order]


def write_split(parts, shard_idx, doc_idx, split_name, out_dir, shard_tokens):
    """Concatenate docs in the given order into .bin files. Returns [(file, tokens)]."""
    files = []
    buffer = np.empty(shard_tokens, dtype=np.uint16)
    filled = 0

    def flush():
        nonlocal filled
        if filled == 0:
            return
        fname = f"{split_name}.bin" if split_name != "train" else f"train_{len(files):03d}.bin"
        buffer[:filled].tofile(out_dir / fname)
        files.append((fname, int(filled)))
        filled = 0

    for s, d in zip(shard_idx.tolist(), doc_idx.tolist()):
        start, end = parts[s]["offsets"][d], parts[s]["offsets"][d + 1]
        doc = parts[s]["tokens"][start:end]
        while len(doc):
            if split_name != "train" and filled + len(doc) > len(buffer):
                buffer = np.concatenate([buffer, np.empty(max(len(doc), len(buffer)), dtype=np.uint16)])
            take = min(len(doc), len(buffer) - filled)
            buffer[filled : filled + take] = doc[:take]
            filled += take
            doc = doc[take:]
            if filled == len(buffer) and split_name == "train":
                flush()
    flush()
    return files


def split_stats(parts, split_name, cfg):
    """Per-source docs / content tokens / bytes, once and with upsampling applied."""
    repeats = cfg["encode"]["wikipedia_train_repeats"] if split_name == "train" else 1
    once = defaultdict(lambda: {"docs": 0, "tokens": 0, "bytes": 0})
    for part in parts:
        for source, v in part["stats"].items():
            for k in once[source]:
                once[source][k] += v[k]
    upsampled = {s: {k: v[k] * (repeats if s == "wikipedia" else 1) for k in v} for s, v in once.items()}
    return {"once": dict(once), "with_upsampling": upsampled}


def assemble(cfg, sp, name):
    out_dir = Path(cfg["runtime"]["work_dir"], "tokenized", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["encode"]["shuffle_seed"]
    meta = {"tokenizer": name, "vocab_size": sp.get_piece_size(), "dtype": cfg["encode"]["dtype"],
            "bos_id": sp.bos_id(), "eos_id": sp.eos_id(), "seed": seed,
            "config_hash": config_hash("configs/tokenizer.yaml"),
            "data_config_hash": config_hash("configs/data.yaml"), "splits": {}}
    for i, split_name in enumerate(SPLITS):
        parts = load_encoded_split(cfg, name, split_name)
        rng = np.random.default_rng(seed + i)
        shard_idx, doc_idx = document_order(parts, split_name, cfg, rng)
        files = write_split(parts, shard_idx, doc_idx, split_name, out_dir, cfg["encode"]["shard_tokens"])
        stats = split_stats(parts, split_name, cfg)
        meta["splits"][split_name] = {
            "files": files, "total_tokens_incl_bos_eos": sum(n for _, n in files),
            "docs": int(len(shard_idx)), "per_source": stats,
        }
        print(f"[assemble] {split_name}: {files}", flush=True)
    once = meta["splits"]["train"]["per_source"]["once"]
    total_bytes = sum(v["bytes"] for v in once.values())
    total_tokens = sum(v["tokens"] for v in once.values())
    meta["bytes_per_token"] = round(total_bytes / total_tokens, 3)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="defaults to recommended.json in the tokenizer repo")
    args = parser.parse_args()
    cfg = load_config("configs/tokenizer.yaml")
    local = Path(cfg["runtime"]["work_dir"], "tok_model")
    name = args.tokenizer
    if not name:
        rec = hub.download(cfg["hf"]["tokenizer_repo"], "recommended.json", local, repo_type="model")
        name = json.loads(Path(rec).read_text())["name"]
    print(f"[encode] tokenizer {name}", flush=True)
    sp = load_tokenizer(cfg, name, local)
    encode_all(cfg, sp, name)
    out_dir, meta = assemble(cfg, sp, name)
    hub.upload_folder(cfg["hf"]["dataset_repo"], out_dir, f"tokenized/{name}", message=f"Tokenized dataset {name}")
    report_dir = Path(cfg["runtime"]["out_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage5_meta.json").write_text(json.dumps(meta, indent=1))
    train_tokens = sum(v["tokens"] for v in meta["splits"]["train"]["per_source"]["with_upsampling"].values())
    print(f"[encode] train content tokens (with Wikipedia 2x): {train_tokens / 1e9:.3f}B; "
          f"bytes/token {meta['bytes_per_token']}", flush=True)
    if train_tokens < cfg["encode"]["min_train_tokens"]:
        print(f"STOP: train tokens below {cfg['encode']['min_train_tokens'] / 1e9:.1f}B", flush=True)


if __name__ == "__main__":
    main()
