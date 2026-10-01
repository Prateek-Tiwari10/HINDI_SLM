"""Stage 4a: tokenizer training sample (train split only) and val text for evaluation.

- Training sample: each train doc is kept with probability p = target_bytes / train_bytes.
  The draw is a hash of the doc id, so it is deterministic. Every source is sampled at
  the same rate, which keeps train's source proportions, with no upsampling.
- Docs are split into lines of at most max_line_chars (SentencePiece's line limit).
- Val text: val_docs docs from the val split, chosen the same way.
Both files are uploaded to the dataset repo so later steps and reruns can reuse them.
"""

import gzip
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import xxhash

from data import hub
from data.config import REPO_ROOT, load_config

HASH_SCALE = float(2**64)


def keep_draw(doc_id, seed):
    """Uniform number in [0, 1) derived from the doc id."""
    return xxhash.xxh64_intdigest(doc_id.encode("utf-8"), seed=seed) / HASH_SCALE


def split_long_line(line, max_chars):
    """Cut a line into pieces of <= max_chars, breaking at the last space where possible."""
    pieces = []
    while len(line) > max_chars:
        cut = line.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(line[:cut].strip())
        line = line[cut:].strip()
    if line:
        pieces.append(line)
    return pieces


def doc_to_lines(text, max_chars):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.extend(split_long_line(line, max_chars))
    return lines


def split_bytes(data_repo, split_name, local):
    """Total text bytes in a split, summed from the final/stats/*.json files."""
    total = 0
    for path in hub.list_files(data_repo, "final/stats/"):
        stats = json.loads(Path(hub.download(data_repo, path, local)).read_text())
        total += sum(v["bytes"] for v in stats.get(split_name, {}).values())
    return total


def iter_split_docs(data_repo, split_name, local):
    """(id, source, text) of every doc in final/<split>/*.parquet."""
    for path in hub.list_files(data_repo, f"final/{split_name}/"):
        if not path.endswith(".parquet"):
            continue
        local_path = hub.download(data_repo, path, local)
        table = pq.read_table(local_path, columns=["id", "source", "text"]).to_pydict()
        yield from zip(table["id"], table["source"], table["text"])
        Path(local_path).unlink()


def build_train_sample(cfg, out_path):
    s = cfg["sample"]
    data_repo = cfg["hf"]["dataset_repo"]
    local = Path(cfg["runtime"]["work_dir"], "tok_src")
    total = split_bytes(data_repo, s["split"], local)
    rate = min(1.0, s["target_bytes"] / total)
    print(f"[sample] train bytes {total / 1e9:.2f} GB -> keep rate {rate:.4f}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_bytes = kept_docs = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for doc_id, _, text in iter_split_docs(data_repo, s["split"], local):
            if keep_draw(doc_id, s["seed"]) >= rate:
                continue
            for line in doc_to_lines(text, s["max_line_chars"]):
                out.write(line + "\n")
            kept_docs += 1
            kept_bytes += len(text.encode("utf-8"))
    return {"train_bytes": total, "rate": rate, "docs": kept_docs, "bytes": kept_bytes}


def build_val_docs(cfg, out_path):
    """The val_docs val docs with the smallest id hash: a fixed random subset."""
    s = cfg["sample"]
    local = Path(cfg["runtime"]["work_dir"], "tok_src")
    docs = [(keep_draw(i, s["seed"]), i, src, t)
            for i, src, t in iter_split_docs(cfg["hf"]["dataset_repo"], "val", local)]
    docs.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for _, doc_id, source, text in docs[: s["val_docs"]]:
            f.write(json.dumps({"id": doc_id, "source": source, "text": text}, ensure_ascii=False) + "\n")
    return {"val_docs_available": len(docs), "val_docs_used": min(len(docs), s["val_docs"])}


def fetch_or_build(cfg):
    """Local paths of the train sample and val docs, building + uploading them once."""
    data_repo, prefix = cfg["hf"]["dataset_repo"], cfg["hf"]["work_prefix"]
    sample_path = REPO_ROOT / cfg["sample"]["out_path"]
    val_path = REPO_ROOT / cfg["sample"]["val_out_path"]
    remote = set(hub.list_files(data_repo, prefix))
    info_remote = f"{prefix}sample_info.json"
    local = Path(cfg["runtime"]["work_dir"], "tok_cache")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if info_remote in remote:
        gz = hub.download(data_repo, f"{prefix}tok_sample.txt.gz", local)
        with gzip.open(gz, "rb") as src, open(sample_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        shutil.copy(hub.download(data_repo, f"{prefix}val_docs.jsonl.gz", local), val_path)
        return sample_path, val_path, json.loads(Path(hub.download(data_repo, info_remote, local)).read_text())

    info = build_train_sample(cfg, sample_path)
    info.update(build_val_docs(cfg, val_path))
    local.mkdir(parents=True, exist_ok=True)
    gz = local / "tok_sample.txt.gz"
    with open(sample_path, "rb") as src, gzip.open(gz, "wb", compresslevel=3) as dst:
        shutil.copyfileobj(src, dst)
    hub.upload_file(data_repo, gz, f"{prefix}tok_sample.txt.gz")
    hub.upload_file(data_repo, val_path, f"{prefix}val_docs.jsonl.gz")
    info_path = local / "sample_info.json"
    info_path.write_text(json.dumps(info, indent=1))
    hub.upload_file(data_repo, info_path, info_remote)
    return sample_path, val_path, info


if __name__ == "__main__":
    print(fetch_or_build(load_config("configs/tokenizer.yaml"))[2])
