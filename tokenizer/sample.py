"""Stage 4a: nested tokenizer training samples (train split only) and val text for evaluation.

- Every train doc gets a deterministic draw in [0, 1) from a hash of its id. With rung fractions
  f0 < f1 < ..., the doc goes to band k when f(k-1) <= draw < f(k). Rung k trains on bands 0..k,
  so the samples are nested (the 5% sample is inside the 15% one) and no line is stored twice.
- Every source is sampled at the same rate, which keeps train's source proportions, with no upsampling.
- Docs are split into lines of at most max_line_chars (SentencePiece's line limit).
- Val text: val_docs docs from the val split, chosen by the same hash.
The samples are rebuilt each run from <final_prefix> (deterministic), so nothing large is uploaded.
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


def band_of(draw, fractions):
    """Index of the first rung whose fraction is above the draw; None if the doc is in no rung."""
    for i, fraction in enumerate(fractions):
        if draw < fraction:
            return i
    return None


def all_fractions(cfg):
    """Every rung that may be trained, in increasing order (the base rungs plus the extension)."""
    lcfg = cfg["ladder"]
    return sorted(set(lcfg["fractions"]) | {lcfg["extend_fraction"]})


def split_bytes(cfg, split_name, local):
    """Total text bytes in a split, summed from the <final_prefix>stats/*.json files."""
    data_repo, prefix = cfg["hf"]["dataset_repo"], cfg["hf"]["final_prefix"]
    total = 0
    for path in hub.list_files(data_repo, f"{prefix}stats/"):
        stats = json.loads(Path(hub.download(data_repo, path, local)).read_text())
        total += sum(v["bytes"] for v in stats.get(split_name, {}).values())
    return total


def iter_split_docs(cfg, split_name, local):
    """(id, source, text) of every doc in <final_prefix><split>/*.parquet."""
    data_repo, prefix = cfg["hf"]["dataset_repo"], cfg["hf"]["final_prefix"]
    for path in hub.list_files(data_repo, f"{prefix}{split_name}/"):
        if not path.endswith(".parquet"):
            continue
        local_path = hub.download(data_repo, path, local)
        table = pq.read_table(local_path, columns=["id", "source", "text"]).to_pydict()
        yield from zip(table["id"], table["source"], table["text"])
        Path(local_path).unlink()


def build_bands(cfg, band_dir):
    """Write band_<k>.txt files; return per-band docs/bytes/chars/lines and the train size."""
    s = cfg["sample"]
    fractions = all_fractions(cfg)
    local = Path(cfg["runtime"]["work_dir"], "tok_src")
    total = split_bytes(cfg, s["split"], local)
    band_dir = Path(band_dir)
    band_dir.mkdir(parents=True, exist_ok=True)
    need = fractions[-1] * total * 1.1
    free = shutil.disk_usage(band_dir).free
    print(f"[sample] train {total / 1e9:.2f} GB; bands up to {100 * fractions[-1]:.0f}% need "
          f"~{need / 1e9:.1f} GB disk, {free / 1e9:.1f} GB free", flush=True)
    if need > free:
        raise SystemExit(f"STOP: not enough disk for the tokenizer samples ({need / 1e9:.1f} GB needed)")
    paths = [band_dir / f"band_{k}.txt" for k in range(len(fractions))]
    bands = [{"fraction": f, "docs": 0, "bytes": 0, "chars": 0, "lines": 0} for f in fractions]
    files = [open(path, "w", encoding="utf-8") for path in paths]
    try:
        for doc_id, _, text in iter_split_docs(cfg, s["split"], local):
            k = band_of(keep_draw(doc_id, s["seed"]), fractions)
            if k is None:
                continue
            lines = doc_to_lines(text, s["max_line_chars"])
            files[k].write("".join(line + "\n" for line in lines))
            band = bands[k]
            band["docs"] += 1
            band["bytes"] += len(text.encode("utf-8"))
            band["chars"] += sum(len(line) for line in lines)
            band["lines"] += len(lines)
    finally:
        for f in files:
            f.close()
    return paths, {"train_bytes": total, "bands": bands}


def rung_input(paths, info, k):
    """Files and cumulative size of rung k (bands 0..k)."""
    bands = info["bands"][: k + 1]
    size = {key: sum(b[key] for b in bands) for key in ("docs", "bytes", "chars", "lines")}
    size["fraction"] = info["bands"][k]["fraction"]
    size["pct_of_train_bytes"] = round(100 * size["bytes"] / info["train_bytes"], 2)
    return [str(p) for p in paths[: k + 1]], size


def build_val_docs(cfg, out_path):
    """The val_docs val docs with the smallest id hash: a fixed random subset."""
    s = cfg["sample"]
    local = Path(cfg["runtime"]["work_dir"], "tok_src")
    docs = [(keep_draw(i, s["seed"]), i, src, t) for i, src, t in iter_split_docs(cfg, "val", local)]
    docs.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for _, doc_id, source, text in docs[: s["val_docs"]]:
            f.write(json.dumps({"id": doc_id, "source": source, "text": text}, ensure_ascii=False) + "\n")
    return {"val_docs_available": len(docs), "val_docs_used": min(len(docs), s["val_docs"])}


def check_final_complete(cfg):
    """STOP unless every shard of <final_prefix> was split by the current selection (one marker per stats file)."""
    repo = cfg["hf"]["dataset_repo"]
    shards = [p for p in hub.list_files(repo, f"{cfg['hf']['final_prefix']}stats/") if p.endswith(".json")]
    markers = hub.done_markers(repo, cfg["hf"]["final_markers_prefix"])
    if not shards or len(markers) < len(shards):
        raise SystemExit(f"STOP: {cfg['hf']['final_prefix']} is not fully split yet "
                         f"({len(markers)} of {len(shards)} shards); finish stage3b first")


def build(cfg):
    """Band files, val docs and their info. The info is also uploaded to <work_prefix> as a record."""
    check_final_complete(cfg)
    paths, info = build_bands(cfg, REPO_ROOT / cfg["sample"]["band_dir"])
    val_path = REPO_ROOT / cfg["sample"]["val_out_path"]
    info.update(build_val_docs(cfg, val_path))
    local = Path(cfg["runtime"]["work_dir"], "tok_cache")
    local.mkdir(parents=True, exist_ok=True)
    (local / "sample_info.json").write_text(json.dumps(info, indent=1))
    hub.upload_file(cfg["hf"]["dataset_repo"], local / "sample_info.json", f"{cfg['hf']['work_prefix']}sample_info.json")
    return paths, val_path, info


if __name__ == "__main__":
    print(build(load_config("configs/tokenizer.yaml"))[2])
