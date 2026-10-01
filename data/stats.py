"""Stage 1 source inventory (samples, stats, previews) and the Stage 3 funnel table."""

import csv
import gzip
import json
import random
import statistics
from pathlib import Path

from data import hub, sources
from data.audit import Reservoir
from data.filters import devanagari_ratio


# ---------------------------------------------------------------- Stage 1: samples

def sample_source(name, src_cfg, sample_cfg):
    """Reservoir-sample docs_per_source docs from the first pool_docs docs of random files."""
    rng = random.Random(f"{sample_cfg['seed']}:{name}")
    files = sources.list_source_files(src_cfg)
    chosen = rng.sample(files, min(sample_cfg["pool_files"], len(files)))
    reservoir = Reservoir(sample_cfg["docs_per_source"], rng)
    for path in chosen:
        for doc in sources.iter_file(name, src_cfg, path, stream=True):
            reservoir.add(doc)
            if reservoir.seen >= sample_cfg["pool_docs"]:
                return reservoir.items
    return reservoir.items


def write_jsonl_gz(docs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def estimate_total_docs(name, src_cfg):
    """Total docs in the full source: parquet footers, or a full count for jsonl.zst."""
    files = sources.list_source_files(src_cfg)
    if src_cfg["format"] == "parquet":
        return sum(sources.count_parquet_rows(src_cfg, f) for f in files)
    return sum(1 for f in files for _ in sources.iter_file(name, src_cfg, f, stream=True))


def sample_stats(docs):
    """Per-sample numbers for stage1_source_stats.csv."""
    chars = [len(d["text"]) for d in docs]
    n_bytes = [len(d["text"].encode("utf-8")) for d in docs]
    ratios = [devanagari_ratio(d["text"]) for d in docs]
    return {
        "docs": len(docs),
        "mean_chars": round(statistics.mean(chars), 1),
        "median_chars": statistics.median(chars),
        "mean_bytes": round(statistics.mean(n_bytes), 1),
        "pct_devanagari_letters": round(100 * statistics.mean(ratios), 2),
    }


def run_stage1(data_cfg, out_dir, local_sample_dir):
    """Sample every source (FineWeb-2 too, for the decision), write stats + previews."""
    sample_cfg = data_cfg["samples"]
    bytes_per_token = data_cfg["target"]["bytes_per_token_estimate"]
    rows, previews = [], []
    all_sources = [(n, c) for n, c in data_cfg["sources"].items() if n != "excluded"]
    for name, src_cfg in all_sources:
        print(f"[stage1] sampling {name}", flush=True)
        docs = sample_source(name, src_cfg, sample_cfg)
        sample_path = Path(local_sample_dir) / f"{name}.jsonl.gz"
        write_jsonl_gz(docs, sample_path)
        hub.upload_file(data_cfg["hf"]["dataset_repo"], sample_path, f"raw/samples/{name}.jsonl.gz")

        row = {"source": name, **sample_stats(docs)}
        total_docs = estimate_total_docs(name, src_cfg)
        row["est_total_docs"] = total_docs
        row["est_total_gb"] = round(total_docs * row["mean_bytes"] / 1e9, 2)
        row["est_total_tokens_b"] = round(total_docs * row["mean_bytes"] / bytes_per_token / 1e9, 2)
        rows.append(row)

        rng = random.Random(f"preview:{name}")
        for doc in rng.sample(docs, sample_cfg["preview_docs"]):
            previews.append((name, doc["id"], doc["text"][: sample_cfg["preview_chars"]]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage1_source_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Stage 1 previews", ""]
    for name, doc_id, text in previews:
        lines += [f"## {name} — {doc_id}", "", "```text", text, "```", ""]
    (out_dir / "stage1_previews.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- Stage 3: funnel

FUNNEL_STEPS = ["raw", "after_filters", "after_exact_dedup", "after_near_dedup", "after_quality", "final"]


def empty_funnel():
    return {step: {"docs": 0, "bytes": 0} for step in FUNNEL_STEPS}


def funnel_markdown(funnel_by_source):
    """Markdown table: rows = source, columns = docs / GB at each funnel step."""
    header = "| source | " + " | ".join(FUNNEL_STEPS) + " |"
    sep = "| --- |" + " --- |" * len(FUNNEL_STEPS)
    lines = [header, sep]
    for source, funnel in funnel_by_source.items():
        cells = [f"{funnel[s]['docs']:,} docs / {funnel[s]['bytes'] / 1e9:.2f} GB" for s in FUNNEL_STEPS]
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
