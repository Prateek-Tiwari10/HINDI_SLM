"""Pipeline CLI: runs each data stage over shards, with done-markers on the Hub.

    python -m data.pipeline stage1      # source inventory: samples, stats, previews
    python -m data.pipeline stage2      # dry run of normalize + filters on the samples
    python -m data.pipeline stage3a     # stream, normalize, filter, exact dedup, sign
    python -m data.pipeline stage3b     # near-dedup, optional quality filter, split

Every long step processes shards. A shard's outputs are uploaded first, then its
done-marker. A rerun skips shards that have a marker, so a Kaggle session that
dies at 12 h loses at most the shard in progress.
"""

import argparse
import itertools
import json
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data import dedup_minhash, hub, quality, sources, split, stats
from data.audit import FilterAudit
from data.config import config_hash, load_config
from data.dedup_exact import ExactDedup, text_hash
from data.filters import FILTER_NAMES, apply_filters, pii_mask
from data.normalize import normalize

DEFAULT_CONFIG = "configs/data.yaml"
PARQUET_ROW_GROUP = 10_000
FINAL_SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("text", pa.string()), ("n_bytes", pa.int64())])


# ---------------------------------------------------------------- helpers

def repo_of(cfg):
    return cfg["hf"]["dataset_repo"]


def out_dir_of(cfg):
    path = Path(cfg["runtime"]["out_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_dir_of(cfg, *parts):
    path = Path(cfg["runtime"]["work_dir"], *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mem_available_bytes():
    """MemAvailable from /proc/meminfo (Linux, which Kaggle is)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


def estimate_train_tokens(kept_bytes_by_source, cfg):
    """Estimated train tokens: bytes x upsample x train fraction / bytes-per-token."""
    total = 0.0
    for source, n_bytes in kept_bytes_by_source.items():
        upsample = cfg["sources"][source]["upsample_train"]
        total += n_bytes * upsample * cfg["split"]["train"]
    return total / cfg["target"]["bytes_per_token_estimate"]


# ---------------------------------------------------------------- stage 1

def run_stage1(cfg):
    local = work_dir_of(cfg, "samples")
    rows = stats.run_stage1(cfg, out_dir_of(cfg), local)
    for row in rows:
        print(row, flush=True)


# ---------------------------------------------------------------- stage 2

def run_stage2(cfg):
    """Normalize + filter the Stage 1 samples; write removal table and audits."""
    audit_cfg = cfg["audit"]
    audit = FilterAudit(audit_cfg["removed_per_stage_per_source"], audit_cfg["chars_shown"], audit_cfg["seed"])
    removed = defaultdict(Counter)
    totals = Counter()
    reasons = Counter()
    local = work_dir_of(cfg, "samples")
    for path in hub.list_files(repo_of(cfg), "raw/samples/"):
        source = Path(path).name.replace(".jsonl.gz", "")
        for doc in stats.read_jsonl_gz(hub.download(repo_of(cfg), path, local)):
            totals[source] += 1
            text = normalize(doc["text"])
            keep, reason = apply_filters(text, cfg["filters"])
            if not keep:
                removed[source][reason] += 1
                reasons[reason] += 1
                audit.record(reason, source, doc["id"], reason, text)

    out = out_dir_of(cfg)
    audit.write(out / "filter_audit")
    header = "| source | docs | " + " | ".join(FILTER_NAMES) + " | surviving |"
    lines = ["# Stage 2 dry run", "", "% of each source's sample removed by each filter (first failing filter).", "",
             header, "| --- | --- |" + " --- |" * (len(FILTER_NAMES) + 1)]
    for source in sorted(totals):
        n = totals[source]
        cells = [f"{100 * removed[source][f] / n:.1f}" for f in FILTER_NAMES]
        surviving = 100 * (n - sum(removed[source].values())) / n
        lines.append(f"| {source} | {n} | " + " | ".join(cells) + f" | {surviving:.1f} |")
    lines += ["", "Top-5 removal reasons (all sources):", ""]
    lines += [f"{i + 1}. `{r}`: {c} docs" for i, (r, c) in enumerate(reasons.most_common(5))]
    report = "\n".join(lines)
    (out / "stage2_dryrun.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


# ---------------------------------------------------------------- stage 3a

_WORKER = {}


def _init_worker(cfg):
    """Pool initializer: each worker builds the MinHash permutations once."""
    mh = cfg["dedup"]["minhash"]
    _WORKER["cfg"] = cfg
    _WORKER["perms"] = dedup_minhash.make_permutations(mh["num_perm"], mh["seed"])


def process_batch(docs):
    """Worker: normalize, filter, mask PII, hash and sign a batch of docs."""
    cfg = _WORKER["cfg"]
    ngram = cfg["dedup"]["minhash"]["ngram"]
    results = []
    for doc in docs:
        raw_bytes = len(doc["text"].encode("utf-8"))
        text = normalize(doc["text"])
        keep, reason = apply_filters(text, cfg["filters"])
        result = {"id": doc["id"], "source": doc["source"], "raw_bytes": raw_bytes, "keep": keep, "reason": reason}
        if keep:
            text = pii_mask(text, cfg["filters"])
            result["text"] = text
            result["n_bytes"] = len(text.encode("utf-8"))
            result["xxhash"] = text_hash(text)
            result["sig"] = dedup_minhash.signature(text, ngram, _WORKER["perms"])
        results.append(result)
    return results


def shard_name(source, file_idx, chunk_idx):
    return f"{source}-{file_idx:04d}-{chunk_idx:02d}"


def _batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def process_shard(pool, docs, exact, cfg, name):
    """Run one shard through the workers, then exact-dedup in order in this process."""
    shard_stats = {"shard": name, "raw_docs": 0, "raw_bytes": 0, "removed": Counter(),
                   "filtered_docs": 0, "filtered_bytes": 0, "exact_dups": 0,
                   "exact_dup_bytes": 0, "kept_docs": 0, "kept_bytes": 0}
    kept = []
    batches = _batched(docs, cfg["runtime"]["batch_docs"])
    for results in pool.imap(process_batch, batches):
        for r in results:
            shard_stats["raw_docs"] += 1
            shard_stats["raw_bytes"] += r["raw_bytes"]
            if not r["keep"]:
                shard_stats["removed"][r["reason"]] += 1
                continue
            shard_stats["filtered_docs"] += 1
            shard_stats["filtered_bytes"] += r["n_bytes"]
            if not exact.is_new(r["xxhash"]):
                shard_stats["exact_dups"] += 1
                shard_stats["exact_dup_bytes"] += r["n_bytes"]
                continue
            shard_stats["kept_docs"] += 1
            shard_stats["kept_bytes"] += r["n_bytes"]
            kept.append(r)
    shard_stats["removed"] = dict(shard_stats["removed"])
    return kept, shard_stats


def write_shard(kept, shard_stats, shard_dir):
    """Survivors parquet + signatures .npy + hashes .npy + small meta parquet + stats.json."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    hashes = np.array([r["xxhash"] for r in kept], dtype=np.uint64)
    table = pa.table({
        "id": [r["id"] for r in kept],
        "source": [r["source"] for r in kept],
        "text": [r["text"] for r in kept],
        "xxhash": pa.array(hashes, type=pa.uint64()),
        "n_bytes": pa.array([r["n_bytes"] for r in kept], type=pa.int64()),
    })
    pq.write_table(table, shard_dir / "docs.parquet", compression="zstd", row_group_size=PARQUET_ROW_GROUP)
    pq.write_table(table.select(["id", "source", "n_bytes"]), shard_dir / "meta.parquet", compression="zstd")
    sigs = np.stack([r["sig"] for r in kept]) if kept else np.zeros((0, 128), dtype=np.uint32)
    np.save(shard_dir / "signatures.npy", sigs.astype(np.uint32))
    np.save(shard_dir / "hashes.npy", hashes)
    (shard_dir / "stats.json").write_text(json.dumps(shard_stats, indent=1))


def load_finished_shards(cfg, exact):
    """On (re)start: stats of finished shards, and their hashes into the exact-dedup set."""
    repo = repo_of(cfg)
    prefix = cfg["hf"]["filtered_prefix"]
    suffix = cfg["pipeline"]["done_marker_suffix"]
    markers = hub.done_markers(repo, prefix, suffix)
    done = sorted(m[len(prefix):-len(suffix)] for m in markers if "/_files/" not in m)
    local = work_dir_of(cfg, "resume")
    finished = {}
    for name in done:
        stats_path = hub.download(repo, f"{prefix}{name}/stats.json", local)
        finished[name] = json.loads(Path(stats_path).read_text())
        exact.load(np.load(hub.download(repo, f"{prefix}{name}/hashes.npy", local)))
    print(f"[3a] resuming: {len(finished)} shards done, {len(exact):,} hashes loaded", flush=True)
    return finished


def kept_bytes_by_source(finished):
    totals = Counter()
    for name, st in finished.items():
        totals[name.rsplit("-", 2)[0]] += st["kept_bytes"]
    return totals


def run_stage3a(cfg, max_hours):
    """Stream sources in priority order until the estimated train-token target is met."""
    started = time.time()
    repo = repo_of(cfg)
    prefix = cfg["hf"]["filtered_prefix"]
    suffix = cfg["pipeline"]["done_marker_suffix"]
    shard_docs = cfg["pipeline"]["shard_docs"]
    stop_at = cfg["target"]["train_tokens_target"] * cfg["target"]["stage3a_overshoot"]
    exact = ExactDedup()
    finished = load_finished_shards(cfg, exact)
    files_done = hub.done_markers(repo, f"{prefix}_files/", suffix)

    with Pool(cfg["runtime"]["workers"], initializer=_init_worker, initargs=(cfg,)) as pool:
        for source, src_cfg in sources.enabled_sources(cfg):
            for file_idx, path in enumerate(sources.list_source_files(src_cfg)):
                est = estimate_train_tokens(kept_bytes_by_source(finished), cfg)
                if src_cfg["use"] != "all" and est >= stop_at:
                    print(f"[3a] target reached: est {est / 1e9:.2f}B train tokens", flush=True)
                    return finished
                if (time.time() - started) / 3600 > max_hours:
                    print("[3a] time budget used; rerun to continue", flush=True)
                    return finished
                file_marker = f"{prefix}_files/{source}-{file_idx:04d}{suffix}"
                if file_marker in files_done:
                    continue
                prefix_names = [n for n in finished if n.startswith(f"{source}-{file_idx:04d}-")]
                docs_iter = sources.iter_file(source, src_cfg, path, local_dir=str(work_dir_of(cfg, "src")))
                for chunk_idx, chunk in enumerate(_batched(docs_iter, shard_docs)):
                    name = shard_name(source, file_idx, chunk_idx)
                    if name in prefix_names:
                        continue
                    t0 = time.time()
                    kept, shard_stats = process_shard(pool, chunk, exact, cfg, name)
                    shard_stats["seconds"] = round(time.time() - t0, 1)
                    shard_dir = work_dir_of(cfg, "shards") / name
                    write_shard(kept, shard_stats, shard_dir)
                    hub.upload_folder(repo, shard_dir, f"{prefix}{name}", message=f"Filtered shard {name}")
                    hub.write_marker(repo, f"{prefix}{name}{suffix}")
                    for f in shard_dir.iterdir():
                        f.unlink()
                    shard_dir.rmdir()
                    finished[name] = shard_stats
                    est = estimate_train_tokens(kept_bytes_by_source(finished), cfg)
                    print(f"[3a] {name}: {shard_stats['raw_docs']} raw -> {shard_stats['kept_docs']} kept "
                          f"in {shard_stats['seconds']}s; est train tokens {est / 1e9:.2f}B", flush=True)
                # Whole input file finished: mark it so a rerun does not even download it.
                hub.write_marker(repo, file_marker)
    print("[3a] all enabled sources exhausted", flush=True)
    return finished


# ---------------------------------------------------------------- stage 3b / 3c / 3d

def finished_shard_names(cfg):
    prefix = cfg["hf"]["filtered_prefix"]
    suffix = cfg["pipeline"]["done_marker_suffix"]
    markers = hub.done_markers(repo_of(cfg), prefix, suffix)
    return sorted(m[len(prefix):-len(suffix)] for m in markers if "/_files/" not in m)


def load_signatures_and_meta(cfg, names):
    """Concatenate all shards' signatures and (id, source, n_bytes, shard, row)."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "near")
    sig_parts, metas = [], []
    for name in names:
        sig_parts.append(np.load(hub.download(repo, f"{prefix}{name}/signatures.npy", local)))
        meta = pq.read_table(hub.download(repo, f"{prefix}{name}/meta.parquet", local)).to_pydict()
        meta["shard"] = [name] * len(meta["id"])
        meta["row"] = list(range(len(meta["id"])))
        metas.append(meta)
    merged = {k: list(itertools.chain.from_iterable(m[k] for m in metas)) for k in metas[0]}
    return np.concatenate(sig_parts), merged


def read_doc_text(cfg, shard, row):
    """Text of one doc: reads just the parquet row group that holds it."""
    local = work_dir_of(cfg, "near")
    path = hub.download(repo_of(cfg), f"{cfg['hf']['filtered_prefix']}{shard}/docs.parquet", local)
    parquet = pq.ParquetFile(path)
    group = parquet.read_row_group(row // PARQUET_ROW_GROUP, columns=["text"])
    return group.column("text")[row % PARQUET_ROW_GROUP].as_py()


def write_pair_audit(cfg, meta, dup_i, dup_j, agreement, out_path):
    """20 random verified duplicate pairs, side by side."""
    rng = random.Random(cfg["dedup"]["minhash"]["seed"])
    picks = rng.sample(range(len(dup_i)), min(cfg["dedup"]["audit_pairs"], len(dup_i)))
    chars = cfg["audit"]["chars_shown"]
    lines = ["# Near-duplicate pairs (random sample)", ""]
    for p in picks:
        a, b = int(dup_i[p]), int(dup_j[p])
        lines += [f"## agreement {agreement[p]:.2f}", ""]
        for idx in (a, b):
            text = read_doc_text(cfg, meta["shard"][idx], meta["row"][idx])[:chars]
            lines += [f"**{meta['id'][idx]}**", "", "```text", text, "```", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_near_dedup(cfg, names):
    """Global MinHash LSH over all shards. Returns the set of dropped ids and per-source stats."""
    mh = cfg["dedup"]["minhash"]
    sigs, meta = load_signatures_and_meta(cfg, names)
    n_docs = len(meta["id"])
    need = n_docs * mh["bytes_per_doc_budget"]
    avail = mem_available_bytes()
    print(f"[3b] {n_docs:,} docs; signatures {need / 1e9:.2f} GB; available RAM {avail / 1e9:.1f} GB", flush=True)
    if need * 4 > avail:  # signatures + keys + sort buffers must fit comfortably
        raise SystemExit(f"STOP: near-dedup needs ~{4 * need / 1e9:.1f} GB, only {avail / 1e9:.1f} GB available")

    keys = dedup_minhash.band_keys(sigs, mh["bands"], mh["rows_per_band"], mh["seed"])
    cand_i, cand_j = dedup_minhash.candidate_pairs(keys)
    del keys
    agreement = dedup_minhash.signature_agreement(sigs, cand_i, cand_j)
    is_dup = agreement >= mh["threshold"]
    dup_i, dup_j, dup_agree = cand_i[is_dup], cand_j[is_dup], agreement[is_dup]
    print(f"[3b] {len(cand_i):,} candidate pairs, {len(dup_i):,} verified duplicates", flush=True)

    priority = {s: r for r, s in enumerate(cfg["dedup"]["keep_priority"])}
    source_rank = np.array([priority[s] for s in meta["source"]], dtype=np.int64)
    n_bytes = np.array(meta["n_bytes"], dtype=np.int64)
    drops, _, _ = dedup_minhash.choose_drops(n_docs, dup_i, dup_j, source_rank, n_bytes)

    dropped_ids = {meta["id"][i] for i in drops.tolist()}
    drop_stats = defaultdict(lambda: {"docs": 0, "bytes": 0})
    for i in drops.tolist():
        drop_stats[meta["source"][i]]["docs"] += 1
        drop_stats[meta["source"][i]]["bytes"] += int(n_bytes[i])
    write_pair_audit(cfg, meta, dup_i, dup_j, dup_agree, out_dir_of(cfg) / "filter_audit" / "near_dedup_pairs.md")

    kept_bytes = Counter()
    for source, b in zip(meta["source"], meta["n_bytes"]):
        kept_bytes[source] += b
    for source, d in drop_stats.items():
        kept_bytes[source] -= d["bytes"]
    info = {"docs_in": n_docs, "candidate_pairs": int(len(cand_i)), "duplicate_pairs": int(len(dup_i)),
            "dropped": {k: dict(v) for k, v in drop_stats.items()}, "kept_bytes": dict(kept_bytes)}
    return dropped_ids, info


def save_drop_list(cfg, dropped_ids, info, name):
    """Upload a drop list + its stats to work/<name>/ so a rerun can skip the step."""
    local = work_dir_of(cfg, name)
    pq.write_table(pa.table({"id": sorted(dropped_ids)}), local / "drops.parquet", compression="zstd")
    (local / "info.json").write_text(json.dumps(info, indent=1))
    hub.upload_folder(repo_of(cfg), local, f"work/{name}", message=f"{name} drop list")


def load_drop_list(cfg, name):
    """(dropped_ids, info) if the step already ran, else None."""
    files = hub.list_files(repo_of(cfg), f"work/{name}/")
    if f"work/{name}/info.json" not in files:
        return None
    local = work_dir_of(cfg, name)
    ids = pq.read_table(hub.download(repo_of(cfg), f"work/{name}/drops.parquet", local)).column("id").to_pylist()
    info = json.loads(Path(hub.download(repo_of(cfg), f"work/{name}/info.json", local)).read_text())
    return set(ids), info


def split_shard(cfg, name, drops):
    """Apply drop lists, split by id hash, upload final/{train,val,test}/<shard>.parquet."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "split_src")
    table = pq.read_table(hub.download(repo, f"{prefix}{name}/docs.parquet", local),
                          columns=["id", "source", "text", "n_bytes"])
    rows = table.to_pydict()
    buckets = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    final_stats = defaultdict(lambda: defaultdict(lambda: {"docs": 0, "bytes": 0}))
    for i, doc_id in enumerate(rows["id"]):
        if doc_id in drops:
            continue
        part = split.assign_split(doc_id, cfg["split"])
        for key in ("id", "source", "text", "n_bytes"):
            buckets[part][key].append(rows[key][i])
        final_stats[part][rows["source"][i]]["docs"] += 1
        final_stats[part][rows["source"][i]]["bytes"] += rows["n_bytes"][i]

    out = work_dir_of(cfg, "split_out", name)
    for part, cols in buckets.items():
        (out / part).mkdir(exist_ok=True)
        columns = {key: cols.get(key, []) for key in FINAL_SCHEMA.names}
        pq.write_table(pa.table(columns, schema=FINAL_SCHEMA), out / part / f"{name}.parquet", compression="zstd", row_group_size=PARQUET_ROW_GROUP)
    (out / "stats").mkdir(exist_ok=True)
    (out / "stats" / f"{name}.json").write_text(json.dumps(final_stats, indent=1))
    hub.upload_folder(repo, out, cfg["hf"]["final_prefix"].rstrip("/"), message=f"Final split {name}")
    hub.write_marker(repo, f"work/final_markers/{name}{cfg['pipeline']['done_marker_suffix']}")
    for path in sorted(out.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    Path(local, prefix, name, "docs.parquet").unlink(missing_ok=True)
    return final_stats


def build_funnel(cfg, names, near_info, quality_info):
    """Per-source funnel from shard stats, drop lists and final split stats."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "funnel")
    funnel = defaultdict(stats.empty_funnel)
    for name in names:
        source = name.rsplit("-", 2)[0]
        st = json.loads(Path(hub.download(repo, f"{prefix}{name}/stats.json", local)).read_text())
        f = funnel[source]
        f["raw"]["docs"] += st["raw_docs"]
        f["raw"]["bytes"] += st["raw_bytes"]
        f["after_filters"]["docs"] += st["filtered_docs"]
        f["after_filters"]["bytes"] += st["filtered_bytes"]
        f["after_exact_dedup"]["docs"] += st["kept_docs"]
        f["after_exact_dedup"]["bytes"] += st["kept_bytes"]
    for source, f in funnel.items():
        near = near_info["dropped"].get(source, {"docs": 0, "bytes": 0})
        f["after_near_dedup"]["docs"] = f["after_exact_dedup"]["docs"] - near["docs"]
        f["after_near_dedup"]["bytes"] = f["after_exact_dedup"]["bytes"] - near["bytes"]
        qual = quality_info.get("dropped", {}).get(source, {"docs": 0, "bytes": 0})
        f["after_quality"]["docs"] = f["after_near_dedup"]["docs"] - qual["docs"]
        f["after_quality"]["bytes"] = f["after_near_dedup"]["bytes"] - qual["bytes"]
    split_totals = defaultdict(lambda: defaultdict(lambda: {"docs": 0, "bytes": 0}))
    for path in hub.list_files(repo, f"{cfg['hf']['final_prefix']}stats/"):
        part_stats = json.loads(Path(hub.download(repo, path, local)).read_text())
        for part, by_source in part_stats.items():
            for source, v in by_source.items():
                split_totals[part][source]["docs"] += v["docs"]
                split_totals[part][source]["bytes"] += v["bytes"]
                funnel[source]["final"]["docs"] += v["docs"]
                funnel[source]["final"]["bytes"] += v["bytes"]
    return funnel, split_totals


def run_stage3b(cfg):
    """Near-dedup, conditional quality filter, then split every shard."""
    names = finished_shard_names(cfg)
    out = out_dir_of(cfg)
    t0 = time.time()

    cached = load_drop_list(cfg, "near_dedup")
    if cached is None:
        near_ids, near_info = run_near_dedup(cfg, names)
        save_drop_list(cfg, near_ids, near_info, "near_dedup")
    else:
        near_ids, near_info = cached
        print("[3b] near-dedup drop list found; skipping", flush=True)
    near_seconds = round(time.time() - t0, 1)

    est = estimate_train_tokens(near_info["kept_bytes"], cfg)
    trigger = cfg["target"]["train_tokens_target"] * cfg["target"]["quality_filter_trigger_multiple"]
    quality_ids, quality_info = set(), {"ran": False, "est_tokens_before": est, "trigger": trigger}
    if est > trigger:
        cached = load_drop_list(cfg, "quality")
        if cached is None:
            quality_ids, quality_info = quality.run_quality(cfg, names, near_ids, est)
            save_drop_list(cfg, quality_ids, quality_info, "quality")
        else:
            quality_ids, quality_info = cached
    print(f"[3c] est train tokens after near-dedup {est / 1e9:.2f}B; quality filter "
          f"{'ran' if quality_info.get('ran') else 'skipped'} (trigger {trigger / 1e9:.2f}B)", flush=True)

    drops = near_ids | quality_ids
    done = {m.rsplit("/", 1)[1][: -len(cfg["pipeline"]["done_marker_suffix"])]
            for m in hub.done_markers(repo_of(cfg), "work/final_markers/", cfg["pipeline"]["done_marker_suffix"])}
    t1 = time.time()
    for name in names:
        if name in done:
            continue
        split_shard(cfg, name, drops)
        print(f"[3d] split {name}", flush=True)
    split_seconds = round(time.time() - t1, 1)

    funnel, split_totals = build_funnel(cfg, names, near_info, quality_info)
    write_stage3_report(cfg, funnel, split_totals, near_info, quality_info,
                        {"near_dedup_s": near_seconds, "split_s": split_seconds}, out)


def write_stage3_report(cfg, funnel, split_totals, near_info, quality_info, timings, out):
    bpt = cfg["target"]["bytes_per_token_estimate"]
    train = split_totals["train"]
    wiki_bytes = train.get("wikipedia", {"bytes": 0})["bytes"]
    other_bytes = sum(v["bytes"] for s, v in train.items() if s != "wikipedia")
    tokens_1x = (wiki_bytes + other_bytes) / bpt
    tokens_wiki_extra = wiki_bytes * (cfg["sources"]["wikipedia"]["upsample_train"] - 1) / bpt
    total = tokens_1x + tokens_wiki_extra
    lines = [
        "# Stage 3 report", "",
        "## Funnel", "", stats.funnel_markdown(funnel), "",
        "## Split (docs / GB)", "",
        "| split | " + " | ".join(sorted(funnel)) + " |",
        "| --- |" + " --- |" * len(funnel),
    ]
    for part in ("train", "val", "test"):
        cells = [f"{split_totals[part][s]['docs']:,} / {split_totals[part][s]['bytes'] / 1e9:.3f}" for s in sorted(funnel)]
        lines.append(f"| {part} | " + " | ".join(cells) + " |")
    lines += [
        "", "## Estimated train tokens (bytes / %.1f)" % bpt, "",
        f"- train text once: {tokens_1x / 1e9:.3f}B",
        f"- Wikipedia 2x upsample adds: {tokens_wiki_extra / 1e9:.3f}B",
        f"- **total: {total / 1e9:.3f}B** (target {cfg['target']['train_tokens_target'] / 1e9:.1f}B)",
        "",
        "## Near-dedup", "", "```json", json.dumps(near_info, indent=1), "```", "",
        "## Quality filter (3c)", "", "```json", json.dumps(quality_info, indent=1, default=str), "```", "",
        "## Timings (s)", "", "```json", json.dumps(timings), "```",
    ]
    report = "\n".join(lines)
    (out / "stage3_report.md").write_text(report, encoding="utf-8")
    (out / "stage3_funnel.json").write_text(json.dumps({"funnel": funnel, "split": split_totals,
                                                        "est_train_tokens": total}, indent=1))
    print(report, flush=True)
    if total < cfg["target"]["train_tokens_target"]:
        print(f"STOP: estimated train tokens {total / 1e9:.2f}B < target", flush=True)


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["stage1", "stage2", "stage3a", "stage3b"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-hours", type=float, default=11.0, help="stop starting new shards after this")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"[pipeline] {args.stage} config={args.config} hash={config_hash(args.config)}", flush=True)
    if args.stage == "stage1":
        run_stage1(cfg)
    elif args.stage == "stage2":
        run_stage2(cfg)
    elif args.stage == "stage3a":
        run_stage3a(cfg, args.max_hours)
    elif args.stage == "stage3b":
        run_stage3b(cfg)


if __name__ == "__main__":
    main()
