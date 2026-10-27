"""Pipeline CLI: runs each data stage over shards, with done-markers on the Hub.

    python -m data.pipeline stage1      # source inventory: samples, stats, previews
    python -m data.pipeline stage_lines # v2: count cross-doc boilerplate lines (on v1 shards)
    python -m data.pipeline stage2      # dry run of normalize + line cleaning + filters on the samples
    python -m data.pipeline stage3a     # stream, normalize, clean lines, filter, exact dedup, sign
    python -m data.pipeline stage3b     # near-dedup, quality ranking, split

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

from data import dedup_minhash, hub, lines, quality, sources, split, stats
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
    """Normalize + clean lines + filter the Stage 1 samples; write removal table and audits."""
    audit_cfg = cfg["audit"]
    audit = FilterAudit(audit_cfg["removed_per_stage_per_source"], audit_cfg["chars_shown"], audit_cfg["seed"])
    cleaner = lines.LineCleaner(cfg["lines"], lines.load_boilerplate(cfg, cfg["runtime"]["work_dir"]))
    removed = defaultdict(Counter)
    totals = Counter()
    reasons = Counter()
    line_chars = defaultdict(Counter)  # source -> {"total", "boilerplate", "pattern"} chars
    local = work_dir_of(cfg, "samples")
    for path in hub.list_files(repo_of(cfg), "raw/samples/"):
        source = Path(path).name.replace(".jsonl.gz", "")
        for doc in stats.read_jsonl_gz(hub.download(repo_of(cfg), path, local)):
            totals[source] += 1
            text = normalize(doc["text"])
            line_chars[source]["total"] += len(text)
            text, removed_lines = cleaner.clean(text)
            for why, line in removed_lines:
                line_chars[source][why] += len(line)
            if removed_lines:
                audit.record(f"lines_{removed_lines[0][0]}", source, doc["id"], "removed lines",
                             "\n".join(f"[{why}] {line}" for why, line in removed_lines))
            keep, reason = apply_filters(text, cfg["filters"])
            if not keep:
                removed[source][reason] += 1
                reasons[reason] += 1
                audit.record(reason, source, doc["id"], reason, text)

    out = out_dir_of(cfg)
    audit.write(out / "filter_audit")
    header = "| source | docs | " + " | ".join(FILTER_NAMES) + " | surviving |"
    rows = ["# Stage 2 dry run", "", "% of each source's sample removed by each filter (first failing filter).", "",
            header, "| --- | --- |" + " --- |" * (len(FILTER_NAMES) + 1)]
    for source in sorted(totals):
        n = totals[source]
        cells = [f"{100 * removed[source][f] / n:.1f}" for f in FILTER_NAMES]
        surviving = 100 * (n - sum(removed[source].values())) / n
        rows.append(f"| {source} | {n} | " + " | ".join(cells) + f" | {surviving:.1f} |")
    rows += ["", "Line cleaning: % of each source's characters removed before the filters.", "",
             "| source | boilerplate lines | pattern lines |", "| --- | --- | --- |"]
    for source in sorted(totals):
        c = line_chars[source]
        rows.append(f"| {source} | {100 * c['boilerplate'] / max(c['total'], 1):.2f} | "
                    f"{100 * c['pattern'] / max(c['total'], 1):.2f} |")
    rows += ["", "Top-5 removal reasons (all sources):", ""]
    rows += [f"{i + 1}. `{r}`: {c} docs" for i, (r, c) in enumerate(reasons.most_common(5))]
    report = "\n".join(rows)
    (out / "stage2_dryrun.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    check_removal_stop_rules(cfg, removed, totals)


def check_removal_stop_rules(cfg, removed, totals):
    """Instruction-file STOP rules: a filter set removing too much of a source halts the run."""
    rules = cfg["stop_rules"]
    for source, n in totals.items():
        share = sum(removed[source].values()) / n
        limit = rules["max_removed_wikipedia"] if source == "wikipedia" else rules["max_removed_any"]
        if share > limit:
            raise SystemExit(f"STOP: filters remove {100 * share:.1f}% of {source} (limit {100 * limit:.0f}%)")


# ---------------------------------------------------------------- stage 3a

_WORKER = {}


def _init_worker(cfg, boilerplate_hashes):
    """Pool initializer: each worker builds the MinHash permutations and the line cleaner once."""
    mh = cfg["dedup"]["minhash"]
    _WORKER["cfg"] = cfg
    _WORKER["perms"] = dedup_minhash.make_permutations(mh["num_perm"], mh["seed"])
    _WORKER["cleaner"] = lines.LineCleaner(cfg["lines"], boilerplate_hashes)


def process_batch(docs):
    """Worker: normalize, clean lines, filter, mask PII, hash and sign a batch of docs."""
    cfg = _WORKER["cfg"]
    ngram = cfg["dedup"]["minhash"]["ngram"]
    results = []
    for doc in docs:
        raw_bytes = len(doc["text"].encode("utf-8"))
        text = normalize(doc["text"])
        text, removed_lines = _WORKER["cleaner"].clean(text)
        keep, reason = apply_filters(text, cfg["filters"])
        result = {"id": doc["id"], "source": doc["source"], "raw_bytes": raw_bytes, "keep": keep, "reason": reason,
                  "lines_removed": Counter(why for why, _ in removed_lines)}
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
    shard_stats = {"shard": name, "raw_docs": 0, "raw_bytes": 0, "removed": Counter(), "lines_removed": Counter(),
                   "filtered_docs": 0, "filtered_bytes": 0, "exact_dups": 0,
                   "exact_dup_bytes": 0, "kept_docs": 0, "kept_bytes": 0}
    kept = []
    batches = _batched(docs, cfg["runtime"]["batch_docs"])
    for results in pool.imap(process_batch, batches):
        for r in results:
            shard_stats["raw_docs"] += 1
            shard_stats["raw_bytes"] += r["raw_bytes"]
            shard_stats["lines_removed"].update(r["lines_removed"])
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
    shard_stats["lines_removed"] = dict(shard_stats["lines_removed"])
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


def raw_tokens(finished, cfg):
    """Estimated tokens in everything read so far, before any cleaning."""
    return sum(st["raw_bytes"] for st in finished.values()) / cfg["target"]["bytes_per_token_estimate"]


def run_stage3a(cfg, max_hours):
    """Stream sources in priority order until the estimated train-token target is met.

    Returns "target" (done), "time" (session budget used; rerun resumes) or
    "exhausted" (every enabled source read and the target still not met).
    v2: every source is `use: all`, so this reads everything; "target" then means the
    raw collection reached target.collect_tokens_target.
    """
    started = time.time()
    repo = repo_of(cfg)
    prefix = cfg["hf"]["filtered_prefix"]
    suffix = cfg["pipeline"]["done_marker_suffix"]
    shard_docs = cfg["pipeline"]["shard_docs"]
    stop_at = cfg["target"]["train_tokens_target"] * cfg["target"]["stage3a_overshoot"]
    exact = ExactDedup()
    finished = load_finished_shards(cfg, exact)
    files_done = hub.done_markers(repo, f"{prefix}_files/", suffix)
    boilerplate = lines.load_boilerplate(cfg, cfg["runtime"]["work_dir"])

    with Pool(cfg["runtime"]["workers"], initializer=_init_worker, initargs=(cfg, boilerplate)) as pool:
        for source, src_cfg in sources.enabled_sources(cfg):
            for file_idx, path in enumerate(sources.list_source_files(src_cfg)):
                est = estimate_train_tokens(kept_bytes_by_source(finished), cfg)
                if src_cfg["use"] != "all" and est >= stop_at:
                    print(f"[3a] target reached: est {est / 1e9:.2f}B train tokens", flush=True)
                    return "target"
                if (time.time() - started) / 3600 > max_hours:
                    print("[3a] time budget used; rerun to continue", flush=True)
                    return "time"
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
                    hub.upload_folder_with_marker(repo, shard_dir, f"{prefix}{name}", f"{prefix}{name}{suffix}",
                                                  message=f"Filtered shard {name}")
                    for f in shard_dir.iterdir():
                        f.unlink()
                    shard_dir.rmdir()
                    finished[name] = shard_stats
                    est = estimate_train_tokens(kept_bytes_by_source(finished), cfg)
                    print(f"[3a] {name}: {shard_stats['raw_docs']} raw -> {shard_stats['kept_docs']} kept "
                          f"in {shard_stats['seconds']}s; raw {raw_tokens(finished, cfg) / 1e9:.2f}B tokens, "
                          f"est train tokens before ranking {est / 1e9:.2f}B", flush=True)
                # Whole input file finished: mark it so a rerun does not even download it.
                hub.write_marker(repo, file_marker)
    est = estimate_train_tokens(kept_bytes_by_source(finished), cfg)
    raw = raw_tokens(finished, cfg)
    print(f"[3a] all enabled sources read: raw {raw / 1e9:.2f}B tokens, est train tokens before ranking "
          f"{est / 1e9:.2f}B", flush=True)
    return "target" if est >= stop_at and raw >= cfg["target"]["collect_tokens_target"] else "exhausted"


# ---------------------------------------------------------------- stage 3b / 3c / 3d

def finished_shard_names(cfg):
    prefix = cfg["hf"]["filtered_prefix"]
    suffix = cfg["pipeline"]["done_marker_suffix"]
    markers = hub.done_markers(repo_of(cfg), prefix, suffix)
    return sorted(m[len(prefix):-len(suffix)] for m in markers if "/_files/" not in m)


def load_signatures_and_meta(cfg, names):
    """All shards' signatures in one preallocated array, plus numpy meta per doc.

    Meta is numpy, not Python lists: at ~17M docs, lists of ids/sources would cost GBs.
    meta = {"shard": int32 index into names, "row": int32, "source": int8 index into
    keep_priority, "n_bytes": int64}.
    """
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "near")
    priority = {s: r for r, s in enumerate(cfg["dedup"]["keep_priority"])}
    n_bytes = []
    for name in names:
        path = hub.download(repo, f"{prefix}{name}/meta.parquet", local)
        n_bytes.append(pq.read_table(path, columns=["n_bytes"]).column("n_bytes").to_numpy())
    sizes = [len(b) for b in n_bytes]
    meta = {
        "shard": np.repeat(np.arange(len(names), dtype=np.int32), sizes),
        "row": np.concatenate([np.arange(n, dtype=np.int32) for n in sizes]),
        "source": np.repeat(np.array([priority[quality.source_of(n)] for n in names], dtype=np.int8), sizes),
        "n_bytes": np.concatenate(n_bytes).astype(np.int64),
    }
    sigs = np.empty((sum(sizes), cfg["dedup"]["minhash"]["num_perm"]), dtype=np.uint32)
    start = 0
    for name, n in zip(names, sizes):
        path = hub.download(repo, f"{prefix}{name}/signatures.npy", local)
        sigs[start : start + n] = np.load(path)
        Path(path).unlink()
        start += n
    return sigs, meta


def read_doc_text(cfg, shard, row):
    """Text of one doc: reads just the parquet row group that holds it."""
    local = work_dir_of(cfg, "near")
    path = hub.download(repo_of(cfg), f"{cfg['hf']['filtered_prefix']}{shard}/docs.parquet", local)
    parquet = pq.ParquetFile(path)
    group = parquet.read_row_group(row // PARQUET_ROW_GROUP, columns=["text"])
    return group.column("text")[row % PARQUET_ROW_GROUP].as_py()


def write_pair_audit(cfg, names, meta, dup_i, dup_j, agreement, out_path):
    """20 random verified duplicate pairs, side by side."""
    rng = random.Random(cfg["dedup"]["minhash"]["seed"])
    picks = rng.sample(range(len(dup_i)), min(cfg["dedup"]["audit_pairs"], len(dup_i)))
    chars = cfg["audit"]["chars_shown"]
    out = ["# Near-duplicate pairs (random sample)", ""]
    for p in picks:
        a, b = int(dup_i[p]), int(dup_j[p])
        out += [f"## agreement {agreement[p]:.2f}", ""]
        for idx in (a, b):
            shard, row = names[meta["shard"][idx]], int(meta["row"][idx])
            text = read_doc_text(cfg, shard, row)[:chars]
            out += [f"**{shard} row {row}**", "", "```text", text, "```", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out), encoding="utf-8")


def drops_by_shard(names, shard_idx, rows):
    """{shard name: set of rows} from parallel index arrays."""
    drops = defaultdict(set)
    for s, r in zip(shard_idx.tolist(), rows.tolist()):
        drops[names[s]].add(r)
    return dict(drops)


def run_near_dedup(cfg, names):
    """Global MinHash LSH over all shards. Returns ({shard: dropped rows}, info)."""
    mh = cfg["dedup"]["minhash"]
    sigs, meta = load_signatures_and_meta(cfg, names)
    n_docs = len(meta["row"])
    keys_bytes = n_docs * mh["bands"] * 8
    avail = mem_available_bytes()
    print(f"[3b] {n_docs:,} docs; signatures {sigs.nbytes / 1e9:.2f} GB loaded; "
          f"available RAM {avail / 1e9:.1f} GB", flush=True)
    if 3 * keys_bytes > avail:  # band keys + one band's sort buffers + candidate pairs
        raise SystemExit(f"STOP: LSH needs ~{3 * keys_bytes / 1e9:.1f} GB more, only {avail / 1e9:.1f} GB available")

    keys = dedup_minhash.band_keys(sigs, mh["bands"], mh["rows_per_band"], mh["seed"])
    cand_i, cand_j = dedup_minhash.candidate_pairs(keys)
    del keys
    agreement = dedup_minhash.signature_agreement(sigs, cand_i, cand_j)
    del sigs
    is_dup = agreement >= mh["threshold"]
    dup_i, dup_j, dup_agree = cand_i[is_dup], cand_j[is_dup], agreement[is_dup]
    print(f"[3b] {len(cand_i):,} candidate pairs, {len(dup_i):,} verified duplicates", flush=True)

    drops, _, _ = dedup_minhash.choose_drops(n_docs, dup_i, dup_j, meta["source"].astype(np.int64), meta["n_bytes"])
    priority = cfg["dedup"]["keep_priority"]
    drop_stats = defaultdict(lambda: {"docs": 0, "bytes": 0})
    for i in drops.tolist():
        source = priority[meta["source"][i]]
        drop_stats[source]["docs"] += 1
        drop_stats[source]["bytes"] += int(meta["n_bytes"][i])
    write_pair_audit(cfg, names, meta, dup_i, dup_j, dup_agree, out_dir_of(cfg) / "filter_audit" / "near_dedup_pairs.md")

    kept_bytes = Counter()
    for r, source in enumerate(priority):
        mask = meta["source"] == r
        if mask.any():
            kept_bytes[source] = int(meta["n_bytes"][mask].sum()) - drop_stats.get(source, {"bytes": 0})["bytes"]
    info = {"docs_in": n_docs, "candidate_pairs": int(len(cand_i)), "duplicate_pairs": int(len(dup_i)),
            "dropped": {k: dict(v) for k, v in drop_stats.items()}, "kept_bytes": dict(kept_bytes)}
    return drops_by_shard(names, meta["shard"][drops], meta["row"][drops]), info


def load_shard_bytes(cfg, names):
    """{shard: n_bytes array}, row-aligned with docs.parquet."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "near")
    return {n: pq.read_table(hub.download(repo, f"{prefix}{n}/meta.parquet", local), columns=["n_bytes"])
            .column("n_bytes").to_numpy() for n in names}


def save_drop_list(cfg, drops, info, name):
    """Upload a (shard, row) drop list + its stats to <work_prefix><name>/ so a rerun can skip the step."""
    local = work_dir_of(cfg, "drop_lists", name)  # own folder: the quality step's scratch dir holds models and scores
    shards = [s for s, rows in sorted(drops.items()) for _ in rows]
    rows = [int(r) for _, rs in sorted(drops.items()) for r in sorted(rs)]
    pq.write_table(pa.table({"shard": shards, "row": pa.array(rows, type=pa.int32())}),
                   local / "drops.parquet", compression="zstd")
    (local / "info.json").write_text(json.dumps(info, indent=1))
    hub.upload_folder(repo_of(cfg), local, f"{cfg['hf']['work_prefix']}{name}", message=f"{name} drop list")


def load_drop_list(cfg, name):
    """({shard: set of rows}, info) if the step already ran, else None."""
    remote = f"{cfg['hf']['work_prefix']}{name}/"
    if f"{remote}info.json" not in hub.list_files(repo_of(cfg), remote):
        return None
    local = work_dir_of(cfg, "drop_lists", name)
    table = pq.read_table(hub.download(repo_of(cfg), f"{remote}drops.parquet", local)).to_pydict()
    drops = defaultdict(set)
    for shard, row in zip(table["shard"], table["row"]):
        drops[shard].add(row)
    info = json.loads(Path(hub.download(repo_of(cfg), f"{remote}info.json", local)).read_text())
    return dict(drops), info


def final_markers_dir(cfg):
    """Split done-markers, one set per selection method: a new method re-splits (overwrites) every shard."""
    method = cfg["quality"]["method"]
    return "final_markers/" if method == "fasttext" else f"final_markers_{method}/"


def split_shard(cfg, name, drop_rows):
    """Drop the given rows, split by id hash, upload <final_prefix>{train,val,test}/<shard>.parquet."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "split_src")
    table = pq.read_table(hub.download(repo, f"{prefix}{name}/docs.parquet", local),
                          columns=["id", "source", "text", "n_bytes"])
    rows = table.to_pydict()
    buckets = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    final_stats = defaultdict(lambda: defaultdict(lambda: {"docs": 0, "bytes": 0}))
    for i, doc_id in enumerate(rows["id"]):
        if i in drop_rows:
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
    hub.upload_folder_with_marker(repo, out, cfg["hf"]["final_prefix"].rstrip("/"),
                                  f"{cfg['hf']['work_prefix']}{final_markers_dir(cfg)}{name}{cfg['pipeline']['done_marker_suffix']}",
                                  message=f"Final split {name}")
    for path in sorted(out.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    Path(local, prefix, name, "docs.parquet").unlink(missing_ok=True)
    return final_stats


def build_funnel(cfg, names, near_info, quality_info):
    """Per-source funnel from shard stats, drop lists and final split stats."""
    repo, prefix = repo_of(cfg), cfg["hf"]["filtered_prefix"]
    local = work_dir_of(cfg, "funnel")
    funnel = defaultdict(stats.empty_funnel)
    lines_removed = defaultdict(Counter)
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
        lines_removed[source].update(st.get("lines_removed", {}))
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
    return funnel, split_totals, {s: dict(c) for s, c in lines_removed.items()}


def run_stage3b(cfg):
    """Near-dedup, quality ranking, then split every shard. Every step resumes from the Hub."""
    names = finished_shard_names(cfg)
    out = out_dir_of(cfg)
    t0 = time.time()

    cached = load_drop_list(cfg, "near_dedup")
    if cached is None:
        near_drops, near_info = run_near_dedup(cfg, names)
        save_drop_list(cfg, near_drops, near_info, "near_dedup")
    else:
        near_drops, near_info = cached
        print("[3b] near-dedup drop list found; skipping", flush=True)
    near_seconds = round(time.time() - t0, 1)

    t1 = time.time()
    est = estimate_train_tokens(near_info["kept_bytes"], cfg)
    trigger = cfg["target"]["train_tokens_target"] * cfg["target"]["quality_filter_trigger_multiple"]
    quality_drops, quality_info = {}, {"ran": False, "est_tokens_before": est, "trigger": trigger}
    if est > trigger:
        cached = load_drop_list(cfg, quality.drop_list_name(cfg))
        if cached is None:
            quality_drops, quality_info = quality.run_quality(cfg, names, near_drops, load_shard_bytes(cfg, names),
                                                              work_dir_of(cfg, "quality"), out)
            quality_info["est_tokens_before"] = est
            save_drop_list(cfg, quality_drops, quality_info, quality.drop_list_name(cfg))
        else:
            quality_drops, quality_info = cached
    print(f"[3c] est train tokens after near-dedup {est / 1e9:.2f}B; quality ranking "
          f"{'ran' if quality_info.get('ran') else 'skipped'} (trigger {trigger / 1e9:.2f}B)", flush=True)
    quality_seconds = round(time.time() - t1, 1)

    suffix = cfg["pipeline"]["done_marker_suffix"]
    done = {m.rsplit("/", 1)[1][: -len(suffix)]
            for m in hub.done_markers(repo_of(cfg), f"{cfg['hf']['work_prefix']}{final_markers_dir(cfg)}", suffix)}
    t2 = time.time()
    for name in names:
        if name in done:
            continue
        drop_rows = set(near_drops.get(name, ())) | set(int(r) for r in quality_drops.get(name, ()))
        split_shard(cfg, name, drop_rows)
        print(f"[3d] split {name}", flush=True)
    split_seconds = round(time.time() - t2, 1)

    funnel, split_totals, lines_removed = build_funnel(cfg, names, near_info, quality_info)
    write_stage3_report(cfg, funnel, split_totals, near_info, quality_info, lines_removed,
                        {"near_dedup_s": near_seconds, "quality_s": quality_seconds, "split_s": split_seconds}, out)


def write_stage3_report(cfg, funnel, split_totals, near_info, quality_info, lines_removed, timings, out):
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
        "## Lines removed by line cleaning (count per reason)", "", "```json", json.dumps(lines_removed, indent=1), "```", "",
        "## Quality ranking (3c)", "", "```json", json.dumps(quality_info, indent=1, default=str), "```", "",
        "## Timings (s)", "", "```json", json.dumps(timings), "```",
    ]
    report = "\n".join(lines)
    (out / "stage3_report.md").write_text(report, encoding="utf-8")
    (out / "stage3_funnel.json").write_text(json.dumps({"funnel": funnel, "split": split_totals,
                                                        "est_train_tokens": total}, indent=1))
    print(report, flush=True)
    if total < cfg["target"]["train_tokens_needed"]:
        raise SystemExit(f"STOP: estimated train tokens {total / 1e9:.2f}B < the D1 need of "
                         f"{cfg['target']['train_tokens_needed'] / 1e9:.1f}B; more raw data needed")


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["stage1", "stage_lines", "stage2", "stage3a", "stage3b"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-hours", type=float, default=11.0, help="stop starting new shards after this")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(f"[pipeline] {args.stage} config={args.config} hash={config_hash(args.config)}", flush=True)
    if args.stage == "stage1":
        run_stage1(cfg)
    elif args.stage == "stage_lines":
        lines.build_boilerplate(cfg, cfg["runtime"]["work_dir"], out_dir_of(cfg))
    elif args.stage == "stage2":
        run_stage2(cfg)
    elif args.stage == "stage3a":
        status = run_stage3a(cfg, args.max_hours)
        if status == "time":
            raise SystemExit("INCOMPLETE: stage3a used its time budget; run again to resume")
        if status == "exhausted":
            raise SystemExit("STOP: all enabled sources read and the token target is not met")
    elif args.stage == "stage3b":
        run_stage3b(cfg)


if __name__ == "__main__":
    main()
