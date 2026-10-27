"""Line-level cleaning (v2). Runs after normalize() and before the doc filters.

Two kinds of lines are removed:
1. Cross-document boilerplate: a line (<= max_chars) that appears in >= min_docs
   different documents. Nav bars, share widgets, bylines and paywall stubs repeat
   across thousands of pages; real sentences almost never do. The frequencies are
   counted once (`stage_lines`) on the v1 filtered shards, skipping v1 near-duplicates,
   so a syndicated article's lines do not count as boilerplate.
2. Short lines matching a pattern list ("यह भी पढ़ें", "फोटो:", "ऐप डाउनलोड करें", ...).

Lines are compared after stripping, so indentation differences do not matter.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import xxhash

from data import hub

_MANY_NEWLINES = re.compile(r"\n{3,}")


def line_hash(line):
    return xxhash.xxh64_intdigest(line.encode("utf-8"))


def compile_patterns(lines_cfg):
    """Regexes from the config. NFC-normalized, since documents are NFC (nukta letters differ otherwise)."""
    return [re.compile(unicodedata.normalize("NFC", p), re.IGNORECASE) for p in lines_cfg["drop_patterns"]]


class LineCleaner:
    """Removes boilerplate lines from one normalized document."""

    def __init__(self, lines_cfg, boilerplate_hashes=()):
        self.max_chars = lines_cfg["max_chars"]
        self.pattern_max_chars = lines_cfg["pattern_max_chars"]
        self.patterns = compile_patterns(lines_cfg)
        self.boilerplate = set(int(h) for h in boilerplate_hashes)

    def reason(self, line):
        """'boilerplate', 'pattern' or '' for one stripped, non-empty line."""
        if len(line) <= self.max_chars and self.boilerplate and line_hash(line) in self.boilerplate:
            return "boilerplate"
        if len(line) <= self.pattern_max_chars and any(p.search(line) for p in self.patterns):
            return "pattern"
        return ""

    def clean(self, text):
        """(cleaned text, removed lines as [(reason, line)])."""
        kept, removed = [], []
        for line in text.split("\n"):
            stripped = line.strip()
            why = self.reason(stripped) if stripped else ""
            if why:
                removed.append((why, stripped))
            else:
                kept.append(line)
        if not removed:
            return text, removed
        return _MANY_NEWLINES.sub("\n\n", "\n".join(kept)).strip(), removed


# ---------------------------------------------------------------- counting (stage_lines)

def doc_line_hashes(text, max_chars):
    """Distinct hashes of the short, non-empty lines of one doc."""
    return {line_hash(s) for s in (line.strip() for line in text.split("\n")) if s and len(s) <= max_chars}


def count_line_doc_frequency(cfg, local):
    """Doc frequency of every short line over the v1 filtered shards, minus v1 near-duplicates.

    Returns (unique hashes, doc counts, names of the shards read).
    """
    lcfg = cfg["lines"]
    repo = cfg["hf"]["dataset_repo"]
    prefix = lcfg["count_prefix"]
    near = pq.read_table(hub.download(repo, lcfg["count_near_drops"], local)).column("id").to_pylist()
    near = set(near)
    suffix = cfg["pipeline"]["done_marker_suffix"]
    names = sorted(m[len(prefix):-len(suffix)] for m in hub.done_markers(repo, prefix, suffix) if "/_files/" not in m)
    parts = []
    for name in names:
        path = hub.download(repo, f"{prefix}{name}/docs.parquet", local)
        table = pq.read_table(path, columns=["id", "text"]).to_pydict()
        Path(path).unlink()
        hashes = []
        for doc_id, text in zip(table["id"], table["text"]):
            if doc_id not in near:
                hashes.extend(doc_line_hashes(text, lcfg["max_chars"]))
        parts.append(np.array(hashes, dtype=np.uint64))
        print(f"[lines] {name}: {len(table['id']):,} docs, {len(hashes):,} short lines", flush=True)
    unique, counts = np.unique(np.concatenate(parts), return_counts=True)
    return unique, counts, names


def line_texts_for(cfg, names, wanted, local, limit_shards):
    """hash -> line text for the wanted hashes, read from a few shards (for the report only)."""
    lcfg = cfg["lines"]
    repo = cfg["hf"]["dataset_repo"]
    found = {}
    by_source = {}
    for name in names:
        by_source.setdefault(name.rsplit("-", 2)[0], []).append(name)
    picks = [n for shard_names in by_source.values() for n in shard_names[:limit_shards]]
    for name in picks:
        path = hub.download(repo, f"{lcfg['count_prefix']}{name}/docs.parquet", local)
        for text in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            for s in (line.strip() for line in text.split("\n")):
                if s and len(s) <= lcfg["max_chars"]:
                    h = line_hash(s)
                    if h in wanted and h not in found:
                        found[h] = s
        Path(path).unlink()
    return found


def build_boilerplate(cfg, work_dir, out_dir):
    """Count, threshold, upload the hash list, and write a report of the most common lines."""
    lcfg = cfg["lines"]
    local = Path(work_dir, "lines")
    local.mkdir(parents=True, exist_ok=True)
    unique, counts, names = count_line_doc_frequency(cfg, local)
    flagged = counts >= lcfg["min_docs"]
    hashes, flagged_counts = unique[flagged], counts[flagged]
    order = np.argsort(-flagged_counts, kind="stable")
    top = [(int(hashes[i]), int(flagged_counts[i])) for i in order[: lcfg["report_top"]]]
    rng = np.random.default_rng(cfg["audit"]["seed"])
    random_idx = rng.choice(len(hashes), size=min(100, len(hashes)), replace=False) if len(hashes) else []
    sample = [(int(hashes[i]), int(flagged_counts[i])) for i in random_idx]
    texts = line_texts_for(cfg, names, {h for h, _ in top + sample}, local, limit_shards=3)

    info = {"distinct_short_lines": int(len(unique)), "flagged_lines": int(flagged.sum()),
            "min_docs": lcfg["min_docs"], "max_chars": lcfg["max_chars"], "shards_counted": len(names),
            "line_occurrences_flagged": int(flagged_counts.sum()),
            "line_occurrences_total": int(counts.sum())}
    out = Path(work_dir, "lines_out")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "boilerplate_hashes.npy", np.sort(hashes))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    hub.upload_folder(cfg["hf"]["dataset_repo"], out, f"{cfg['hf']['work_prefix']}lines", message="Boilerplate line hashes")

    def rows(items):
        return [f"| {c:,} | `{texts.get(h, '(not in the shards read for the report)')[:150]}` |" for h, c in items]

    report = ["# Boilerplate lines (v2 line cleaning)", "", "```json", json.dumps(info, indent=1), "```", "",
              f"## Top {len(top)} by document frequency", "", "| docs | line |", "| --- | --- |", *rows(top), "",
              f"## {len(sample)} random flagged lines", "", "| docs | line |", "| --- | --- |", *rows(sample), ""]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "boilerplate_lines.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(info), flush=True)
    return info


def load_boilerplate(cfg, work_dir):
    """The flagged line hashes from the Hub. STOP if stage_lines has not run yet."""
    remote = f"{cfg['hf']['work_prefix']}lines/boilerplate_hashes.npy"
    if remote not in hub.list_files(cfg["hf"]["dataset_repo"], f"{cfg['hf']['work_prefix']}lines/"):
        raise SystemExit("STOP: run `python -m data.pipeline stage_lines` first")
    return np.load(hub.download(cfg["hf"]["dataset_repo"], remote, Path(work_dir, "lines")))
