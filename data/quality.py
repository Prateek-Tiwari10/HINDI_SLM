"""Optional quality classifier (Stage 3c): fastText trained from scratch on our own data.

Runs only if the survivors exceed 2x the token target.
- Positives: Wikipedia + manual docs. Negatives: a random web sample (Sangraha / FineWeb-2).
- Every web doc gets a P(positive) score. We keep the highest-scoring docs until the
  kept bytes reach ~1.5x the target.
- Wikipedia and manual docs are never dropped.
"""

import random
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from data import hub
from data.audit import FilterAudit

LABEL_POS = "__label__hq"
LABEL_NEG = "__label__web"


def _clean_for_fasttext(text):
    """fastText reads one example per line."""
    return " ".join(text.split())


def iter_shard_docs(cfg, names, skip_ids):
    """(id, source, text, n_bytes) for every doc that survived near-dedup."""
    repo, prefix = cfg["hf"]["dataset_repo"], cfg["hf"]["filtered_prefix"]
    local = Path(cfg["runtime"]["work_dir"], "quality")
    for name in names:
        path = hub.download(repo, f"{prefix}{name}/docs.parquet", local)
        table = pq.read_table(path, columns=["id", "source", "text", "n_bytes"]).to_pydict()
        for i, doc_id in enumerate(table["id"]):
            if doc_id not in skip_ids:
                yield doc_id, table["source"][i], table["text"][i], table["n_bytes"][i]
        Path(path).unlink()


def train_classifier(cfg, names, skip_ids):
    """Reservoir-sample positives and negatives, then train fastText."""
    import fasttext  # imported here: only this optional step needs it

    qcfg = cfg["quality"]
    rng = random.Random(qcfg["seed"])
    k = qcfg["docs_per_class"]
    samples = {LABEL_POS: [], LABEL_NEG: []}
    seen = {LABEL_POS: 0, LABEL_NEG: 0}
    for _, source, text, _ in iter_shard_docs(cfg, names, skip_ids):
        if source in qcfg["positive_sources"]:
            label = LABEL_POS
        elif source in qcfg["negative_sources"]:
            label = LABEL_NEG
        else:
            continue
        seen[label] += 1
        if len(samples[label]) < k:
            samples[label].append(text)
        elif (slot := rng.randrange(seen[label])) < k:
            samples[label][slot] = text

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        lines = [f"{label} {_clean_for_fasttext(t)}" for label, texts in samples.items() for t in texts]
        rng.shuffle(lines)
        f.write("\n".join(lines))
        train_path = f.name
    model = fasttext.train_supervised(train_path, epoch=5, wordNgrams=2, dim=64, seed=qcfg["seed"], thread=cfg["runtime"]["workers"])
    return model, {label: len(t) for label, t in samples.items()}


def score(model, text):
    """P(high quality) for one doc."""
    labels, probs = model.predict(_clean_for_fasttext(text), k=2)
    return dict(zip(labels, probs)).get(LABEL_POS, 0.0)


def run_quality(cfg, names, near_ids, est_tokens):
    """Returns (dropped ids, info). Keeps protected sources plus the best-scoring web docs."""
    qcfg = cfg["quality"]
    model, class_sizes = train_classifier(cfg, names, near_ids)

    protected_bytes = 0
    scored = []  # (score, id, source, n_bytes, text prefix)
    chars = cfg["audit"]["chars_shown"]
    for doc_id, source, text, n_bytes in iter_shard_docs(cfg, names, near_ids):
        if source in qcfg["never_drop"]:
            protected_bytes += n_bytes * cfg["sources"][source]["upsample_train"]
            continue
        scored.append((score(model, text), doc_id, source, n_bytes, text[:chars]))

    bpt = cfg["target"]["bytes_per_token_estimate"]
    keep_budget = cfg["target"]["train_tokens_target"] * cfg["target"]["quality_filter_keep_multiple"] * bpt / cfg["split"]["train"]
    scored.sort(key=lambda row: -row[0])
    budget_left = keep_budget - protected_bytes
    dropped, drop_stats = set(), defaultdict(lambda: {"docs": 0, "bytes": 0})
    audit = FilterAudit(cfg["audit"]["removed_per_stage_per_source"], chars, qcfg["seed"])
    for s, doc_id, source, n_bytes, prefix in scored:
        if budget_left > 0:
            budget_left -= n_bytes
            continue
        dropped.add(doc_id)
        drop_stats[source]["docs"] += 1
        drop_stats[source]["bytes"] += n_bytes
        audit.record("quality", source, doc_id, f"score={s:.3f}", prefix)
    audit.write(Path(cfg["runtime"]["out_dir"]) / "filter_audit")
    info = {"ran": True, "est_tokens_before": est_tokens, "class_sizes": class_sizes,
            "dropped": {k: dict(v) for k, v in drop_stats.items()}}
    return dropped, info
