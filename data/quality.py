"""Quality ranking (Stage 3c, v2): a fastText classifier trained from scratch on our own data.

- Positives: Wikipedia + manual news docs. Negatives: random web docs (Sangraha / FineWeb-2).
  So the score is "reads like an encyclopedia or edited news", the same idea as the
  Wikipedia-reference classifiers of GPT-3 and LLaMA.
- Every doc that survived near-dedup gets P(positive). Web docs are ranked by it and
  the best are kept until the estimated train tokens reach target.train_tokens_target.
- Wikipedia and manual docs are never dropped.

Built for ~17M docs on a 4-core Kaggle machine:
- the model is uploaded once, so a resumed session scores with the same model;
- scores are one float32 array per shard (row-aligned), uploaded in batches of shards,
  so a session that dies loses at most one batch;
- workers read their own parquet row groups, so texts are never pickled between processes.
"""

import json
import random
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data import hub
from data.audit import FilterAudit, Reservoir

LABEL_POS = "__label__hq"
LABEL_NEG = "__label__web"
SCORE_UPLOAD_BATCH = 10  # shards per Hub commit (HF rate-limits commits per hour)


def _clean_for_fasttext(text):
    """fastText reads one example per line."""
    return " ".join(text.split())


def _prefix(cfg, *parts):
    return cfg["hf"]["work_prefix"] + "quality/" + "".join(parts)


def source_of(shard_name):
    return shard_name.rsplit("-", 2)[0]


def download_docs(cfg, name, local):
    return hub.download(cfg["hf"]["dataset_repo"], f"{cfg['hf']['filtered_prefix']}{name}/docs.parquet", local)


# ---------------------------------------------------------------- training

def _sample_training_docs(cfg, names, near_drops, local):
    """Reservoir samples of positive and negative texts (near-duplicates excluded)."""
    qcfg = cfg["quality"]
    rng = random.Random(qcfg["seed"])
    pos_names = [n for n in names if source_of(n) in qcfg["positive_sources"]]
    neg_pool = [n for n in names if source_of(n) in qcfg["negative_sources"]]
    neg_names = sorted(rng.sample(neg_pool, min(qcfg["negative_shards"], len(neg_pool))))
    samples = {LABEL_POS: Reservoir(qcfg["docs_per_class"], rng), LABEL_NEG: Reservoir(qcfg["docs_per_class"], rng)}
    for label, shard_names in ((LABEL_POS, pos_names), (LABEL_NEG, neg_names)):
        for name in shard_names:
            path = download_docs(cfg, name, local)
            texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
            Path(path).unlink()
            dropped = near_drops.get(name, set())
            for row, text in enumerate(texts):
                if row not in dropped:
                    samples[label].add(text)
    return {label: r.items for label, r in samples.items()}, neg_names


def train_classifier(cfg, names, near_drops, local):
    """Train fastText on a 95/5 split of the samples; report holdout precision. Returns (model path, info)."""
    import fasttext  # imported here: only this step needs it

    qcfg, ft = cfg["quality"], cfg["quality"]["fasttext"]
    samples, neg_names = _sample_training_docs(cfg, names, near_drops, local)
    rng = random.Random(qcfg["seed"])
    lines = [f"{label} {_clean_for_fasttext(t)}" for label, texts in samples.items() for t in texts]
    rng.shuffle(lines)
    n_test = len(lines) // 20
    train_path, test_path = Path(local, "ft_train.txt"), Path(local, "ft_test.txt")
    train_path.write_text("\n".join(lines[n_test:]), encoding="utf-8")
    test_path.write_text("\n".join(lines[:n_test]), encoding="utf-8")
    del lines, samples
    model = fasttext.train_supervised(str(train_path), epoch=ft["epoch"], wordNgrams=ft["word_ngrams"],
                                      dim=ft["dim"], bucket=ft["bucket"], seed=qcfg["seed"],
                                      thread=cfg["runtime"]["workers"])
    n, precision, _ = model.test(str(test_path))
    model_path = Path(local, "quality_model.bin")
    model.save_model(str(model_path))
    train_path.unlink()
    test_path.unlink()
    info = {"holdout_docs": n, "holdout_accuracy": round(precision, 4), "negative_shards": neg_names,
            "docs_per_class": qcfg["docs_per_class"]}
    print(f"[3c] classifier holdout accuracy {precision:.4f} on {n:,} docs", flush=True)
    return model_path, info


def get_model(cfg, names, near_drops, local):
    """The classifier from the Hub, or train and upload it first (one model for every session)."""
    repo = cfg["hf"]["dataset_repo"]
    remote = set(hub.list_files(repo, _prefix(cfg)))
    if _prefix(cfg, "model/quality_model.bin") in remote:
        model_path = hub.download(repo, _prefix(cfg, "model/quality_model.bin"), local)
        info = json.loads(Path(hub.download(repo, _prefix(cfg, "model/train_info.json"), local)).read_text())
        return model_path, info
    model_path, info = train_classifier(cfg, names, near_drops, local)
    up = Path(local, "model_up")
    up.mkdir(parents=True, exist_ok=True)
    model_path.rename(up / "quality_model.bin")
    (up / "train_info.json").write_text(json.dumps(info, indent=1))
    hub.upload_folder(repo, up, _prefix(cfg, "model"), message="Quality classifier")
    return str(up / "quality_model.bin"), info


# ---------------------------------------------------------------- scoring

_WORKER = {}


def _init_worker(model_path):
    import fasttext

    _WORKER["model"] = fasttext.load_model(model_path)


def _score_row_group(task):
    """Worker: P(hq) for every doc of one parquet row group.

    Uses the list form of predict(): the single-string form calls np.array(copy=False),
    which fails under numpy 2.
    """
    path, group = task
    texts = pq.ParquetFile(path).read_row_group(group, columns=["text"]).column("text").to_pylist()
    labels, probs = _WORKER["model"].predict([_clean_for_fasttext(t) for t in texts], k=1)
    return np.array([p_hq(doc_labels, doc_probs) for doc_labels, doc_probs in zip(labels, probs)], dtype=np.float32)


def p_hq(doc_labels, doc_probs):
    """P(hq) from fastText's top-1 prediction. Two classes, so P(hq) = 1 - P(web).

    v2's first scoring run read P(hq) out of a k=2 prediction and stored values that were
    never below 0.5 (the top label's confidence, whichever label it was), so the ranking
    kept confident docs, not good ones. Top-1 plus the complement cannot mix the two up.
    """
    label, p = doc_labels[0], min(float(doc_probs[0]), 1.0)  # fastText adds 1e-5 before normalizing
    if label == LABEL_POS:
        return p
    if label == LABEL_NEG:
        return 1.0 - p
    raise ValueError(f"unexpected fastText label {label!r}")


def check_scores(scores, names, never_drop):
    """STOP unless the scores look like P(hq): spread over [0, 1] and higher for the positive sources."""
    web = np.concatenate([scores[n] for n in names if source_of(n) not in never_drop])
    pos = np.concatenate([scores[n] for n in names if source_of(n) in never_drop])
    below = float((web < 0.5).mean())
    print(f"[3c] score check: web median {np.median(web):.3f}, {100 * below:.1f}% below 0.5; "
          f"positive-source median {np.median(pos):.3f}", flush=True)
    if below < 0.05 or np.median(pos) <= np.median(web):
        raise SystemExit("STOP: quality scores do not look like P(hq); check the scoring step before selecting")


def score_all(cfg, names, model_path, local):
    """Score every shard not yet scored on the Hub; upload in batches. Returns {shard: scores}."""
    repo = cfg["hf"]["dataset_repo"]
    prefix = _prefix(cfg, "scores/")
    done = {Path(p).stem for p in hub.list_files(repo, prefix) if p.endswith(".npy")}
    batch_dir = Path(local, "scores_up")
    batch_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if n not in done]
    print(f"[3c] scoring {len(todo)} shards ({len(done)} already scored)", flush=True)
    with Pool(cfg["runtime"]["workers"], initializer=_init_worker, initargs=(str(model_path),)) as pool:
        for k, name in enumerate(todo):
            path = download_docs(cfg, name, local)
            groups = pq.ParquetFile(path).num_row_groups
            scores = np.concatenate(pool.map(_score_row_group, [(path, g) for g in range(groups)]))
            Path(path).unlink()
            np.save(batch_dir / f"{name}.npy", scores)
            print(f"[3c] scored {name}: {len(scores):,} docs, mean {scores.mean():.3f}", flush=True)
            if (k + 1) % SCORE_UPLOAD_BATCH == 0 or k == len(todo) - 1:
                hub.upload_folder(repo, batch_dir, prefix.rstrip("/"), message="Quality scores")
                for f in batch_dir.iterdir():
                    f.unlink()
    return {n: np.load(hub.download(repo, f"{prefix}{n}.npy", local)) for n in names}


# ---------------------------------------------------------------- selection

def select(cfg, names, scores, meta, near_drops):
    """Keep protected sources, then the best web docs until the train-token target.

    meta: {shard: n_bytes array}. Returns (drops {shard: rows array}, threshold, info).
    """
    qcfg, tcfg = cfg["quality"], cfg["target"]
    bpt, train_frac = tcfg["bytes_per_token_estimate"], cfg["split"]["train"]
    budget = tcfg["train_tokens_target"] * bpt / train_frac  # upsampled bytes before the split

    protected_bytes = 0
    web_shard, web_row, web_score, web_bytes = [], [], [], []
    for s, name in enumerate(names):
        source = source_of(name)
        alive = np.ones(len(meta[name]), dtype=bool)
        alive[list(near_drops.get(name, ()))] = False
        if source in qcfg["never_drop"]:
            protected_bytes += int(meta[name][alive].sum()) * cfg["sources"][source]["upsample_train"]
            continue
        rows = np.nonzero(alive)[0]
        web_shard.append(np.full(len(rows), s, dtype=np.int32))
        web_row.append(rows.astype(np.int32))
        web_score.append(scores[name][rows])
        web_bytes.append(meta[name][rows] * cfg["sources"][source]["upsample_train"])
    web_shard, web_row = np.concatenate(web_shard), np.concatenate(web_row)
    web_score, web_bytes = np.concatenate(web_score), np.concatenate(web_bytes)

    order = np.argsort(-web_score, kind="stable")
    cum = np.cumsum(web_bytes[order])
    n_keep = int(np.searchsorted(cum, max(budget - protected_bytes, 0), side="right"))
    threshold = float(web_score[order[n_keep - 1]]) if n_keep else 1.0
    drop_idx = order[n_keep:]

    drops = defaultdict(list)
    for s, r in zip(web_shard[drop_idx].tolist(), web_row[drop_idx].tolist()):
        drops[names[s]].append(r)
    drops = {k: np.array(sorted(v), dtype=np.int32) for k, v in drops.items()}

    dropped = defaultdict(lambda: {"docs": 0, "bytes": 0})
    for name, rows in drops.items():
        dropped[source_of(name)]["docs"] += len(rows)
        dropped[source_of(name)]["bytes"] += int(meta[name][rows].sum())
    deciles = {}
    for source in sorted({source_of(n) for n in names}):
        vals = np.concatenate([scores[n] for n in names if source_of(n) == source])
        deciles[source] = [round(float(x), 3) for x in np.quantile(vals, np.linspace(0, 1, 11))]
    info = {
        "ran": True, "budget_bytes": int(budget), "protected_bytes_upsampled": int(protected_bytes),
        "web_docs_ranked": int(len(order)), "web_docs_kept": n_keep,
        "web_bytes_ranked": int(web_bytes.sum()), "web_bytes_kept": int(cum[n_keep - 1]) if n_keep else 0,
        "threshold": round(threshold, 4), "dropped": {k: dict(v) for k, v in dropped.items()},
        "score_deciles": deciles,
    }
    return drops, threshold, info


def write_audit(cfg, names, scores, drops, near_drops, threshold, local, out_dir, ranked=True):
    """Re-read a few web shards: random dropped docs, docs on both sides of the cut, top docs.

    ranked=False (random selection): only the random dropped docs; a cut line means nothing there.
    """
    qcfg = cfg["quality"]
    rng = random.Random(qcfg["seed"])
    web = [n for n in names if source_of(n) not in qcfg["never_drop"]]
    picks = sorted(rng.sample(web, min(qcfg["audit_shards"], len(web))))
    chars = cfg["audit"]["chars_shown"]
    audit = FilterAudit(cfg["audit"]["removed_per_stage_per_source"], chars, qcfg["seed"])
    just_kept, just_dropped, top = (Reservoir(qcfg["audit_docs"], rng) for _ in range(3))
    for name in picks:
        path = download_docs(cfg, name, local)
        table = pq.read_table(path, columns=["id", "text"]).to_pydict()
        Path(path).unlink()
        dropped = set(drops.get(name, np.zeros(0)).tolist())
        near = near_drops.get(name, set())
        for row, (doc_id, text) in enumerate(zip(table["id"], table["text"])):
            if row in near:
                continue
            s = float(scores[name][row])
            entry = (s, doc_id, text[:chars])
            if row in dropped:
                audit.record("quality", source_of(name), doc_id, f"score={s:.3f}", text)
                if s >= threshold - 0.05:
                    just_dropped.add(entry)
            else:
                if s <= threshold + 0.05:
                    just_kept.add(entry)
                if s >= 0.95:
                    top.add(entry)
    audit.write(Path(out_dir) / "filter_audit")
    if not ranked:
        (Path(out_dir) / "quality_audit.md").write_text(
            "# Web selection audit\n\nMethod: random draw (no ranking), so there is no cut line to show. "
            f"Random dropped docs: filter_audit/quality/. Shards read: {', '.join(picks)}.\n", encoding="utf-8")
        return

    lines = ["# Quality ranking audit", "", f"Threshold P(hq) = {threshold:.4f}. Shards read: {', '.join(picks)}.", ""]
    for title, res in (("Just above the cut (kept, score <= threshold + 0.05)", just_kept),
                       ("Just below the cut (dropped, score >= threshold - 0.05)", just_dropped),
                       ("Top-scored (kept, score >= 0.95)", top)):
        lines += [f"## {title}", ""]
        for s, doc_id, text in sorted(res.items, key=lambda e: -e[0]):
            lines += [f"### {doc_id} — {s:.3f}", "", "```text", text, "```", ""]
    (Path(out_dir) / "quality_audit.md").write_text("\n".join(lines), encoding="utf-8")


def random_scores(cfg, names, meta):
    """A uniform draw per doc, seeded by (seed, shard name): a reproducible random order of the web docs."""
    seed = cfg["quality"]["seed"]
    return {n: np.random.default_rng([seed, zlib.crc32(n.encode())]).random(len(meta[n])).astype(np.float32)
            for n in names}


def drop_list_name(cfg):
    """Where the selection is cached: the fastText run keeps its original name, other methods get their own."""
    method = cfg["quality"]["method"]
    return "quality" if method == "fasttext" else f"quality_{method}"


def run_quality(cfg, names, near_drops, meta, local, out_dir):
    """Score every web doc (classifier or random draw), select up to the target. Returns (drops, info)."""
    Path(local).mkdir(parents=True, exist_ok=True)
    if cfg["quality"]["method"] == "random":
        scores = random_scores(cfg, names, meta)
        drops, threshold, info = select(cfg, names, scores, meta, near_drops)
        info.pop("score_deciles")  # deciles of a uniform draw say nothing
        info["method"] = "random"
        write_audit(cfg, names, scores, drops, near_drops, threshold, local, out_dir, ranked=False)
        return drops, info
    model_path, train_info = get_model(cfg, names, near_drops, local)
    scores = score_all(cfg, names, model_path, local)
    check_scores(scores, names, cfg["quality"]["never_drop"])
    drops, threshold, info = select(cfg, names, scores, meta, near_drops)
    info["method"] = "fasttext"
    info["classifier"] = train_info
    write_audit(cfg, names, scores, drops, near_drops, threshold, local, out_dir)
    return drops, info
