"""Stage 7c (D10): the Hinglish -> Devanagari pair dataset. Runs on Kaggle (CPU).

    python -m data.hinglish.stage7c realism   # public data only (no HF_TOKEN): sources, lexicon, realism
    python -m data.hinglish.stage7c all       # + v2 scan, contamination, gate, splits, upload (HF_TOKEN)

Steps (spec 7c.1-7c.6):
  1. Stream the Dakshina tar (keep hi/lexicons + hi/romanized), download Aksharantar hin.zip.
  2. Lexicon: Dakshina train lexicon first, Aksharantar train only for words Dakshina lacks (filtered);
     sampling p proportional to count^2 (build_lexicon, lexicon_weights).
  3. Realism: romanize Dakshina dev natives; word agreement + char-bigram KL vs the humans,
     for the generator and the rule-only baseline. Stops if agreement < gate.min_realism.
  4. One parallel pass over every v2 train shard: per-shard samples of candidate sentences, and
     the contamination scan (Dakshina dev/test natives and the chat set in pretraining: exact
     everywhere, near-dup in the contamination_near_dup_sources). Real eval records are tagged
     contamination = exact / near / clean.
  5. Exact dedup, leakage removal (exact + MinHash) against Dakshina dev/test and the chat set,
     coverage check (stops if < gate.min_coverage).
  6. Splits (train / val_syn / dev_real / test_real / test_chat), JSONL + pre-tokenized arrays,
     zero-leakage re-check, upload to the private pairs repo.
Report: OUT_DIR/stage7c.json, samples, plot.
"""

import gzip
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data import dedup_minhash, hub
from data.config import config_hash, load_config
from data.hinglish.romanize import (DEV_UNIT, GENERATOR_VERSION, SOURCES, SHORTCUTS, Romanizer,
                                    build_sampler, rule_romanize, seeded_rng)
from data.hinglish.sentences import classify, is_conversational, sentence_key, split_sentences
from data.normalize import normalize
from eval.metrics import edit_distance
from finetune.data import decode_budget
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer, normalize_rom

CONFIG = "configs/hinglish_data.yaml"
OUT_DIR = Path(os.environ.get("STAGE7C_OUT", "/kaggle/working/out/stage7c"))
WORK_DIR = Path(os.environ.get("STAGE7C_WORK", "/tmp/hindi-lm/stage7c"))
_DEV_WORD = re.compile(f"{DEV_UNIT}+")
_LATIN_WORD = re.compile("[a-z]+")
_G = {}                                  # read-only state shared with forked workers


def log(msg):
    print(f"[stage7c {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _HashingReader(io.RawIOBase):
    """File-like wrapper that sha256-hashes and counts every byte read through it."""

    def __init__(self, raw):
        self.raw, self.sha, self.n = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buf):
        data = self.raw.read(len(buf))
        self.sha.update(data)
        self.n += len(data)
        buf[: len(data)] = data
        return len(data)


# ------------------------------------------------------------------ 1. sources
def fetch_dakshina(cfg, report):
    """Stream the tar once; keep <lang>/lexicons and <lang>/romanized. Records sha256 of the whole archive."""
    lang = cfg["sources"]["dakshina_lang"]
    dest = WORK_DIR / "dakshina"
    keep = re.compile(rf"(^|/){lang}/(lexicons|romanized)/[^/]+$")
    done = dest / "_done.json"
    if done.exists():
        report["dakshina"] = json.loads(done.read_text())
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    log(f"streaming {cfg['sources']['dakshina_url']}")
    with urllib.request.urlopen(cfg["sources"]["dakshina_url"], timeout=600) as resp:
        reader = _HashingReader(resp)
        files = {}
        with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|") as tar:
            for member in tar:
                if member.isfile() and keep.search(member.name):
                    data = tar.extractfile(member).read()
                    name = member.name.split("/")[-1]
                    (dest / name).write_bytes(data)
                    files[name] = {"bytes": len(data), "lines": data.count(b"\n")}
        while reader.read(1 << 20):          # drain padding so the sha covers the whole archive
            pass
    info = {"url": cfg["sources"]["dakshina_url"], "archive_bytes": reader.n,
            "archive_sha256": reader.sha.hexdigest(), "license": cfg["sources"]["dakshina_license"],
            "files": dict(sorted(files.items()))}
    done.write_text(json.dumps(info, indent=1))
    report["dakshina"] = info
    log(f"dakshina: {len(files)} {lang} files, archive {reader.n / 1e9:.2f} GB")
    return dest


def _dakshina_file(dest, lang, suffix):
    path = dest / f"{lang}.{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not in the Dakshina archive; have: {sorted(p.name for p in dest.iterdir())}")
    return path


def dakshina_pairs(dest, lang, part):
    """[(native, roman)] from the rejoined dev/test files (line-aligned)."""
    native = _dakshina_file(dest, lang, f"romanized.rejoined.{part}.native.txt").read_text(encoding="utf-8").split("\n")
    roman = _dakshina_file(dest, lang, f"romanized.rejoined.{part}.roman.txt").read_text(encoding="utf-8").split("\n")
    if native and native[-1] == "" and roman and roman[-1] == "":
        native, roman = native[:-1], roman[:-1]
    if len(native) != len(roman):
        raise ValueError(f"dakshina {part}: {len(native)} native vs {len(roman)} roman lines")
    return [(n.strip(), r.strip()) for n, r in zip(native, roman) if n.strip() and r.strip()]


def fetch_aksharantar(cfg, report):
    """Aksharantar Hindi train rows as dicts; records zip sha256, member names and columns."""
    src = cfg["sources"]
    path = Path(hub.download(src["aksharantar_repo"], src["aksharantar_file"], WORK_DIR / "aksharantar"))
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        train = [n for n in names if "train" in n.lower() and not n.endswith("/")]
        if len(train) != 1:
            raise ValueError(f"expected one train file in {src['aksharantar_file']}, got {train} (all: {names})")
        raw = zf.read(train[0]).decode("utf-8")
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        rows = json.loads(raw)
    report["aksharantar"] = {"repo": src["aksharantar_repo"], "file": src["aksharantar_file"],
                             "sha256": hub_sha(path), "members": names, "train_member": train[0],
                             "train_rows": len(rows), "columns": sorted(rows[0]), "license": src["aksharantar_license"],
                             "by_source": dict(Counter(r.get("source", "?") for r in rows).most_common())}
    log(f"aksharantar: {len(rows)} train rows from {train[0]}")
    return rows


def hub_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 2. lexicon
_FOLD = (("chh", "Ch"), ("ch", "C"), ("sh", "s"), ("ph", "f"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "j"),
         ("x", "ks"))
_VOWEL_RUN = re.compile("[aeiouy]*$")
_NASAL_END = ("\u0902", "\u0901")          # anusvara, chandrabindu


def _fold(lat):
    for a, b in _FOLD:
        lat = lat.replace(a, b)
    return lat


def first_sound(lat):
    """Onset class of a Latin spelling: 'V' for a vowel, else the first consonant (+ 'h' if aspirated).

    Folds c/k/q, w/v, z/j, ph/f, sh/s so that spelling habits match; aspiration is kept (कोई koi != khoyi).
    """
    f = _fold(lat)
    if not f:
        return ""
    if f[0] in "aeiou":
        return "V"
    return f[:2] if len(f) > 1 and f[1] == "h" else f[0]


def final_vowel(lat, nasal=False):
    """Class of the final vowel: A, E (e/ai/ay), I (i/ee/y), O (o/au), U (u/oo), or C (consonant).

    nasal: the Devanagari word ends in anusvara/chandrabindu, so a final n/m is the nasal, not a
    consonant (donon -> O, sakte -> E; hain -> E). A final h after a vowel is ignored (yeh, vah).
    """
    if nasal and lat.endswith(("n", "m")):
        lat = lat[:-1]
    if len(lat) > 1 and lat.endswith("h") and lat[-2] in "aeiou":
        lat = lat[:-1]
    run = _VOWEL_RUN.search(lat).group()
    if not run:
        return "C"
    for ends, cls in ((("ee", "ii"), "I"), (("oo", "uu"), "U"), (("ai", "ay", "ae", "ey", "e"), "E"),
                      (("au", "o"), "O"), (("i", "y"), "I"), (("u",), "U")):
        if run.endswith(ends):
            return cls
    return "A"


def build_lexicon(dakshina_tsv, aks_rows, cfg, report, key="lexicon"):
    """{dev_word: Counter(latin: count)} and stats. Dakshina train only (never dev/test lexicons).

    Precedence (D10, romanize-v2): a word in the Dakshina train lexicon keeps ONLY its Dakshina
    spellings (counts = annotator attestations). Aksharantar train rows are used only for words
    Dakshina lacks, and each spelling must pass, against rule_romanize(word):
      - edit distance / longer length <= max_rule_distance,
      - the same first sound (first_sound) and final vowel class (final_vowel),
      - the inflection guard: not the rule spelling or a Dakshina spelling of a different word
        (sakta is सकता's spelling, so it is rejected for सकते). Dakshina spellings are never
        removed by this guard: shared spellings (ki for कि and की) are real ambiguity.
    Then, per word, count-1 spellings are dropped when the top spelling has count >= drop_singletons_if_top_ge.
    Sampling weights (count ** weight_power) are applied by lexicon_weights().
    """
    lcfg = cfg["lexicon"]
    lo, hi, max_dist = lcfg["min_len_ratio"], lcfg["max_len_ratio"], lcfg["max_rule_distance"]
    kept, rejected = Counter(), Counter()
    rule_cache = {}

    def rule(dev):
        r = rule_cache.get(dev)
        if r is None:
            r = rule_cache[dev] = rule_romanize(dev)
        return r

    def clean(native, roman, source):
        dev, lat = normalize(native).strip(), normalize_rom(roman)
        if not _DEV_WORD.fullmatch(dev) or not _LATIN_WORD.fullmatch(lat):
            rejected[f"{source}:charset"] += 1
            return None
        if not lo <= len(lat) / len(dev) <= hi:
            rejected[f"{source}:length"] += 1
            return None
        return dev, lat

    dakshina = defaultdict(Counter)
    for line in dakshina_tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            rejected["dakshina:format"] += 1
            continue
        pair = clean(parts[0], parts[1], "dakshina")
        if pair:
            dakshina[pair[0]][pair[1]] += int(parts[2])
            kept["dakshina"] += 1

    aks_pairs = []
    for row in aks_rows:
        pair = clean(row["native word"], row["english word"], "aksharantar")
        if pair is None:
            continue
        if pair[0] in dakshina:
            rejected["aksharantar:word_in_dakshina"] += 1
            continue
        aks_pairs.append(pair)
    owners = defaultdict(set)            # latin spelling -> words whose rule or Dakshina spelling it is
    for dev, spellings in dakshina.items():
        owners[rule(dev)].add(dev)
        for lat in spellings:
            owners[lat].add(dev)
    for dev, _ in aks_pairs:
        owners[rule(dev)].add(dev)

    aksharantar = defaultdict(Counter)
    for dev, lat in aks_pairs:
        r = rule(dev)
        reason = None
        if edit_distance(lat, r) / max(len(lat), len(r), 1) > max_dist:
            reason = "rule_distance"
        elif first_sound(lat) != first_sound(r):
            reason = "first_sound"
        elif final_vowel(lat, dev.endswith(_NASAL_END)) != final_vowel(r, dev.endswith(_NASAL_END)):
            reason = "final_vowel"
        elif lat != r and owners.get(lat, set()) - {dev}:
            reason = "inflection_guard"
        if reason:
            rejected[f"aksharantar:{reason}"] += 1
        else:
            aksharantar[dev][lat] += 1
            kept["aksharantar"] += 1

    table, singletons_dropped = {}, 0
    for source_table in (dakshina, aksharantar):
        for dev, spellings in source_table.items():
            if spellings.most_common(1)[0][1] >= lcfg["drop_singletons_if_top_ge"]:
                n = len(spellings)
                spellings = Counter({lat: c for lat, c in spellings.items() if c > 1})
                singletons_dropped += n - len(spellings)
            table[dev] = spellings
    n_spellings = [len(v) for v in table.values()]
    rng = random.Random(cfg["seed"])
    sample = rng.sample(sorted(table), min(10, len(table)))
    report[key] = {
        "precedence": "dakshina words: dakshina spellings only; aksharantar only for words dakshina lacks",
        "max_rule_distance": max_dist, "weight_power": lcfg["weight_power"],
        "drop_singletons_if_top_ge": lcfg["drop_singletons_if_top_ge"],
        "word_types": len(table), "word_types_dakshina": len(dakshina), "word_types_aksharantar": len(aksharantar),
        "aksharantar_words_fully_rejected": len({d for d, _ in aks_pairs} - set(aksharantar)),
        "spelling_pairs": sum(n_spellings), "avg_spellings_per_word": round(sum(n_spellings) / len(table), 3),
        "singleton_spellings_dropped": singletons_dropped,
        "kept_rows": dict(kept), "rejected_rows": dict(sorted(rejected.items())),
        "random_entries": {w: dict(table[w].most_common(6)) for w in sample},
    }
    log(f"{key}: {len(table)} word types ({len(dakshina)} dakshina, {len(aksharantar)} aksharantar), "
        f"{sum(n_spellings)} spellings; rejected {dict(rejected)}")
    return table


def lexicon_weights(table, cfg):
    """Sampling weights: count ** weight_power (D10: p proportional to count^2)."""
    power = cfg["lexicon"]["weight_power"]
    return {dev: {lat: c ** power for lat, c in spellings.items()} for dev, spellings in table.items()}


# ------------------------------------------------------------------ 3. realism
def _clean_tokens(tokens):
    return [t for t in (re.sub("[^a-z0-9]", "", x) for x in tokens) if t]


def _bigrams(texts):
    c = Counter()
    for t in texts:
        s = f" {t} "
        c.update(zip(s, s[1:]))
    return c


def kl_divergence(p_counts, q_counts, alpha=0.5):
    """KL(P || Q) in nats, additive smoothing alpha over the union of bigrams."""
    keys = set(p_counts) | set(q_counts)
    p_tot = sum(p_counts.values()) + alpha * len(keys)
    q_tot = sum(q_counts.values()) + alpha * len(keys)
    return sum(((p_counts[k] + alpha) / p_tot) * math.log(((p_counts[k] + alpha) / p_tot) / ((q_counts[k] + alpha) / q_tot))
               for k in keys)


def human_spellings(dev_pairs, cfg):
    """{devanagari word: Counter(human latin spelling)} from Dakshina dev sentences whose word counts align."""
    rule = Romanizer({}, cfg["generator"], use_shortcuts=False, use_lexicon=False)
    human = defaultdict(Counter)
    for native, roman in dev_pairs:
        hum = _clean_tokens(normalize_rom(roman).split())
        nat = [t for t in normalize(native).split()
               if re.sub("[^a-z0-9]", "", rule.sentence(t, random.Random(0), level="clean").rom)]
        if len(nat) == len(hum):
            for n, h in zip(nat, hum):
                human[n.strip("।॥?!.,;:'\"()")][h] += 1
    return human


def _weights_of(sampler, word, k=6):
    entry = sampler.get(word) if sampler is not None else None
    if entry is None:
        return None
    spellings, cum = entry
    total = cum[-1]
    weights = [c - p for c, p in zip(cum, [0] + cum[:-1])]
    return {s: round(w / total, 3) for s, w in list(zip(spellings, weights))[:k]}


def word_diagnostics(human, sampler, top=40):
    """For the most frequent Dakshina dev words: human spellings vs shortcut / lexicon (sampling p) / rule."""
    shortcuts = build_sampler(SHORTCUTS)
    out = []
    for word, spellings in sorted(human.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
        out.append({"word": word, "count": sum(spellings.values()), "human": dict(spellings.most_common(5)),
                    "shortcut_p": _weights_of(shortcuts, word), "lexicon_p": _weights_of(sampler, word),
                    "rules": rule_romanize(word)})
    return out


def realism(dev_pairs, sampler, cfg, report, human_table):
    """Word agreement (aligned sentences only) and bigram KL vs the human romanizations of Dakshina dev."""
    gcfg = cfg["generator"]
    variants = {"generator": (Romanizer(sampler, gcfg), None),
                "generator_clean": (Romanizer(sampler, gcfg), "clean"),
                "rule_only": (Romanizer(sampler, gcfg, use_shortcuts=False, use_lexicon=False), None)}
    human_texts = []
    out = {}
    for name, (romanizer, level) in variants.items():
        agree, total, aligned = 0, 0, 0
        per_source = {s: [0, 0] for s in SOURCES}
        texts, coverage = [], Counter()
        for i, (native, roman) in enumerate(dev_pairs):
            human = _clean_tokens(normalize_rom(roman).split())
            if name == "generator":
                human_texts.append(" ".join(human))
            r = romanizer.sentence(normalize(native), seeded_rng(cfg["seed"], "realism", name, i), level=level)
            coverage.update(r.source_counts())
            ours = [(t, s) for t, s in ((re.sub("[^a-z0-9]", "", tok), srcs) for tok, srcs in r.tokens) if t]
            texts.append(" ".join(t for t, _ in ours))
            if len(ours) != len(human):
                continue
            aligned += 1
            for (tok, srcs), h in zip(ours, human):
                total += 1
                agree += tok == h
                if len(srcs) == 1:
                    per_source[srcs[0]][0] += 1
                    per_source[srcs[0]][1] += tok == h
        out[name] = {
            "aligned_sentences": aligned, "aligned_words": total,
            "word_agreement": round(agree / total, 4) if total else 0.0,
            "agreement_by_source": {s: {"words": n, "agreement": round(a / n, 4) if n else None}
                                    for s, (n, a) in per_source.items()},
            "word_source_share": {s: round(coverage[s] / sum(coverage.values()), 4) for s in SOURCES},
            "_texts": texts,
        }
    human_bigrams = _bigrams(human_texts)
    report["realism_diagnostics"] = word_diagnostics(human_table, sampler)
    for name in variants:
        out[name]["bigram_kl_human_vs_this"] = round(kl_divergence(human_bigrams, _bigrams(out[name].pop("_texts"))), 5)
    ends_danda = [(n, r) for n, r in dev_pairs if n.rstrip().endswith(("।", "॥"))]
    report["realism"] = {
        "dev_sentences": len(dev_pairs), **out,
        "human_dot_after_danda": round(sum(r.rstrip().endswith(".") for _, r in ends_danda) / max(1, len(ends_danda)), 4),
        "dev_sentences_ending_in_danda": len(ends_danda),
        "examples": [{"dev": n, "human": r, "generator": variants["generator"][0].sentence(
            normalize(n), seeded_rng(cfg["seed"], "realism", "generator", i)).rom}
            for i, (n, r) in enumerate(dev_pairs[:8])],
    }
    log(f"realism: generator {out['generator']['word_agreement']:.3f}, rule-only {out['rule_only']['word_agreement']:.3f}"
        f" word agreement on {out['generator']['aligned_sentences']}/{len(dev_pairs)} aligned sentences")
    return out["generator"]["word_agreement"]


# ------------------------------------------------------------------ 4. v2 scan (forked workers)
def _signature(text):
    return dedup_minhash.signature(text, _G["shingle"], _G["perms"])


def _ref_index(keys_by_idx):
    """MinHash LSH index of reference sentences: (signatures, [ {band_key: [idx]} per band ])."""
    lk = _G["lk"]
    sigs = np.stack([_signature(k) for k in keys_by_idx]) if keys_by_idx else np.zeros((0, lk["num_perm"]), np.uint32)
    bands = dedup_minhash.band_keys(sigs, lk["bands"], lk["rows_per_band"], lk["seed"])
    maps = []
    for b in range(lk["bands"]):
        m = defaultdict(list)
        for i, key in enumerate(bands[:, b].tolist()):
            m[key].append(i)
        maps.append(m)
    return sigs, maps


def _near_matches(sigs, index):
    """For each row of sigs: set of reference indices with signature agreement >= threshold."""
    lk = _G["lk"]
    ref_sigs, maps = index
    if not len(sigs) or not len(ref_sigs):
        return [set() for _ in range(len(sigs))]
    keys = dedup_minhash.band_keys(sigs, lk["bands"], lk["rows_per_band"], lk["seed"])
    out = [set() for _ in range(len(sigs))]
    for b, m in enumerate(maps):
        hits = np.nonzero(np.isin(keys[:, b], np.fromiter(m.keys(), np.uint64, len(m))))[0]
        for i in hits.tolist():
            out[i].update(m[int(keys[i, b])])
    for i, cands in enumerate(out):
        if cands:
            cands_l = sorted(cands)
            agree = (ref_sigs[cands_l] == sigs[i][None, :]).mean(axis=1)
            out[i] = {c for c, a in zip(cands_l, agree.tolist()) if a >= lk["threshold"]}
    return out


def _scan_shard(path_in_repo):
    import pyarrow.parquet as pq

    cfg, scfg = _G["cfg"], _G["cfg"]["sentences"]
    name = path_in_repo.rsplit("/", 1)[-1].removesuffix(".parquet")
    source = name.rsplit("-", 2)[0]
    near = source in cfg["leakage"]["contamination_near_dup_sources"]
    local_dir = WORK_DIR / "shards" / name
    path = Path(hub.download(scfg["data_repo"], path_in_repo, local_dir))
    conv, general = [], []
    exact_hits, near_hits = set(), set()
    stats = Counter()
    near_batch = []

    def flush_near():
        if near_batch:
            sigs = np.stack([_signature(k) for k in near_batch])
            for hit in _near_matches(sigs, _G["ref_index"]):
                near_hits.update(hit)
            near_batch.clear()

    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1000):
        for text in batch.column(0).to_pylist():
            stats["docs"] += 1
            for s in split_sentences(text):
                stats["sentences"] += 1
                key = sentence_key(s)
                hit = _G["ref_keys"].get(key)
                if hit is not None:
                    exact_hits.update(hit)
                if near and len(key.split()) >= 4:
                    near_batch.append(key)
                    if len(near_batch) >= 20000:
                        flush_near()
                c = classify(s, scfg, _G["conv_words"])
                if c is None:
                    continue
                stats["candidates"] += 1
                stats["conversational"] += c
                (conv if c else general).append(s)
    flush_near()
    path.unlink(missing_ok=True)
    rng = seeded_rng(cfg["seed"], "shard", name)
    conv = rng.sample(conv, min(len(conv), scfg["per_shard_conversational"]))
    general = rng.sample(general, min(len(general), scfg["per_shard_general"]))
    return {"name": name, "source": source, "stats": dict(stats), "conv": conv, "general": general,
            "exact_hits": sorted(exact_hits), "near_hits": sorted(near_hits), "near_scanned": near}


def _signatures_chunk(keys):
    return np.stack([_signature(k) for k in keys]) if keys else np.zeros((0, _G["lk"]["num_perm"]), np.uint32)


def _parallel_signatures(pool, keys, chunk=20000):
    parts = pool.map(_signatures_chunk, [keys[i:i + chunk] for i in range(0, len(keys), chunk)])
    return np.concatenate(parts) if parts else np.zeros((0, _G["lk"]["num_perm"]), np.uint32)


# ------------------------------------------------------------------ 6. splits
def _record(split, i, rom, dev, source, noise_level, conversational, mix, n_tokens, prompt_len):
    return {"id": f"{split}-{i:06d}", "rom": rom, "dev": dev, "source": source, "noise_level": noise_level,
            "conversational": conversational, "rom_source_mix": mix, "n_tokens": n_tokens, "prompt_len": prompt_len}


def write_split(out_dir, split, records, tokenized):
    """<split>.jsonl.gz + <split>.ids.npy (uint16, concatenated input_ids) + offsets (int64) + prompt_len (uint16)."""
    with gzip.open(out_dir / f"{split}.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lengths = np.array([len(ids) for ids, _ in tokenized], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = np.fromiter((t for ids, _ in tokenized for t in ids), dtype=np.int64, count=int(offsets[-1]))
    assert flat.max(initial=0) < 2 ** 16
    np.save(out_dir / f"{split}.ids.npy", flat.astype(np.uint16))
    np.save(out_dir / f"{split}.offsets.npy", offsets)
    np.save(out_dir / f"{split}.prompt_len.npy", np.array([p for _, p in tokenized], dtype=np.uint16))


def build_synthetic(pools, n_conv, n_gen, split, romanizer, tok, cfg, start_counter, spell=None):
    """Draw sentences from the shared pools until n_conv + n_gen pairs fit max_tokens.

    spell: optional {"freq", "pre", "post", "src"} Counters, filled per Devanagari word unit with its
    spelling before/after noise and its source (for the top-200 spelling report).
    """
    max_tokens = cfg["splits"]["max_tokens"]
    records, tokenized, dropped = [], [], Counter()
    for kind, need in (("conv", n_conv), ("general", n_gen)):
        got = 0
        while got < need:
            if not pools[kind]:
                raise RuntimeError(f"{split}: {kind} pool exhausted after {got}/{need}")
            dev = pools[kind].pop()
            i = start_counter[0]
            start_counter[0] += 1
            r = romanizer.sentence(dev, seeded_rng(cfg["seed"], "pair", split, i))
            if not r.rom:
                dropped["empty_rom"] += 1
                continue
            input_ids, labels = tok.build_pair(r.rom, dev)
            if len(input_ids) > max_tokens:
                dropped["over_max_tokens"] += 1
                continue
            prompt_len = len(input_ids) - sum(lab != -100 for lab in labels)
            records.append(_record(split, len(records), r.rom, dev, "v2", r.noise_level, kind == "conv",
                                   r.source_mix(), len(input_ids), prompt_len))
            tokenized.append((input_ids, prompt_len))
            got += 1
            if spell is not None:
                for unit, pre, post, src in r.units:
                    spell["freq"][unit] += 1
                    spell["pre"][unit, pre] += 1
                    spell["post"][unit, post] += 1
                    spell["src"][unit, src] += 1
    order = list(range(len(records)))
    random.Random(f"{cfg['seed']}:{split}:order").shuffle(order)
    records = [dict(records[j], id=f"{split}-{k:06d}") for k, j in enumerate(order)]
    tokenized = [tokenized[j] for j in order]
    return records, tokenized, dict(dropped)


def build_real(pairs, split, source, tok, conv_words, cfg, contamination):
    """Human pairs are kept whole (eval only); n_tokens over max_tokens is reported, not dropped.

    contamination: per pair "exact" / "near" / "clean" (seen in v2 pretraining or not), stored on the record.
    """
    records, tokenized = [], []
    for i, (rom, native) in enumerate(pairs):
        dev = normalize(native)
        input_ids, labels = tok.build_pair(rom, dev)
        prompt_len = len(input_ids) - sum(lab != -100 for lab in labels)
        records.append(dict(_record(split, i, rom, dev, source, None, is_conversational(dev, conv_words), None,
                                    len(input_ids), prompt_len), contamination=contamination[i]))
        tokenized.append((input_ids, prompt_len))
    return records, tokenized


def split_summary(records, tokenized, tok, cfg):
    n = np.array([r["n_tokens"] for r in records])
    lunk = tok.token_to_id["<lunk>"]
    return {"pairs": len(records), "tokens": int(n.sum()),
            "n_tokens": {"mean": round(float(n.mean()), 1), "p50": int(np.percentile(n, 50)),
                         "p99": int(np.percentile(n, 99)), "max": int(n.max())},
            "over_max_tokens": int((n > cfg["splits"]["max_tokens"]).sum()),
            "pairs_with_lunk": sum(lunk in ids for ids, _ in tokenized),
            "conversational_share": round(sum(r["conversational"] for r in records) / len(records), 4),
            "noise_levels": dict(Counter(r["noise_level"] for r in records if r["noise_level"])),
            "source_share": _source_share(records),
            "contamination": dict(Counter(r["contamination"] for r in records)) if "contamination" in records[0] else None}


def decode_cap_stats(tokenized, cfg):
    """Ratio of Devanagari tokens (incl. eos) to romanized characters; its quantile sets the decode cap.

    cap(prompt) = ceil(ratio_q * factor * rom_chars) + extra, rom_chars = prompt_len - 2 (one id per char).
    """
    dcfg = cfg["decode_cap"]
    ratios = np.array([(len(ids) - p) / max(1, p - 2) for ids, p in tokenized])
    return {"ratio_quantile": dcfg["quantile"], "ratio": round(float(np.percentile(ratios, dcfg["quantile"])), 4),
            "factor": dcfg["factor"], "extra": dcfg["extra"],
            "formula": "max_new = ceil(ratio * factor * (prompt_len - 2)) + extra, capped by eval.max_new_tokens"}


def spelling_report(spell, human, top=200, k=6):
    """Spelling distribution in the built train split for its `top` most frequent Devanagari words."""
    def dist(counter, word, n):
        items = sorted(((s, c) for (w, s), c in counter.items() if w == word), key=lambda x: -x[1])[:k]
        return {s: round(c / n, 3) for s, c in items}

    words = [w for w, _ in spell["freq"].most_common(top)]
    wanted = set(words)
    sub = {name: Counter({key: c for key, c in spell[name].items() if key[0] in wanted}) for name in ("pre", "post", "src")}
    out = []
    for w in words:
        n = spell["freq"][w]
        h = human.get(w)
        out.append({"word": w, "count": n, "spelling": dist(sub["pre"], w, n), "after_noise": dist(sub["post"], w, n),
                    "source": dist(sub["src"], w, n), "rules": rule_romanize(w),
                    "human_dakshina_dev": {s: round(c / sum(h.values()), 3) for s, c in h.most_common(4)} if h else None})
    return out


def _source_share(records):
    tot = Counter()
    for r in records:
        if r["rom_source_mix"]:
            words = len(r["rom"].split())   # approximate weight; exact counts are in the coverage step
            for s, frac in r["rom_source_mix"].items():
                tot[s] += frac * words
    total = sum(tot.values())
    return {s: round(tot[s] / total, 4) for s in SOURCES} if total else None


def plot_lengths(splits, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 260, 8)
    for name, records in splits.items():
        ax.hist([r["n_tokens"] for r in records], bins=bins, density=True, histtype="step", linewidth=1.6,
                label=f"{name} (n={len(records):,})")
    ax.axvline(256, color="grey", linestyle=":", label="max_tokens = 256")
    ax.set_title("Stage 7c: built pair length (<rom> … <dev> … </s>)")
    ax.set_xlabel("tokens per pair")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dataset_card(cfg, report):
    lex, sp = report["lexicon"], report["splits"]
    rows = "\n".join(f"| {k} | {v['pairs']:,} | {v['n_tokens']['mean']} | {v['conversational_share']} |" for k, v in sp.items())
    return f"""---
license: other
language: [hi]
pretty_name: Hindi LM Hinglish pairs
---
# Hinglish → Devanagari pairs ({GENERATOR_VERSION})

Private training data for the Hindi LM D7 fine-tune (Stage 7c). Files are under `{GENERATOR_VERSION}/`.

| split | pairs | mean tokens | conversational |
|---|---|---|---|
{rows}

- `train`, `val_syn`: v2 Hindi sentences (Sangraha verified, Wikipedia, course crawl) romanized by
  `data/hinglish/romanize.py` ({GENERATOR_VERSION}): chat shortcuts, then human spellings sampled from
  the lexicon ({lex['word_types']:,} word types), then rules; noise levels clean/mild/heavy.
- `dev_real`, `test_real`: Dakshina v1.0 Hindi romanized sentences (rejoined dev/test), human-written.
  `test_real` is held out for the final evaluation. Each record has `contamination`: exact / near / clean
  (whether the Devanagari sentence occurs in the v2 pretraining corpus); report clean and contaminated apart.
- `meta.json` → `decode_cap`: per-prompt greedy decoding budget measured on `train`.
- Lexicon sources: Dakshina train lexicon ({cfg['sources']['dakshina_license']}) and Aksharantar Hindi
  train ({cfg['sources']['aksharantar_license']}). Derived files carry those licenses' terms.
- Format: `<split>.jsonl.gz` (rom, dev, source, noise_level, conversational, rom_source_mix) and
  pre-tokenized `<split>.ids.npy` / `.offsets.npy` / `.prompt_len.npy` built with tokenizer
  `{report['tokenizer']['version']}` (sha {report['tokenizer']['base_sha256'][:12]}).
"""


# ------------------------------------------------------------------ main
def main(mode):
    cfg = load_config(CONFIG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    report = {"mode": mode, "generator_version": GENERATOR_VERSION, "config_hash": config_hash(CONFIG),
              "shortcuts": len(SHORTCUTS), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    report_path = OUT_DIR / "stage7c.json"

    def save():
        report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")

    def stop(reason):
        report["stopped"] = reason
        save()
        log(f"STOP: {reason}")
        sys.exit(2)

    if mode == "all" and not os.environ.get("HF_TOKEN"):
        stop("HF_TOKEN missing: 'all' reads the private v2 data. Run it with Save Version (Secrets ticked).")

    # 1-2. sources + lexicon
    lang = cfg["sources"]["dakshina_lang"]
    dk = fetch_dakshina(cfg, report)
    dev_pairs, test_pairs = dakshina_pairs(dk, lang, "dev"), dakshina_pairs(dk, lang, "test")
    report["dakshina"]["dev_pairs"], report["dakshina"]["test_pairs"] = len(dev_pairs), len(test_pairs)
    aks = fetch_aksharantar(cfg, report)
    lex_tsv = _dakshina_file(dk, lang, "translit.sampled.train.tsv")
    table = build_lexicon(lex_tsv, aks, cfg, report)
    del aks
    sampler = build_sampler(lexicon_weights(table, cfg))
    with gzip.open(WORK_DIR / "lexicon.json.gz", "wt", encoding="utf-8") as f:   # raw counts; p ~ count^2
        json.dump({w: dict(v) for w, v in table.items()}, f, ensure_ascii=False)
    save()

    # 3. realism
    human = human_spellings(dev_pairs, cfg)
    agreement = realism(dev_pairs, sampler, cfg, report, human)
    save()
    if agreement < cfg["gate"]["min_realism"]:
        stop(f"realism word agreement {agreement:.3f} < {cfg['gate']['min_realism']}")
    if mode == "realism":
        report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()
        log("realism mode done")
        return

    # 4. v2 scan
    scfg, lk = cfg["sentences"], cfg["leakage"]
    conv_words = {normalize(w) for w in scfg["conversational_words"]}
    chat_path = Path(cfg["sources"]["chat_test"])
    chat_pairs = []
    if chat_path.exists():
        for line in chat_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                chat_pairs.append((parts[0].strip(), parts[1].strip()))
    report["test_chat"] = {"present": chat_path.exists(), "pairs": len(chat_pairs)}

    # reference sentences (Devanagari side of every real eval split), global index order dev, test, chat
    ref_splits = [("dev_real", [n for n, _ in dev_pairs]), ("test_real", [n for n, _ in test_pairs]),
                  ("test_chat", [d for _, d in chat_pairs])]
    ref_keys, ref_owner = [], []
    for split, natives in ref_splits:
        for i, n in enumerate(natives):
            ref_keys.append(sentence_key(normalize(n)))
            ref_owner.append((split, i))
    key_to_refs = defaultdict(list)
    for g, k in enumerate(ref_keys):
        key_to_refs[k].append(g)
    _G.update(cfg=cfg, conv_words=conv_words, ref_keys=dict(key_to_refs), lk=lk, shingle=lk["shingle"],
              perms=dedup_minhash.make_permutations(lk["num_perm"], lk["seed"]))
    ref_index = _G["ref_index"] = _ref_index(ref_keys)

    shard_paths = hub.list_files(scfg["data_repo"], scfg["train_prefix"])
    shard_paths = [p for p in shard_paths if p.endswith(".parquet")]
    log(f"scanning {len(shard_paths)} v2 train shards with {scfg['workers']} workers")
    pool_conv, pool_gen = [], []
    exact_hits, near_hits, near_sources = set(), set(), set()
    per_source = defaultdict(Counter)
    ctx = mp.get_context("fork")
    with ctx.Pool(scfg["workers"]) as pool:
        for k, res in enumerate(pool.imap_unordered(_scan_shard, shard_paths), 1):
            pool_conv.extend(res["conv"])
            pool_gen.extend(res["general"])
            exact_hits.update(res["exact_hits"])
            near_hits.update(res["near_hits"])
            if res["near_scanned"]:
                near_sources.add(res["source"])
            per_source[res["source"]].update(res["stats"])
            per_source[res["source"]]["shards"] += 1
            if k % 10 == 0 or k == len(shard_paths):
                log(f"  {k}/{len(shard_paths)} shards; pool conv {len(pool_conv):,} general {len(pool_gen):,}")
    report["scan"] = {"shards": len(shard_paths), "per_source": {s: dict(c) for s, c in per_source.items()},
                      "sampled_conversational": len(pool_conv), "sampled_general": len(pool_gen)}
    contamination = {s: ["clean"] * len(n) for s, n in ref_splits}      # per real record: exact / near / clean
    for g in near_hits:
        split, i = ref_owner[g]
        contamination[split][i] = "near"
    for g in exact_hits:
        split, i = ref_owner[g]
        contamination[split][i] = "exact"
    report["contamination"] = {
        **{s: {"sentences": len(f), "exact": f.count("exact"), "near_only": f.count("near"),
               "contaminated": len(f) - f.count("clean"), "clean": f.count("clean"),
               "contaminated_share": round((len(f) - f.count("clean")) / max(1, len(f)), 4)}
           for s, f in contamination.items()},
        "definition": "contaminated = exact or near-duplicate of a v2 pretraining sentence; clean = neither",
        "near_dup_sources_scanned": sorted(near_sources),
        "method": "exact: sentence key (letters+digits) over every v2 train sentence; near: MinHash "
                  f"word-{lk['shingle']}-gram, {lk['bands']}x{lk['rows_per_band']} LSH, agreement >= {lk['threshold']}, "
                  "only in the listed sources",
    }
    log(f"contamination: { {s: v['contaminated'] for s, v in report['contamination'].items() if isinstance(v, dict) and 'clean' in v} }")
    save()

    # 5. exact dedup, leakage removal, coverage
    def dedup(sentences, seen):
        out = []
        for s in sentences:
            k = sentence_key(s)
            if k and k not in seen:
                seen.add(k)
                out.append(s)
        return out

    seen = set()
    pool_conv, pool_gen = dedup(pool_conv, seen), dedup(pool_gen, seen)
    ref_key_set = set(ref_keys)
    leak = Counter()
    with ctx.Pool(scfg["workers"]) as pool:
        clean_pools = {}
        for kind, sentences in (("conv", pool_conv), ("general", pool_gen)):
            keys = [sentence_key(s) for s in sentences]
            near = _near_matches(_parallel_signatures(pool, keys), ref_index)
            kept = []
            for s, k, nm in zip(sentences, keys, near):
                if k in ref_key_set:
                    leak["exact"] += 1
                elif nm:
                    leak["near"] += 1
                else:
                    kept.append(s)
            clean_pools[kind] = kept
    report["pool"] = {"after_dedup": {"conv": len(pool_conv), "general": len(pool_gen)},
                      "leakage_removed": dict(leak),
                      "after_leakage": {k: len(v) for k, v in clean_pools.items()}}
    log(f"pool after dedup+leakage: {report['pool']['after_leakage']} (removed {dict(leak)})")

    romanizer = Romanizer(sampler, cfg["generator"])
    cov_rng = random.Random(cfg["seed"])
    cov_sample = cov_rng.sample(clean_pools["conv"], min(35000, len(clean_pools["conv"]))) + \
        cov_rng.sample(clean_pools["general"], min(15000, len(clean_pools["general"])))
    cov = Counter()
    for i, s in enumerate(cov_sample):
        cov.update(romanizer.sentence(s, seeded_rng(cfg["seed"], "coverage", i)).source_counts())
    coverage = (cov["shortcut"] + cov["lexicon"]) / sum(cov.values())
    report["coverage_pre_gate"] = {"sentences": len(cov_sample), "coverage": round(coverage, 4),
                                   "by_source": {s: round(cov[s] / sum(cov.values()), 4) for s in SOURCES}}
    save()
    if coverage < cfg["gate"]["min_coverage"]:
        stop(f"lexicon coverage {coverage:.3f} < {cfg['gate']['min_coverage']}")

    # 6. splits
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    report["tokenizer"] = {"version": tok.version, "base_sha256": tok.base_sha256, "vocab_size": tok.vocab_size}
    for kind in clean_pools:
        random.Random(f"{cfg['seed']}:pool:{kind}").shuffle(clean_pools[kind])
    share = scfg["conversational_share"]
    counter = [0]
    splits, tokenized, dropped = {}, {}, {}
    spell = {name: Counter() for name in ("freq", "pre", "post", "src")}
    for split in ("val_syn", "train"):
        n = cfg["splits"][split]
        n_conv = round(n * share)
        splits[split], tokenized[split], dropped[split] = build_synthetic(
            clean_pools, n_conv, n - n_conv, split, romanizer, tok, cfg, counter, spell if split == "train" else None)
        log(f"{split}: {len(splits[split]):,} pairs, dropped {dropped[split]}")
    splits["dev_real"], tokenized["dev_real"] = build_real(
        [(r, n) for n, r in dev_pairs], "dev_real", "dakshina", tok, conv_words, cfg, contamination["dev_real"])
    splits["test_real"], tokenized["test_real"] = build_real(
        [(r, n) for n, r in test_pairs], "test_real", "dakshina", tok, conv_words, cfg, contamination["test_real"])
    if chat_pairs:
        splits["test_chat"], tokenized["test_chat"] = build_real(chat_pairs, "test_chat", "chat", tok, conv_words, cfg,
                                                             contamination["test_chat"])

    # zero-leakage and disjointness re-check on the final splits
    syn_keys = {s: [sentence_key(r["dev"]) for r in splits[s]] for s in ("train", "val_syn")}
    with ctx.Pool(scfg["workers"]) as pool:
        recheck = {}
        for s, keys in syn_keys.items():
            near = _near_matches(_parallel_signatures(pool, keys), ref_index)
            recheck[s] = {"exact": sum(k in ref_key_set for k in keys), "near": sum(bool(x) for x in near)}
    overlap = {"train_val_syn": len(set(syn_keys["train"]) & set(syn_keys["val_syn"])),
               "dev_real_test_real": len({sentence_key(r["dev"]) for r in splits["dev_real"]}
                                         & {sentence_key(r["dev"]) for r in splits["test_real"]})}
    report["leakage_recheck"] = {"hits": recheck, "overlap": overlap}
    if any(v["exact"] or v["near"] for v in recheck.values()) or overlap["train_val_syn"]:
        stop(f"leakage in final splits: {report['leakage_recheck']}")

    train_cov = Counter()
    for r in splits["train"]:
        words = sum(1 for _ in re.finditer(f"{DEV_UNIT}+", r["dev"]))
        for s, frac in r["rom_source_mix"].items():
            train_cov[s] += round(frac * words)
    report["coverage_train"] = {"coverage": round((train_cov["shortcut"] + train_cov["lexicon"]) / sum(train_cov.values()), 4),
                                "by_source": {s: round(train_cov[s] / sum(train_cov.values()), 4) for s in SOURCES}}
    report["splits"] = {s: split_summary(splits[s], tokenized[s], tok, cfg) for s in splits}
    report["dropped"] = dropped
    cap = decode_cap_stats(tokenized["train"], cfg)
    cap["refs_longer_than_cap"] = {
        s: sum(len(ids) - p > min(decode_budget(p, cap), cfg["decode_cap"]["max_new_tokens"]) for ids, p in tokenized[s])
        for s in splits}
    report["decode_cap"] = cap
    log(f"decode cap: {cap}")
    report["spelling_top200"] = spelling_report(spell, human)
    del spell
    sample_rng = random.Random(f"{cfg['seed']}:samples:{GENERATOR_VERSION}")
    report["samples_train"] = [{k: r[k] for k in ("rom", "dev", "noise_level", "conversational", "rom_source_mix")}
                               for r in sample_rng.sample(splits["train"], 30)]
    report["samples_dev_real"] = [{k: r[k] for k in ("rom", "dev")} for r in sample_rng.sample(splits["dev_real"], 10)]
    plot_lengths({s: splits[s] for s in ("train", "val_syn", "dev_real")}, OUT_DIR / "stage7c_lengths.png")

    pairs_dir = WORK_DIR / "pairs"
    data_dir = pairs_dir / GENERATOR_VERSION
    data_dir.mkdir(parents=True, exist_ok=True)
    for s in splits:
        write_split(data_dir, s, splits[s], tokenized[s])
    (data_dir / "lexicon.json.gz").write_bytes((WORK_DIR / "lexicon.json.gz").read_bytes())
    meta = {k: report[k] for k in ("generator_version", "config_hash", "tokenizer", "lexicon", "splits", "dropped",
                                   "decode_cap", "contamination")}
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    (pairs_dir / "README.md").write_text(dataset_card(cfg, report))
    save()

    repo = cfg["hf"]["pairs_repo"]
    api = hub.get_api()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    hub.upload_folder(repo, pairs_dir, "", message=f"Stage 7c pairs {GENERATOR_VERSION}")
    report["upload"] = {"repo": repo, "path": GENERATOR_VERSION, "revision": api.dataset_info(repo).sha,
                        "bytes": sum(p.stat().st_size for p in pairs_dir.rglob("*") if p.is_file())}
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    log(f"uploaded to {repo} @ {report['upload']['revision'][:10]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "realism")
