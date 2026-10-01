"""Stage 4c: intrinsic metrics for every tokenizer on the val text, plus plots and examples.

Metrics per tokenizer (results/tokenizer_sweep.csv):
  fertility        tokens per whitespace word
  chars_per_token  characters per token
  byte_fallback_%  % of tokens that are byte-fallback pieces (<0x..>)
  heavy_split_%    % of words split into >= 4 pieces
  embedding_%      vocab x d_model / total params
  morphology_%     see morph_words.py
  train time and peak RAM, from train.py
"""

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sentencepiece as spm  # noqa: E402

from data.config import REPO_ROOT  # noqa: E402
from tokenizer import morph_words  # noqa: E402

SERIES_STYLE = {  # fixed categorical order: slot 1 blue, slot 2 orange; marker shape as second cue
    "bpe": {"color": "#2a78d6", "marker": "o", "label": "BPE"},
    "unigram": {"color": "#eb6834", "marker": "s", "label": "Unigram"},
}


def load_val_texts(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def word_counts(texts):
    return Counter(word for text in texts for word in text.split())


def corpus_metrics(sp, texts, words, threads):
    """Fertility, chars/token and byte-fallback rate over whole docs."""
    ids_per_doc = sp.encode(texts, num_threads=threads)
    n_tokens = sum(len(ids) for ids in ids_per_doc)
    n_bytes_pieces = sum(1 for ids in ids_per_doc for i in ids if sp.is_byte(i))
    n_words = sum(words.values())
    n_chars = sum(len(t) for t in texts)
    return {
        "fertility": n_tokens / n_words,
        "chars_per_token": n_chars / n_tokens,
        "byte_fallback_pct": 100.0 * n_bytes_pieces / n_tokens,
    }


def heavy_split_pct(sp, words, min_pieces, threads):
    """% of word occurrences split into >= min_pieces pieces (each unique word encoded once)."""
    unique = list(words)
    pieces = sp.encode(unique, num_threads=threads)
    heavy = sum(words[w] for w, ids in zip(unique, pieces) if len(ids) >= min_pieces)
    return 100.0 * heavy / sum(words.values())


def evaluate_all(cfg, infos, val_path):
    """Metrics rows for every successfully trained tokenizer."""
    texts = load_val_texts(val_path)
    words = word_counts(texts)
    morph = morph_words.load(cfg)
    m = cfg["metrics"]
    threads = cfg["runtime"]["threads"]
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    rows = []
    for info in infos:
        if "error" in info:
            continue
        sp = spm.SentencePieceProcessor(model_file=str(out_dir / f"{info['name']}.model"))
        row = {"name": info["name"], "model_type": info["model_type"], "vocab_size": info["vocab_size"]}
        row.update(corpus_metrics(sp, texts, words, threads))
        row["heavy_split_pct"] = heavy_split_pct(sp, words, m["heavy_split_pieces"], threads)
        row["embedding_pct"] = 100.0 * info["vocab_size"] * m["embedding_dim"] / m["model_params"]
        row["morphology_pct"] = morph_words.morphology_score(sp, morph)
        row["train_seconds"] = info["train_seconds"]
        row["peak_ram_gb"] = info["peak_ram_gb"]
        rows.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        print(f"[eval] {row['name']}: fertility {row['fertility']:.3f}", flush=True)
    return rows, {"val_docs": len(texts), "val_words": sum(words.values()), "morph_words": len(morph)}


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, metric, ylabel, title, path):
    """One line per algorithm vs vocab size, legend + direct end labels."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for model_type, style in SERIES_STYLE.items():
        pts = sorted((r["vocab_size"], r[metric]) for r in rows if r["model_type"] == model_type)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], markersize=8,
                linewidth=2, label=style["label"])
        ax.annotate(style["label"], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#3d3d3a")
    ax.set_title(title)
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({r["vocab_size"] for r in rows}))
    ax.grid(True, color="#e4e3dc", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_plots(rows, plot_dir):
    plot_metric(rows, "fertility", "Tokens per word", "Fertility vs vocabulary size", plot_dir / "fertility_vs_vocab.png")
    plot_metric(rows, "heavy_split_pct", "% of words split into 4+ pieces", "Heavy splits vs vocabulary size",
                plot_dir / "heavy_split_vs_vocab.png")
    plot_metric(rows, "morphology_pct", "% words split at stem/suffix boundary", "Morphology score vs vocabulary size",
                plot_dir / "morphology_vs_vocab.png")


def example_tokenizations(cfg, rows):
    """Markdown: the same example sentences tokenized by every tokenizer."""
    out_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    lines = ["# Example tokenizations", ""]
    for row in rows:
        sp = spm.SentencePieceProcessor(model_file=str(out_dir / f"{row['name']}.model"))
        lines += [f"## {row['name']}", ""]
        for sentence in cfg["examples"]:
            pieces = sp.encode(sentence, out_type=str)
            lines.append(f"- ({len(pieces)}) " + " | ".join(pieces))
        lines.append("")
    return "\n".join(lines)


def write_outputs(cfg, rows, out_dir):
    out_dir = Path(out_dir)
    write_csv(rows, out_dir / "tokenizer_sweep.csv")
    write_plots(rows, out_dir / "plots")
    (out_dir / "tokenizer_examples.md").write_text(example_tokenizations(cfg, rows), encoding="utf-8")
