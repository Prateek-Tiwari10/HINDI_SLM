"""Stage 4a': does a bigger tokenizer sample give a better tokenizer?

One tokenizer (ladder.model_type, ladder.vocab_size) is trained per sample rung (5%, 15%, 30% of
train) and scored on the same val docs. The rule, fixed in configs/tokenizer.yaml before any result:

  use the smallest rung whose fertility is within fertility_tol (relative) of the largest trained
  rung and whose heavy-split rate is within heavy_split_tol_pp (absolute) of it.

The extension rung (50%) is trained only when the last two base rungs still differ by more than
the tolerances, i.e. the curve has not flattened. A rung that the RAM guard rejects is skipped and
logged; the rule then uses the largest rung that was trained.
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import hub  # noqa: E402
from tokenizer import evaluate, sample, train  # noqa: E402


def rung_name(lcfg, fraction):
    return f"ladder_{lcfg['model_type']}_{lcfg['vocab_size']}_p{round(100 * fraction):02d}"


def available_ram_bytes():
    """MemAvailable from /proc/meminfo."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def ram_check(lcfg, text_bytes, available):
    """(fits, estimated peak bytes) for training on text_bytes of sample."""
    need = lcfg["ram_per_text_byte"] * text_bytes
    return need <= lcfg["ram_headroom"] * available, need


def within(row, ref, lcfg):
    """True when row is as good as ref within the ladder tolerances."""
    fert_ok = row["fertility"] <= ref["fertility"] * (1 + lcfg["fertility_tol"])
    heavy_ok = row["heavy_split_pct"] <= ref["heavy_split_pct"] + lcfg["heavy_split_tol_pp"]
    return fert_ok and heavy_ok


def choose_fraction(rows, lcfg, log):
    """The rule. rows: evaluated rungs in increasing sample order. Returns the chosen row."""
    ref = rows[-1]
    log.append(f"- reference = largest trained rung: {ref['sample_pct']:.1f}% of train "
               f"(fertility {ref['fertility']:.4f}, heavy splits {ref['heavy_split_pct']:.2f}%)")
    for row in rows:
        gap = 100 * (row["fertility"] - ref["fertility"]) / ref["fertility"]
        heavy_gap = row["heavy_split_pct"] - ref["heavy_split_pct"]
        ok = within(row, ref, lcfg)
        log.append(f"- {row['sample_pct']:.1f}%: fertility {row['fertility']:.4f} ({gap:+.2f}% vs reference, "
                   f"tolerance {100 * lcfg['fertility_tol']:.1f}%), heavy splits {row['heavy_split_pct']:.2f}% "
                   f"({heavy_gap:+.2f} pp, tolerance {lcfg['heavy_split_tol_pp']:.2f} pp) -> "
                   f"{'within tolerance' if ok else 'worse'}")
        if ok:
            log.append(f"  -> choose the {row['sample_pct']:.1f}% sample: the smallest within tolerance")
            return row
    return ref  # unreachable: the reference is always within tolerance of itself


def should_extend(rows, lcfg):
    """Train the extension rung only if the last two rungs still differ by more than the tolerances."""
    return len(rows) >= 2 and not within(rows[-2], rows[-1], lcfg)


def plot_ladder(rows, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    xs = [r["sample_pct"] for r in rows]
    ax.plot(xs, [r["fertility"] for r in rows], color="#eb6834", marker="s", markersize=8, linewidth=2)
    for r in rows:
        ax.annotate(f"{r['fertility']:.4f}", (r["sample_pct"], r["fertility"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, color="#3d3d3a")
    ax.set_title("Tokenizer sample size vs fertility (unigram 8192)")
    ax.set_xlabel("Tokenizer sample, % of train bytes")
    ax.set_ylabel("Tokens per word")
    ax.set_xticks(xs)
    ax.grid(True, color="#e4e3dc", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run(cfg, paths, sample_info, val_path, out_dir, before_train):
    """Train + evaluate the rungs, apply the rule. Returns (chosen rung index, chosen info, markdown log)."""
    lcfg = cfg["ladder"]
    fractions = sample.all_fractions(cfg)
    base = [i for i, f in enumerate(fractions) if f in lcfg["fractions"]]
    extension = [i for i, f in enumerate(fractions) if f == lcfg["extend_fraction"] and i not in base]
    available = available_ram_bytes()
    log = ["## Ladder: tokenizer sample size", "",
           f"- machine: {train.resolve_threads(cfg)} threads, {available / 1e9:.0f} GB RAM available", ""]
    infos, skipped = {}, []

    def train_rung(k):
        files, size = sample.rung_input(paths, sample_info, k)
        fits, need = ram_check(lcfg, size["bytes"], available)
        if not fits:
            skipped.append(f"- {size['pct_of_train_bytes']:.1f}% skipped: needs ~{need / 1e9:.0f} GB RAM, "
                           f"{available / 1e9:.0f} GB available (headroom {lcfg['ram_headroom']})")
            print(f"[ladder] {skipped[-1]}", flush=True)
            return
        info = train.train_cached(cfg, files, size, rung_name(lcfg, fractions[k]), lcfg["model_type"],
                                  lcfg["vocab_size"], before_train)
        if "error" in info:
            skipped.append(f"- {size['pct_of_train_bytes']:.1f}% failed: {info['error']}")
        else:
            infos[k] = info

    def evaluated():
        ks = sorted(infos)
        rows, _ = evaluate.evaluate_all(cfg, [infos[k] for k in ks], val_path)
        return ks, rows

    for k in base:
        train_rung(k)
    ks, rows = evaluated()
    if not rows:
        raise SystemExit("STOP: no ladder rung trained; see the log")
    if extension and should_extend(rows, lcfg):
        log.append(f"- the last two rungs still differ by more than the tolerances -> trying the "
                   f"{100 * lcfg['extend_fraction']:.0f}% extension")
        train_rung(extension[0])
        ks, rows = evaluated()
    elif extension:
        log.append(f"- the last two rungs are within the tolerances -> the {100 * lcfg['extend_fraction']:.0f}% "
                   "extension is not needed")
    log += skipped + ["", "### Rule", ""]
    chosen = choose_fraction(rows, lcfg, log)
    k_chosen = ks[rows.index(chosen)]

    keys = ["sample_pct", "sample_lines", "fertility", "chars_per_token", "byte_fallback_pct", "heavy_split_pct",
            "morphology_pct", "train_seconds", "peak_ram_gb"]
    log += ["", "### Rungs", "", "| " + " | ".join(keys) + " |", "|" + " --- |" * len(keys)]
    log += ["| " + " | ".join(str(r[key]) for key in keys) + " |" for r in rows]
    with open(out_dir / "tokenizer_ladder.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_ladder(rows, out_dir / "plots" / "ladder_fertility.png")
    record = out_dir / "ladder.json"
    record.write_text(json.dumps({"rows": rows, "chosen": chosen, "skipped": skipped, "rule": lcfg}, indent=1))
    hub.upload_file(cfg["hf"]["tokenizer_repo"], record, f"{cfg['hf']['sweep_prefix']}ladder.json", repo_type="model")
    return k_chosen, infos[k_chosen], "\n".join(log)
