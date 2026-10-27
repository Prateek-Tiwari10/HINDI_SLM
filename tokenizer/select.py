"""Stage 4e: pick the tokenizer by the D4 rule, applied mechanically, printing every step.

1. Vocab size, per algorithm: the smallest size where the next size improves fertility
   by < 3%, provided embedding share <= 15%. If none qualifies: 16384, and say so.
2. If the two algorithms chose different sizes, compare both at the larger one.
3. At that size: if fertility differs by more than 2%, the lower fertility wins.
   Otherwise, a clear win on heavy splits or morphology (without losing the other)
   decides it. Ties go to Unigram.
4. If fertility is within 2%, flag that the optional proxy-run tie-break exists (not run).
"""


def choose_vocab(rows, model_type, sel, log):
    """Step 1 for one algorithm. rows: metric dicts from evaluate.py."""
    by_size = sorted((r for r in rows if r["model_type"] == model_type), key=lambda r: r["vocab_size"])
    for current, nxt in zip(by_size, by_size[1:]):
        gain = (current["fertility"] - nxt["fertility"]) / current["fertility"]
        ok_embed = current["embedding_pct"] <= 100 * sel["max_embedding_share"]
        log.append(f"- {model_type} {current['vocab_size']} -> {nxt['vocab_size']}: fertility "
                   f"{current['fertility']:.4f} -> {nxt['fertility']:.4f}, gain {100 * gain:.2f}% "
                   f"(threshold {100 * sel['fertility_gain_threshold']:.0f}%), embedding {current['embedding_pct']:.1f}%")
        if gain < sel["fertility_gain_threshold"] and ok_embed:
            log.append(f"  -> {model_type}: choose {current['vocab_size']}")
            return current["vocab_size"]
    log.append(f"  -> {model_type}: no size meets the rule; falling back to {sel['fallback_vocab']}")
    return sel["fallback_vocab"]


def _pct_diff(a, b):
    return abs(a - b) / min(a, b)


def compare_algorithms(bpe, uni, sel, log):
    """Steps 3-4 at one vocab size. Returns (winner, tie_break_available)."""
    diff = _pct_diff(bpe["fertility"], uni["fertility"])
    log.append(f"- fertility: BPE {bpe['fertility']:.4f} vs Unigram {uni['fertility']:.4f} "
               f"(diff {100 * diff:.2f}%, tie margin {100 * sel['tie_fertility_margin']:.0f}%)")
    log.append(f"- heavy splits: BPE {bpe['heavy_split_pct']:.2f}% vs Unigram {uni['heavy_split_pct']:.2f}%")
    log.append(f"- morphology: BPE {bpe['morphology_pct']:.1f}% vs Unigram {uni['morphology_pct']:.1f}%")
    if diff > sel["tie_fertility_margin"]:
        winner = "bpe" if bpe["fertility"] < uni["fertility"] else "unigram"
        log.append(f"  -> fertility differs by more than the margin: {winner} wins")
        return winner, False

    heavy_gap = uni["heavy_split_pct"] - bpe["heavy_split_pct"]   # > 0 means BPE splits less
    morph_gap = bpe["morphology_pct"] - uni["morphology_pct"]     # > 0 means BPE is better
    heavy_m, morph_m = sel["clear_margin_heavy_split_pp"], sel["clear_margin_morphology_pp"]
    bpe_clear = (heavy_gap >= heavy_m and morph_gap >= 0) or (morph_gap >= morph_m and heavy_gap >= 0)
    uni_clear = (-heavy_gap >= heavy_m and -morph_gap >= 0) or (-morph_gap >= morph_m and -heavy_gap >= 0)
    if bpe_clear and not uni_clear:
        log.append("  -> fertility tied; BPE clearly better on a secondary metric: bpe wins")
        return "bpe", True
    if uni_clear and not bpe_clear:
        log.append("  -> fertility tied; Unigram clearly better on a secondary metric: unigram wins")
        return "unigram", True
    log.append(f"  -> tie: goes to {sel['tie_winner']} (decision log: Unigram preferred on ties)")
    return sel["tie_winner"], True


def select(rows, sel):
    """Apply the rule. Returns (chosen row, markdown log)."""
    log = ["## Step 1 — vocab size per algorithm", ""]
    trained = {r["model_type"] for r in rows}
    if len(trained) == 1:
        only = trained.pop()
        size = choose_vocab(rows, only, sel, log)
        chosen = next(r for r in rows if r["model_type"] == only and r["vocab_size"] == size)
        log += ["", f"Only {only} tokenizers trained successfully, so steps 2-4 do not apply.",
                "", f"**Recommended tokenizer: `{chosen['name']}`**"]
        return chosen, "\n".join(log)
    sizes = {t: choose_vocab(rows, t, sel, log) for t in ("bpe", "unigram")}
    size = max(sizes.values()) if sel["common_size_rule"] == "larger" else min(sizes.values())
    log += ["", "## Step 2 — common size", "",
            f"- BPE chose {sizes['bpe']}, Unigram chose {sizes['unigram']} -> compare at {size}", "",
            "## Step 3 — algorithm", ""]
    at_size = {r["model_type"]: r for r in rows if r["vocab_size"] == size}
    winner, tie_break = compare_algorithms(at_size["bpe"], at_size["unigram"], sel, log)
    log += ["", "## Step 4 — optional tie-break", ""]
    if tie_break:
        log.append("- BPE and Unigram are within 2% on fertility: the optional tie-break "
                   "(one ~20-minute proxy run each) is available. Not run; Prateek decides.")
    else:
        log.append("- Not applicable (fertility gap exceeds 2%).")
    chosen = at_size[winner]
    log += ["", f"**Recommended tokenizer: `{chosen['name']}`**"]
    return chosen, "\n".join(log)
