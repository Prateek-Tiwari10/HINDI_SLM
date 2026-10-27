"""Plotting only: every function takes already-computed numbers and writes one PNG.

Each plot has a title, axis labels, and a legend when it shows more than one line.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict]:
    """Records of a JSONL training log."""
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overfit(losses: list[float], path: str | Path, threshold: float = 0.5) -> None:
    """Train loss per step when overfitting one fixed batch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, label="train loss (fixed 4×1024 batch)")
    ax.axhline(threshold, color="red", linestyle="--", label=f"gate: {threshold}")
    ax.set_yscale("log")
    ax.set_title("Stage 2: overfitting one batch (LR 1e-3)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss (nats, log scale)")
    ax.legend()
    _save(fig, path)


def plot_bench(records: list[dict], path: str | Path) -> None:
    """Tokens/s (both GPUs) per micro-batch, one bar group per compile setting."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = sorted({r["micro_batch"] for r in records})
    width = 0.38
    for j, comp in enumerate([False, True]):
        vals = [next((r["tok_per_s"] for r in records if r["micro_batch"] == b and r["compile"] == comp), 0)
                for b in sizes]
        ax.bar([i + (j - 0.5) * width for i in range(len(sizes))], vals, width,
               label=f"compile {'on' if comp else 'off'}")
    ax.set_xticks(range(len(sizes)), [str(b) for b in sizes])
    ax.set_title("Stage 2: training throughput on 2×T4 (0 = OOM or failed)")
    ax.set_xlabel("micro-batch per GPU")
    ax.set_ylabel("tokens / s (both GPUs)")
    ax.legend()
    _save(fig, path)


def plot_resume(a: dict[int, float], b: dict[int, float], b_killed: dict[int, float], path: str | Path) -> None:
    """Stage 3: per-step loss of run A, run B before the kill, and run B after resuming."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax.plot(list(a), list(a.values()), label="run A (uninterrupted)", lw=2)
    ax.plot(list(b_killed), list(b_killed.values()), "--", label="run B before kill (to step 95)")
    resumed = {s: v for s, v in b.items() if s > 80}
    ax.plot(list(resumed), list(resumed.values()), ":", lw=2, label="run B resumed from step 80")
    ax.axvline(80, color="grey", lw=0.8)
    ax.set_title("Stage 3: kill-and-resume test")
    ax.set_ylabel("train loss (nats)")
    ax.legend()
    steps = [s for s in resumed if s in a]
    ax2.plot(steps, [abs(a[s] - resumed[s]) for s in steps], marker=".", label="|A − B resumed|")
    ax2.axhline(1e-4, color="red", linestyle="--", label="tolerance 1e-4")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("abs loss difference")
    ax2.legend()
    _save(fig, path)


def plot_lr_sweep(runs: dict[str, list[dict]], path: str | Path) -> None:
    """Train loss (thin) and val loss (markers) against step, one colour per LR."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, recs) in enumerate(runs.items()):
        color = f"C{i}"
        tr = [r for r in recs if r["type"] == "train"]
        va = [r for r in recs if r["type"] == "val"]
        ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], color=color, alpha=0.6, lw=1,
                label=f"{label} train")
        ax.plot([r["step"] for r in va], [r["val_loss"] for r in va], color=color, marker="o", lw=2,
                label=f"{label} val")
    ax.set_ylim(top=min(8.0, ax.get_ylim()[1]))
    ax.set_title("Stage 4: LR sweep (600 steps, warmup 200, cosine)")
    ax.set_xlabel("optimizer step (262,144 tokens each)")
    ax.set_ylabel("loss (nats/token)")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_loss_vs_tokens(recs: list[dict], path: str | Path) -> None:
    """Train and val loss against tokens seen."""
    tr = [r for r in recs if r["type"] == "train"]
    va = [r for r in recs if r["type"] == "val"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["tokens"] / 1e9 for r in tr], [r["loss"] for r in tr], lw=0.8, alpha=0.7, label="train")
    ax.plot([r["tokens"] / 1e9 for r in va], [r["val_loss"] for r in va], marker="o", lw=2, label="val")
    ax.set_ylim(top=min(6.0, ax.get_ylim()[1]))
    ax.set_title("Pretraining loss")
    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel("loss (nats/token)")
    ax.legend()
    _save(fig, path)


def plot_series(recs: list[dict], key: str, ylabel: str, title: str, path: str | Path,
                log_y: bool = False) -> None:
    """One train-record field against step (LR, grad norm, GradScaler scale, tokens/s)."""
    tr = [r for r in recs if r["type"] == "train"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["step"] for r in tr], [r[key] for r in tr], lw=0.8)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(ylabel)
    _save(fig, path)
