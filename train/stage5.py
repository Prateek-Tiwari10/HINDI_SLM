"""Stage 5 driver (Kaggle T4×2, needs HF_TOKEN): one pretraining session + its summary.

Runs train.train with configs/pretrain_61m.yaml (resumes from latest.txt if present, stops
itself at the 11h15m budget), then summarises this session from the log and writes plots
of the whole run so far. Re-run the same kernel (Save Version) for the next session.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from eval.plots import plot_loss_vs_tokens, plot_series, read_jsonl


def main() -> None:
    """Run one session and write stage5_session.json + plots to --out."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d5d6")
    p.add_argument("--peak-lr", type=float, help="default: configs/pretrain_61m.yaml optim.peak_lr")
    args = p.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "-m", "train.train"]
    if args.peak_lr is not None:
        cmd += ["--peak-lr", str(args.peak_lr)]
    print(f"[stage5] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode

    log_path = Path(cfg["logging"]["jsonl"])
    recs = read_jsonl(log_path)
    starts = [i for i, r in enumerate(recs) if r["type"] == "event" and r["event"] in ("resume", "fresh_start")]
    session = recs[starts[-1]:] if starts else recs
    first = session[0]
    train = [r for r in session if r["type"] == "train"]
    val = [r for r in session if r["type"] == "val"]
    events = [r for r in session if r["type"] == "event"]
    end = next((r for r in reversed(events) if r["event"] == "session_end"), None)
    summary = {
        "exit_code": rc,
        "started": first.get("event"), "resumed_from": first.get("from"),
        "start_step": first["step"], "end_step": train[-1]["step"] if train else first["step"],
        "total_steps": first.get("total_steps"),
        "tokens_seen": train[-1]["tokens"] if train else None,
        "last_train_loss": train[-1]["loss"] if train else None,
        "last_val": val[-1] if val else None,
        "tok_per_s_median": float(np.median([r["tok_per_s"] for r in train[1:]])) if len(train) > 1 else None,
        "mfu_median": float(np.median([r["mfu"] for r in train[1:]])) if len(train) > 1 else None,
        "min_scale": min(r["scale"] for r in train) if train else None,
        "max_grad_norm": max(r["grad_norm"] for r in train) if train else None,
        "mem_gb": train[-1]["mem_gb"] if train else None,
        "halts": [r for r in events if r["event"] == "halt"],
        "stop_wall_clock": any(r["event"] == "stop_wall_clock" for r in events),
        "done": bool(end and end["done"]),
        "session_end": end,
    }
    (out / "stage5_session.json").write_text(json.dumps(summary, indent=1))
    (out / log_path.name).write_text(log_path.read_text())
    plot_loss_vs_tokens(recs, out / "plots" / "stage5_loss_vs_tokens.png")
    plot_series(recs, "lr", "learning rate", "Learning rate schedule", out / "plots" / "stage5_lr.png")
    plot_series(recs, "grad_norm", "global grad norm (before clipping)", "Gradient norm",
                out / "plots" / "stage5_grad_norm.png", log_y=True)
    print(f"[stage5] {json.dumps(summary, indent=1)}", flush=True)
    if rc != 0:
        sys.exit(f"[stage5] training exited with {rc}")


if __name__ == "__main__":
    main()
