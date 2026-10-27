"""Stage 4 driver (Kaggle T4×2, needs HF_TOKEN): short LR sweep and the mechanical pick.

Peak LR in {3e-4, 6e-4, 1e-3}, each with the real config but 600 steps (warmup 200, cosine
over 600), each in its own HF folder sweep/lr_X/. Runs are resumable: if the session dies,
re-running the kernel continues each unfinished run from its latest checkpoint.

Rule: lowest final val loss wins; if 1e-3 wins but had a spike (train loss > 0.5 above
its running mean), take the next best.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from eval.plots import plot_lr_sweep, read_jsonl

LRS = [3e-4, 6e-4, 1e-3]
SPIKE = 0.5
MEAN_WINDOW = 20                       # logged points (every 10 steps -> 200 steps)


def label(lr: float) -> str:
    """Folder-safe name, e.g. 3e-4."""
    return f"{lr:.0e}".replace("e-0", "e-")


def spikes(train: list[dict]) -> list[dict]:
    """Logged points whose loss is > SPIKE above the mean of the previous MEAN_WINDOW points."""
    out = []
    losses = [r["loss"] for r in train]
    for i in range(MEAN_WINDOW, len(losses)):
        mean = float(np.mean(losses[i - MEAN_WINDOW:i]))
        if losses[i] > mean + SPIKE or not np.isfinite(losses[i]):
            out.append({"step": train[i]["step"], "loss": losses[i], "running_mean": round(mean, 4)})
    return out


def main() -> None:
    """Run (or finish) the three runs, then summarise and pick."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d5d6")
    p.add_argument("--steps", type=int, default=600)
    args = p.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    runs, summary = {}, {}
    for lr in LRS:
        name = label(lr)
        log = f"logs/sweep_lr_{name}.jsonl"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "-m", "train.train",
               "--peak-lr", str(lr), "--total-steps", str(args.steps), "--warmup-steps", "200",
               "--hf-prefix", f"sweep/lr_{name}/", "--log-path", log, "--save-every", str(args.steps),
               "--eval-every", "100", "--budget-s", "41000"]
        print(f"[stage4] $ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        recs = read_jsonl(log)
        runs[f"lr {name}"] = recs
        train = [r for r in recs if r["type"] == "train"]
        val = [r for r in recs if r["type"] == "val"]
        scales = [r["scale"] for r in train]
        halts = [r for r in recs if r["type"] == "event" and r["event"] == "halt"]
        summary[name] = {
            "halted": halts[-1] if halts else None,
            "lr": lr, "exit_code": rc, "steps_done": train[-1]["step"] if train else 0,
            "final_val_loss": val[-1]["val_loss"] if val and val[-1]["step"] == args.steps else None,
            "final_val_bpb": val[-1]["val_bpb"] if val else None,
            "final_val_windows": val[-1]["windows"] if val else None,
            "val_curve": [(r["step"], r["val_loss"]) for r in val],
            "final_train_loss": train[-1]["loss"] if train else None,
            "spikes": spikes(train),
            "scaler_decreases": sum(1 for x, y in zip(scales, scales[1:]) if y < x),
            "min_scale": min(scales) if scales else None,
            "max_grad_norm": max(r["grad_norm"] for r in train) if train else None,
            "tok_per_s_median": float(np.median([r["tok_per_s"] for r in train[1:]])) if len(train) > 1 else None,
        }
        print(f"[stage4] {name}: {json.dumps({k: v for k, v in summary[name].items() if k != 'val_curve'})}",
              flush=True)
        (out / f"sweep_lr_{name}.jsonl").write_text(Path(log).read_text())

    plot_lr_sweep(runs, out / "plots" / "stage4_lr_sweep.png")
    done = {k: v for k, v in summary.items() if v["final_val_loss"] is not None}
    ranked = sorted(done, key=lambda k: done[k]["final_val_loss"])
    pick = ranked[0] if ranked else None
    note = "lowest final val loss"
    if pick == label(1e-3) and done[pick]["spikes"] and len(ranked) > 1:
        pick, note = ranked[1], "1e-3 had the lowest val loss but spiked; next best taken"
    result = {"runs": summary, "ranking": ranked, "pick": pick, "pick_lr": done[pick]["lr"] if pick else None,
              "rule_note": note,
              "all_complete": all(v["final_val_loss"] is not None or v["halted"] for v in summary.values())}
    (out / "stage4.json").write_text(json.dumps(result, indent=1))
    print(f"[stage4] ranking {ranked} -> pick {pick} ({note})", flush=True)
    if not result["all_complete"]:
        sys.exit("[stage4] not all runs completed; re-run the kernel to resume")


if __name__ == "__main__":
    main()
