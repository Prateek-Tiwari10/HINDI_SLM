"""Stages 3 → 4 → 5 in one Kaggle kernel (hindi-lm-d6-pretrain), with the gates enforced in code.

Every kernel run needs a manual Save Version (API pushes cannot read HF_TOKEN), so the
stages that need HF are chained to keep those clicks to one per ~11 h session:

1. Stage 3 (kill-and-resume test). If d6_results/stage3.json on HF says it passed, skip.
   If it fails: stop here (gate 3).
2. Stage 4 (LR sweep). If d6_results/stage4.json says all runs completed, skip and reuse
   its pick. The pick follows the pre-registered rule (lowest final val loss; 1e-3 only if
   it did not spike).
3. Stage 5 (pretraining session) with the picked LR, until the epoch ends or the 11h15m
   budget. Re-running the kernel resumes from pretrain_61m/latest.txt.

After each stage its outputs (out/d5d6/*) are uploaded to d6_results/ in the checkpoints
repo, so progress can be read from the laptop while the kernel runs.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml

from data import hub

OUT = Path("/kaggle/working/out/d5d6")
RESULTS_PREFIX = "d6_results/"


def repo() -> str:
    """HF checkpoints repo from the pretrain config."""
    return yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())["checkpoint"]["hf_repo"]


def remote_json(name: str) -> dict | None:
    """d6_results/<name> from HF, or None if absent."""
    if f"{RESULTS_PREFIX}{name}" not in hub.list_files(repo(), RESULTS_PREFIX, repo_type="model"):
        return None
    path = hub.download(repo(), f"{RESULTS_PREFIX}{name}", "/kaggle/working/_results_dl", repo_type="model")
    return json.loads(Path(path).read_text())


def publish(stage: str) -> None:
    """Upload everything in out/d5d6 to d6_results/ (one commit)."""
    if OUT.exists():
        hub.upload_folder(repo(), OUT, RESULTS_PREFIX.rstrip("/"), repo_type="model", message=f"d6 results after {stage}")


def run(module: str, *args: str) -> int:
    """python -m module args...; returns the exit code."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"[d6] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    """Run whatever stages are not done yet."""
    OUT.mkdir(parents=True, exist_ok=True)

    s3 = remote_json("stage3.json")
    if s3 and s3.get("pass"):
        print("[d6] stage 3 already passed; skipping", flush=True)
    else:
        rc = run("train.stage3")
        publish("stage3")
        if rc != 0:
            sys.exit("[d6] GATE 3 FAILED — stopping before the LR sweep")

    s4 = remote_json("stage4.json")
    if not (s4 and s4.get("all_complete")):
        rc = run("train.stage4")
        publish("stage4")
        s4 = json.loads((OUT / "stage4.json").read_text()) if (OUT / "stage4.json").exists() else None
        if rc != 0 or not s4 or not s4.get("pick_lr"):
            sys.exit("[d6] stage 4 incomplete — re-run the kernel to finish the sweep")
    else:
        print(f"[d6] stage 4 already done; pick {s4['pick']}", flush=True)
    lr = s4["pick_lr"]
    print(f"[d6] GATE 4: peak LR = {lr} ({s4['rule_note']})", flush=True)

    rc = run("train.stage5", "--peak-lr", str(lr))
    publish("stage5 session")
    if rc != 0:
        sys.exit(f"[d6] stage 5 session exited with {rc}")


if __name__ == "__main__":
    main()
