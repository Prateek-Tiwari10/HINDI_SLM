"""Stage 3 driver (Kaggle T4×2, needs HF_TOKEN): kill-and-resume test.

Run A: 120 steps straight through. Run B: same config, killed with os._exit(1) right after
step 95 (last checkpoint on HF: step 80), then a fresh process resumes from HF and runs to
120. Gate: per-step train loss of steps 81–120 matches run A within 1e-4, and the token
batches of steps 81–120 hash identically. resume_test/ is deleted from HF afterwards.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from eval.plots import plot_resume, read_jsonl
from train.checkpoint import CheckpointManager

PREFIX = "resume_test/"
TOL = 1e-4


def torchrun(extra: list[str], env: dict) -> int:
    """Run train.train on 2 GPUs; return the exit code."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "-m", "train.train", *extra]
    print(f"[stage3] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def by_step(recs: list[dict]) -> dict[int, dict]:
    """Train records keyed by step."""
    return {r["step"]: r for r in recs if r["type"] == "train"}


def main() -> None:
    """Run A, B (killed), B (resumed); compare; clean up; write stage3.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d5d6")
    p.add_argument("--peak-lr", type=float, default=6e-4)
    args = p.parse_args()
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())
    ck = cfg["checkpoint"]
    scratch = CheckpointManager(ck["hf_repo"], PREFIX, ck["local_dir"], 2, "x.jsonl")
    if _list(scratch):                                     # leftovers of an earlier attempt
        scratch.delete_all()

    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    common = ["--total-steps", "120", "--save-every", "40", "--warmup-steps", "20", "--grad-accum", "2",
              "--micro-batch", "8",                        # math SDPA (deterministic) keeps the T×T scores
              "--compile", "0",                            # inductor kernels may use atomics; this tests resume
              "--peak-lr", str(args.peak_lr), "--log-every", "1", "--eval-every", "100000",
              "--final-full-eval", "0", "--log-batch-hashes", "--deterministic", "--budget-s", "100000"]
    Path("logs").mkdir(exist_ok=True)
    for f in ("logs/resume_A.jsonl", "logs/resume_B.jsonl"):
        Path(f).unlink(missing_ok=True)
    shutil.rmtree(ck["local_dir"], ignore_errors=True)

    rc_a = torchrun(common + ["--hf-prefix", PREFIX + "A/", "--log-path", "logs/resume_A.jsonl"], env)
    shutil.rmtree(ck["local_dir"], ignore_errors=True)
    rc_b1 = torchrun(common + ["--hf-prefix", PREFIX + "B/", "--log-path", "logs/resume_B.jsonl",
                               "--kill-at-step", "95"], env)
    shutil.copy("logs/resume_B.jsonl", "logs/resume_B_before_kill.jsonl")
    latest_b = CheckpointManager(ck["hf_repo"], PREFIX + "B/", ck["local_dir"], 2, "resume_B.jsonl").latest()
    shutil.rmtree(ck["local_dir"], ignore_errors=True)     # the resumed process must use HF, not local files
    rc_b2 = torchrun(common + ["--hf-prefix", PREFIX + "B/", "--log-path", "logs/resume_B.jsonl"], env)

    a = read_jsonl("logs/resume_A.jsonl")
    b_kill = read_jsonl("logs/resume_B_before_kill.jsonl")
    b = read_jsonl("logs/resume_B.jsonl")
    A, BK, B = by_step(a), by_step(b_kill), by_step(b)
    resume_events = [r for r in b if r["type"] == "event" and r["event"] == "resume"]
    steps = range(81, 121)
    loss_diff = [abs(A[s]["loss"] - B[s]["loss"]) for s in steps if s in A and s in B]
    pre_diff = [abs(A[s]["loss"] - BK[s]["loss"]) for s in range(1, 96) if s in A and s in BK]
    kill_vs_resume = [abs(BK[s]["loss"] - B[s]["loss"]) for s in range(81, 96) if s in BK and s in B]
    hashes_equal = all(A[s].get("batch_hash") == B[s].get("batch_hash") for s in steps if s in A and s in B)
    complete = all(s in A and s in B for s in steps)
    result = {
        "exit_codes": {"A": rc_a, "B_killed": rc_b1, "B_resumed": rc_b2},
        "latest_after_kill": latest_b,
        "resume_event": resume_events[-1] if resume_events else None,
        "steps_compared": len(loss_diff),
        "max_loss_diff_81_120": max(loss_diff) if loss_diff else None,
        "max_loss_diff_A_vs_B_before_kill_1_95": max(pre_diff) if pre_diff else None,
        "max_loss_diff_B_before_kill_vs_resumed_81_95": max(kill_vs_resume) if kill_vs_resume else None,
        "batch_hashes_equal_81_120": hashes_equal,
        "pass": bool(complete and rc_a == 0 and rc_b1 != 0 and rc_b2 == 0 and latest_b == "step_000080"
                     and loss_diff and max(loss_diff) < TOL and hashes_equal),
    }
    plot_resume({s: A[s]["loss"] for s in A}, {s: B[s]["loss"] for s in B},
                {s: BK[s]["loss"] for s in BK}, out / "plots" / "stage3_resume.png")
    for f in ("resume_A.jsonl", "resume_B.jsonl", "resume_B_before_kill.jsonl"):
        shutil.copy(f"logs/{f}", out / f)
    scratch.delete_all()
    result["hf_resume_test_deleted"] = not _list(scratch)
    (out / "stage3.json").write_text(json.dumps(result, indent=1))
    print(f"[stage3] {json.dumps(result, indent=1)}", flush=True)
    if not result["pass"]:
        sys.exit("[stage3] GATE 3 FAILED")


def _list(manager: CheckpointManager) -> list[str]:
    from data import hub                                   # noqa: PLC0415
    return hub.list_files(manager.repo_id, manager.prefix, repo_type="model")


if __name__ == "__main__":
    main()
