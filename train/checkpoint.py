"""Checkpoints that survive Kaggle session limits: local save, HF upload, resume, pruning.

Layout in the HF checkpoints repo (model type), under a run prefix such as pretrain_61m/:
    step_000500/ckpt.pt        full training state (see make_state)
    best/model.pt, best/info.json   model weights + config of the best val loss
    logs/pretrain_61m.jsonl    the append-only training log
    latest.txt                 name of the newest complete step_ folder

A checkpoint, its log and latest.txt are uploaded in ONE commit, so latest.txt can
never point at a half-uploaded checkpoint. Only rank 0 calls the upload methods.
"""

import json
import random
import shutil
import threading
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from data import hub

REPO_TYPE = "model"


def rng_state() -> dict:
    """Python, numpy, torch-CPU and current-CUDA-device RNG states of this rank."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    """Restore the states saved by rng_state()."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def step_name(step: int) -> str:
    """Folder name of a checkpoint after `step` completed optimizer steps."""
    return f"step_{step:06d}"


class CheckpointManager:
    """Save/upload/download/prune checkpoints of one run (one HF prefix)."""

    def __init__(self, repo_id: str, prefix: str, local_dir: str | Path, keep_last: int, log_name: str) -> None:
        self.repo_id = repo_id
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.local_dir = Path(local_dir)
        self.keep_last = keep_last
        self.log_name = log_name
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    # ---------- local ----------
    def save_local(self, step: int, state: dict) -> Path:
        """Write ckpt.pt for this step and delete older local checkpoints (disk is 20 GB)."""
        self.wait()                                        # never delete a file that is still uploading
        out = self.local_dir / step_name(step)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "ckpt.pt.tmp"
        torch.save(state, tmp)
        tmp.rename(out / "ckpt.pt")
        for old in self.local_dir.glob("step_*"):
            if old != out:
                shutil.rmtree(old, ignore_errors=True)
        return out / "ckpt.pt"

    # ---------- background upload ----------
    def upload_async(self, step: int, ckpt_path: Path, log_path: Path | None,
                     best: tuple[dict, dict] | None = None) -> None:
        """Upload in a background thread so training continues (a 735 MB upload takes minutes).

        The log is snapshotted first, because the training loop keeps appending to it.
        Waits for the previous upload before starting; errors surface on the next wait().
        """
        self.wait()
        snap = None
        if log_path is not None and Path(log_path).exists():
            snap = ckpt_path.parent / self.log_name
            shutil.copy(log_path, snap)

        def work() -> None:
            try:
                self.upload(step, ckpt_path, snap)
                if best is not None:
                    self.upload_best(*best)
                print(f"[ckpt] uploaded {step_name(step)}" + (" + best/" if best else ""), flush=True)
            except BaseException as err:  # noqa: BLE001 - re-raised in wait()
                self._error = err

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        """Block until the background upload is done; re-raise its error if it failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err

    # ---------- HF ----------
    def upload(self, step: int, ckpt_path: Path, log_path: Path | None) -> None:
        """One commit: step_XXXXXX/ckpt.pt + the log + latest.txt; then prune old steps."""
        name = step_name(step)
        ops = [
            CommitOperationAdd(f"{self.prefix}{name}/ckpt.pt", str(ckpt_path)),
            CommitOperationAdd(f"{self.prefix}latest.txt", f"{name}\n".encode()),
        ]
        if log_path is not None and Path(log_path).exists():
            ops.append(CommitOperationAdd(f"{self.prefix}logs/{self.log_name}", str(log_path)))
        api = hub.get_api()
        hub._retry(lambda: api.create_commit(self.repo_id, operations=ops, repo_type=REPO_TYPE,
                                             commit_message=f"{self.prefix}{name}"), f"upload {name}")
        self.prune()

    def upload_best(self, model_state: dict, info: dict) -> None:
        """Replace best/ with these weights and their info (step, val loss, config)."""
        tmp = self.local_dir / "best"
        tmp.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, tmp / "model.pt")
        (tmp / "info.json").write_text(json.dumps(info, indent=1))
        hub.upload_folder(self.repo_id, tmp, f"{self.prefix}best", repo_type=REPO_TYPE,
                          message=f"{self.prefix}best (step {info['step']})")

    def upload_log(self, log_path: Path) -> None:
        """Upload just the log (e.g. after a final eval)."""
        hub.upload_file(self.repo_id, log_path, f"{self.prefix}logs/{self.log_name}", repo_type=REPO_TYPE)

    def prune(self) -> None:
        """Delete all but the newest keep_last step_ folders on HF."""
        files = hub.list_files(self.repo_id, self.prefix, repo_type=REPO_TYPE)
        steps = sorted({f[len(self.prefix):].split("/")[0] for f in files
                        if f[len(self.prefix):].startswith("step_")})
        old = steps[:-self.keep_last] if self.keep_last > 0 else steps
        if not old:
            return
        ops = [CommitOperationDelete(f"{self.prefix}{s}/", is_folder=True) for s in old]
        api = hub.get_api()
        hub._retry(lambda: api.create_commit(self.repo_id, operations=ops, repo_type=REPO_TYPE,
                                             commit_message=f"prune {', '.join(old)}"), "prune")

    def latest(self) -> str | None:
        """Name in latest.txt, or None if this run has no checkpoint yet."""
        if f"{self.prefix}latest.txt" not in hub.list_files(self.repo_id, self.prefix, repo_type=REPO_TYPE):
            return None
        path = hub.download(self.repo_id, f"{self.prefix}latest.txt", self.local_dir / "_dl", repo_type=REPO_TYPE)
        return Path(path).read_text().strip()

    def download(self, name: str) -> tuple[Path, Path | None]:
        """Download step folder `name` and the run log; return (ckpt path, log path or None)."""
        dl = self.local_dir / "_dl"
        ckpt = Path(hub.download(self.repo_id, f"{self.prefix}{name}/ckpt.pt", dl, repo_type=REPO_TYPE))
        log_remote = f"{self.prefix}logs/{self.log_name}"
        log = None
        if log_remote in hub.list_files(self.repo_id, self.prefix, repo_type=REPO_TYPE):
            log = Path(hub.download(self.repo_id, log_remote, dl, repo_type=REPO_TYPE))
        return ckpt, log

    def delete_all(self) -> None:
        """Remove this run's whole prefix from HF (used for scratch runs)."""
        api = hub.get_api()
        hub._retry(lambda: api.delete_folder(self.prefix.rstrip("/"), repo_id=self.repo_id, repo_type=REPO_TYPE,
                                             commit_message=f"delete {self.prefix}"), "delete")
