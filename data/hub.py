"""Thin wrapper over the Hugging Face Hub: our private repos are the pipeline's storage.

Done-markers live on the Hub too, because Kaggle sessions lose their disk when they end.
The token comes from the HF_TOKEN environment variable (Kaggle Secret) or the local login.
It is never printed.
"""

import os
import tempfile
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

RETRIES = 8
MAX_WAIT_SECONDS = 300  # HF commit rate limits can need minutes, not seconds


def get_api():
    """HfApi authenticated from HF_TOKEN if set, else from the cached login."""
    return HfApi(token=os.environ.get("HF_TOKEN"))


def _retry(fn, what):
    """Call fn() with exponential backoff; network hiccups must not kill a 10-hour job."""
    for attempt in range(RETRIES):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - any network error is retried
            if attempt == RETRIES - 1:
                raise
            wait = min(MAX_WAIT_SECONDS, 2 ** attempt * 5)
            print(f"[hub] {what} failed ({type(err).__name__}: {err}); retry in {wait}s", flush=True)
            time.sleep(wait)


def list_files(repo_id, prefix="", repo_type="dataset"):
    """All file paths in a repo that start with prefix."""
    api = get_api()
    files = _retry(lambda: api.list_repo_files(repo_id, repo_type=repo_type), "list")
    return sorted(f for f in files if f.startswith(prefix))


def download(repo_id, path_in_repo, local_dir, repo_type="dataset"):
    """Download one file and return its local path."""
    return _retry(
        lambda: hf_hub_download(
            repo_id, path_in_repo, repo_type=repo_type, local_dir=local_dir,
            token=os.environ.get("HF_TOKEN"),
        ),
        f"download {path_in_repo}",
    )


def upload_file(repo_id, local_path, path_in_repo, repo_type="dataset", message=None):
    """Upload one file (one commit)."""
    api = get_api()
    _retry(
        lambda: api.upload_file(
            path_or_fileobj=str(local_path), path_in_repo=path_in_repo,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=message or f"Upload {path_in_repo}",
        ),
        f"upload {path_in_repo}",
    )


def upload_folder(repo_id, local_dir, path_in_repo, repo_type="dataset", message=None):
    """Upload a whole folder as one atomic commit."""
    api = get_api()
    _retry(
        lambda: api.upload_folder(
            folder_path=str(local_dir), path_in_repo=path_in_repo,
            repo_id=repo_id, repo_type=repo_type,
            commit_message=message or f"Upload {path_in_repo}",
        ),
        f"upload folder {path_in_repo}",
    )


def upload_folder_with_marker(repo_id, local_dir, path_in_repo, marker_path, repo_type="dataset", message=None):
    """Upload a folder and its done-marker in ONE commit.

    A Hub commit is atomic: either every file lands, or none does. So the
    marker exists only if the data upload succeeded. That is the done-marker
    rule, with half the commits (HF rate-limits commits per hour).
    """
    local_dir = Path(local_dir)
    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    api = get_api()

    def commit():
        # Fresh operation objects per attempt: a failed attempt leaves state on them.
        ops = [
            CommitOperationAdd(path_in_repo=f"{path_in_repo}/{f.relative_to(local_dir).as_posix()}",
                               path_or_fileobj=str(f))
            for f in files
        ]
        ops.append(CommitOperationAdd(path_in_repo=marker_path, path_or_fileobj=b"done\n"))
        api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=ops,
                          commit_message=message or f"Upload {path_in_repo}")

    _retry(commit, f"commit {path_in_repo}")


def write_marker(repo_id, marker_path, repo_type="dataset"):
    """Write an empty done-marker. Call only after the real upload succeeded."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "done"
        local.write_text("done\n")
        upload_file(repo_id, local, marker_path, repo_type, message=f"Mark {marker_path}")


def done_markers(repo_id, prefix, suffix=".done", repo_type="dataset"):
    """Set of marker paths under prefix, for skipping finished shards on a rerun."""
    return {f for f in list_files(repo_id, prefix, repo_type) if f.endswith(suffix)}
