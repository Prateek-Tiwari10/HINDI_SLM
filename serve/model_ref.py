"""Which model the server runs: ONE env var, MODEL_ID = "<hf repo>@<revision>".

The default is the pinned line in serve/MODEL_ID, so swapping in a retrained model is a one-line change.
The Docker build downloads that revision into the HF cache (python -m serve.model_ref); at runtime the
server reads it from the cache and only downloads if MODEL_ID was overridden to something not baked in.
"""

import os
from pathlib import Path

PIN_FILE = Path(__file__).with_name("MODEL_ID")


def model_id() -> str:
    return (os.environ.get("MODEL_ID") or PIN_FILE.read_text()).strip()


def split_model_id(ref: str) -> tuple[str, str | None]:
    repo, _, rev = ref.partition("@")
    if not repo or "/" not in repo:
        raise ValueError(f"MODEL_ID must look like 'owner/repo@revision', got {ref!r}")
    return repo, rev or None


def model_dir(ref: str | None = None, allow_download: bool = True) -> Path:
    """Local snapshot of the release: from the HF cache first, downloaded only if missing."""
    from huggingface_hub import snapshot_download

    repo, rev = split_model_id(ref or model_id())
    try:
        return Path(snapshot_download(repo, revision=rev, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached (e.g. MODEL_ID overridden at runtime)
        if not allow_download:
            raise
        return Path(snapshot_download(repo, revision=rev))


if __name__ == "__main__":                       # Docker build step: bake the pinned weights into the image
    print(model_dir(allow_download=True))
