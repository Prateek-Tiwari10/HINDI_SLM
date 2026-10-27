"""Kaggle kernel (CPU runner, GPU or TPU VM): fetch the repo code, install it, run one stage.

Generated from run_template.py by kaggle/push.sh; STAGE_COMMAND is filled in there.
The code comes from the private HF repo (code/) until the public GitHub repo exists;
it is uploaded by push.sh right before each push, so it matches the local tree.
With EMBED_CODE=1, push.sh instead embeds the code tree in CODE_B64 (a base64 tar.gz),
so stages that never touch HF run from an API push without the HF_TOKEN secret.
"""

import base64
import io
import os
import subprocess
import sys
import tarfile
import time

STAGE_COMMAND = "__STAGE_COMMAND__"
CODE_B64 = "__CODE_B64__"
CODE_REPO = "Prateek-Tiwari10/hindi-lm-data"
CODE_DIR = "/kaggle/working/hindi-lm"


def sh(cmd, **kwargs):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def main():
    os.environ["RUN_STARTED_AT"] = str(time.time())   # train.py measures its wall-clock budget from here
    sh("nproc; free -g; df -h /tmp /kaggle/working; python3 --version; nvidia-smi || true")
    from kaggle_secrets import UserSecretsClient

    # Token goes only into the environment of child processes; never printed.
    try:
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 - API-pushed versions cannot read secrets
        if not CODE_B64:
            raise
        print("[runner] HF_TOKEN not available (API-pushed version); running with embedded code only", flush=True)

    # Progress bars flood the Kaggle log; keep it to our own [stage] lines.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    if CODE_B64:
        sh(f"rm -rf {CODE_DIR} && mkdir -p {CODE_DIR}")
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz") as tar:
            tar.extractall(CODE_DIR)
    else:
        # Use Kaggle's preinstalled huggingface_hub (<2.0); upgrading it breaks datasets/transformers.
        from huggingface_hub import snapshot_download

        snapshot_download(CODE_REPO, repo_type="dataset", allow_patterns=["code/**"],
                          local_dir="/kaggle/working/_code", token=os.environ["HF_TOKEN"])
        sh(f"rm -rf {CODE_DIR} && cp -r /kaggle/working/_code/code {CODE_DIR} && rm -rf /kaggle/working/_code")
    sh(f"{sys.executable} -m pip install -q -e {CODE_DIR}")
    sh(f"{sys.executable} -m pytest -v {CODE_DIR}/tests", cwd=CODE_DIR)
    sh(STAGE_COMMAND.replace("python ", f"{sys.executable} ", 1), cwd=CODE_DIR)


if __name__ == "__main__":
    main()
