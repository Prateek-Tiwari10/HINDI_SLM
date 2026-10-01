"""Kaggle CPU kernel: fetch the repo code, install it, run one stage.

Generated from run_template.py by kaggle/push.sh; STAGE_COMMAND is filled in there.
The code comes from the private HF repo (code/) until the public GitHub repo exists;
it is uploaded by push.sh right before each push, so it matches the local tree.
"""

import os
import subprocess
import sys

STAGE_COMMAND = "__STAGE_COMMAND__"
CODE_REPO = "Prateek-Tiwari10/hindi-lm-data"
CODE_DIR = "/kaggle/working/hindi-lm"


def sh(cmd, **kwargs):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def main():
    sh("nproc; free -g; df -h /tmp /kaggle/working")
    from kaggle_secrets import UserSecretsClient

    # Token goes only into the environment of child processes; never printed.
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

    # Progress bars flood the Kaggle log; keep it to our own [stage] lines.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    # Use Kaggle's preinstalled huggingface_hub (<2.0); upgrading it breaks datasets/transformers.
    from huggingface_hub import snapshot_download

    snapshot_download(CODE_REPO, repo_type="dataset", allow_patterns=["code/**"],
                      local_dir="/kaggle/working/_code", token=os.environ["HF_TOKEN"])
    sh(f"rm -rf {CODE_DIR} && cp -r /kaggle/working/_code/code {CODE_DIR} && rm -rf /kaggle/working/_code")
    sh(f"{sys.executable} -m pip install -q -e {CODE_DIR}")
    sh(f"{sys.executable} -m pytest -q {CODE_DIR}/tests", cwd=CODE_DIR)
    sh(STAGE_COMMAND.replace("python ", f"{sys.executable} ", 1), cwd=CODE_DIR)


if __name__ == "__main__":
    main()
