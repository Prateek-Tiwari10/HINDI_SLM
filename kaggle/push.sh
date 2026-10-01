#!/usr/bin/env bash
# Push one pipeline stage to the Kaggle CPU kernel prateek99491/hindi-lm-runner.
#
#   kaggle/push.sh "python -m data.pipeline stage1"
#
# 1. Verify the pinned Kaggle account is exactly prateek99491 (else stop).
# 2. Upload the current code tree to the private HF repo under code/.
# 3. Generate run.py for this stage and push the kernel.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_dir="${repo_root}/kaggle/hindi-lm-runner"
stage_command="$1"

user="$("${repo_root}/kaggle/k.sh" config view | awk -F': ' '/username/ {print $2}')"
if [[ "${user}" != "prateek99491" ]]; then
    echo "push.sh: Kaggle account is '${user}', not prateek99491. Not pushing." >&2
    exit 1
fi
echo "Kaggle account verified: ${user}"

"${repo_root}/.venv/bin/python" - "${repo_root}" <<'PY'
import shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi

root = Path(sys.argv[1])
keep = ["pyproject.toml", "README.md", "configs", "data", "tokenizer", "tests"]
with tempfile.TemporaryDirectory() as tmp:
    for name in keep:
        src = root / name
        if src.is_dir():
            shutil.copytree(src, Path(tmp) / name, ignore=shutil.ignore_patterns(
                "__pycache__", "raw", "work", "out", "_hub", "*.bin", "*.parquet", "*.jsonl.gz"))
        else:
            shutil.copy(src, Path(tmp) / name)
    HfApi().upload_folder(folder_path=tmp, path_in_repo="code", repo_id="Prateek-Tiwari10/hindi-lm-data",
                          repo_type="dataset", delete_patterns="*", commit_message="Sync code snapshot")
print("Code uploaded to HF code/")
PY

python3 - "${kernel_dir}" "${stage_command}" <<'PY'
import sys
from pathlib import Path
kernel_dir, command = Path(sys.argv[1]), sys.argv[2]
template = (kernel_dir / "run_template.py").read_text()
(kernel_dir / "run.py").write_text(template.replace("__STAGE_COMMAND__", command))
PY

"${repo_root}/kaggle/k.sh" kernels push -p "${kernel_dir}"
