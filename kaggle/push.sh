#!/usr/bin/env bash
# Push one pipeline stage to a Kaggle kernel of prateek99491 (default: the CPU kernel hindi-lm-runner).
#
#   kaggle/push.sh "python -m data.pipeline stage1"
#   kaggle/push.sh "python -m tokenizer.sweep" hindi-lm-tpu     # the TPU-VM kernel (big RAM)
#   EMBED_CODE=1 kaggle/push.sh "python -m model.model" d6_tests # code embedded in run.py; no HF_TOKEN needed
#
# 1. Verify the pinned Kaggle account is exactly prateek99491 (else stop).
# 2. Upload the current code tree to the private HF repo under code/ (or, with EMBED_CODE=1,
#    embed it in run.py as a base64 tar.gz so the run starts from an API push without secrets).
# 3. Generate run.py for this stage and push the kernel.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stage_command="$1"
kernel_dir="${repo_root}/kaggle/${2:-hindi-lm-runner}"
template="${repo_root}/kaggle/hindi-lm-runner/run_template.py"   # one template for every kernel
[[ -f "${kernel_dir}/kernel-metadata.json" ]] || { echo "push.sh: no kernel-metadata.json in ${kernel_dir}" >&2; exit 1; }

user="$("${repo_root}/kaggle/k.sh" config view | awk -F': ' '/username/ {print $2}')"
if [[ "${user}" != "prateek99491" ]]; then
    echo "push.sh: Kaggle account is '${user}', not prateek99491. Not pushing." >&2
    exit 1
fi
echo "Kaggle account verified: ${user}"

if [[ "${EMBED_CODE:-0}" != "1" ]]; then
"${repo_root}/.venv/bin/python" - "${repo_root}" <<'PY'
import shutil, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi

root = Path(sys.argv[1])
keep = ["pyproject.toml", "README.md", "configs", "data", "tokenizer", "tests", "model", "train", "eval"]
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
fi

python3 - "${kernel_dir}" "${stage_command}" "${template}" "${repo_root}" "${EMBED_CODE:-0}" <<'PY'
import base64, io, sys, tarfile
from pathlib import Path
kernel_dir, command, template, root, embed = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]).read_text(), Path(sys.argv[4]), sys.argv[5] == "1"
code_b64 = ""
if embed:
    keep = ["pyproject.toml", "README.md", "configs", "data", "tokenizer", "tests", "model", "train", "eval"]
    skip_dirs = {"__pycache__", "raw", "work", "out", "_hub"}
    skip_ext = {".bin", ".parquet", ".gz", ".model", ".vocab"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in keep:
            for path in sorted([root / name] if (root / name).is_file() else (root / name).rglob("*")):
                rel = path.relative_to(root)
                if path.is_file() and not skip_dirs & set(rel.parts) and path.suffix not in skip_ext:
                    tar.add(path, arcname=str(rel))
    code_b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"Code embedded in run.py ({len(code_b64) / 1024:.0f} KB base64)")
run = template.replace("__STAGE_COMMAND__", command).replace("__CODE_B64__", code_b64)
(kernel_dir / "run.py").write_text(run)
PY

"${repo_root}/kaggle/k.sh" kernels push -p "${kernel_dir}"
