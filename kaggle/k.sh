#!/usr/bin/env bash
# Run the Kaggle CLI pinned to the prateek99491 account.
#
# The laptop has two Kaggle accounts. Instead of touching the global config,
# every call goes through this wrapper, which points KAGGLE_CONFIG_DIR at the
# directory holding prateek99491's credentials. That directory path is read
# from the git-ignored file .kaggle_account_dir at the repo root.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
account_file="${repo_root}/.kaggle_account_dir"

if [[ ! -f "${account_file}" ]]; then
    echo "k.sh: ${account_file} not found; refusing to run with the default account." >&2
    exit 1
fi

config_dir="$(head -n 1 "${account_file}")"
if [[ ! -d "${config_dir}" ]]; then
    echo "k.sh: credentials directory '${config_dir}' does not exist." >&2
    exit 1
fi

# Drop any ambient credentials so only the pinned directory is used.
unset KAGGLE_USERNAME KAGGLE_KEY KAGGLE_API_TOKEN

# Prefer the repo's own venv CLI; fall back to one on PATH.
kaggle_bin="${repo_root}/.venv/bin/kaggle"
if [[ ! -x "${kaggle_bin}" ]]; then
    kaggle_bin="kaggle"
fi

KAGGLE_CONFIG_DIR="${config_dir}" exec "${kaggle_bin}" "$@"
