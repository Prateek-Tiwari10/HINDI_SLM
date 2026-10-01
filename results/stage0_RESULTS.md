# Stage 0 — Environment check and scaffold

**Status:** done, except the git remote, which Prateek will set up.

## Environment

| Item | Value |
| --- | --- |
| Python (laptop) | 3.14.4 (system). A `.venv/` holds only the `kaggle` and `huggingface_hub` CLIs. |
| Disk free | 204 GB of 457 GB |
| RAM | 14 GiB total, ~3 GiB available at check time |
| CPU cores | 12 |
| Kaggle CLI | 2.2.4 (in `.venv/`) |
| **Kaggle username verified** | **`prateek99491`** (`kaggle/k.sh config view`) |
| **HF user verified** | **`Prateek-Tiwari10`** (`hf auth whoami`) |
| Git repo | Not yet; Prateek will initialize the repo and set the remote |

## Compute rule (Prateek, 2026-10-01)

- Nothing heavy runs on the laptop: no tests, samples, filtering, or tokenizer training.
- The laptop only runs light CLI calls: `kaggle/k.sh ...` and `hf ...`.
- All compute runs in Kaggle CPU kernels under `prateek99491`.

## Kaggle account pinning (section 1a)

- **Credentials:** `~/.kaggle/access_token` belongs to `prateek99491`. Prateek identified it, and `config view` confirmed it. The other file, `access_token_tprateek2001`, is never used.
- **Pinned directory:** `~/.kaggle_prateek99491/` (mode 700) holds a *symlink* to that token. The key was not copied or printed.
- **Pin file:** `.kaggle_account_dir`, which is git-ignored, contains the pinned directory path.
- **`kaggle/k.sh`:**
  - It clears `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `KAGGLE_API_TOKEN`.
  - It sets `KAGGLE_CONFIG_DIR` to the pinned directory.
  - It runs `.venv/bin/kaggle`.
  - If the pin file is missing, it refuses to run.
- The global Kaggle config was not changed.

## HF repos

| Repo | Type | Private | Result |
| --- | --- | --- | --- |
| `Prateek-Tiwari10/hindi-lm-data` | dataset | yes | created |
| `Prateek-Tiwari10/hindi-lm-tokenizer` | model | yes | created |

## Scaffold

- **Layout:** created from section 3 at the repo root (`/home/prateek/slm`).
  - `data/` and `tokenizer/` hold stub modules, filled in later stages.
  - `model/`, `train/`, `finetune/`, `eval/`, `serve/`, and `notebooks/` each hold a `.gitkeep`.
- **Setup files:** `pyproject.toml` lists the required dependencies. `.gitignore` covers `data/raw/`, `data/work/`, `*.bin`, `*.parquet`, `*.jsonl.gz`, `.kaggle_account_dir`, and `.venv/`.
- **Skeletons:** `README.md` (What it is, Results, Data, Tokenizer, Reproduce, Links), `DECISIONS.md`, and `data/SOURCES.md`.
- **Configs:** `configs/data.yaml` and `configs/tokenizer.yaml` hold every value from the instruction file.

## Failures / notes

- An early `pip install -e .` on the laptop failed because `fasttext-wheel` has no Python 3.14 wheel. It was stopped when the no-local-compute rule came in.
  - This is harmless: Kaggle runs Python 3.11/3.12, and the install will happen in the kernels.

## Blockers

- **Git remote:** Kaggle kernels `git clone` the public repo, so Stage 1 kernels need it to exist and be pushed. Prateek is handling the git setup.
