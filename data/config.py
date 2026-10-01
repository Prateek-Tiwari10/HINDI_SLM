"""Load YAML configs. Every script reads its numbers from here, never from constants."""

import hashlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    """Read a YAML config file into a dict. Relative paths resolve from the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(path):
    """Short sha256 of a config file's bytes, recorded in outputs for reproducibility."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
