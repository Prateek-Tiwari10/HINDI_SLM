"""One reader per source. Every reader yields {"id": "<source>:<native_id>", "source", "text"}.

Sources are read file by file: a parquet (or .jsonl.zst) file is the unit of work,
so a long job can resume at file granularity.
- stream=True reads over HTTP with range requests (only the row groups we touch),
  used for small samples.
- stream=False downloads the file first, used by the full pipeline because it
  retries more reliably.
"""

import io
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
import zstandard
from huggingface_hub import HfFileSystem

from data import hub

BATCH_ROWS = 2_000


def _natural_key(path):
    """Sort 'data-2.parquet' before 'data-10.parquet'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path)]


def _check_allowed(path):
    """Sangraha Synthetic is machine-translated and excluded (D2). Refuse it outright."""
    if "synthetic" in path.lower():
        raise ValueError(f"Refusing to read excluded source file: {path}")


def list_source_files(src_cfg):
    """Input files for one source, in a fixed order."""
    files = hub.list_files(src_cfg["hf_repo"], src_cfg["prefix"])
    files = [f for f in files if f.endswith(src_cfg["file_suffix"])]
    for f in files:
        _check_allowed(f)
    return sorted(files, key=_natural_key)


def _hf_uri(src_cfg, path):
    return f"datasets/{src_cfg['hf_repo']}/{path}"


def _open_remote(src_cfg, path, stream, local_dir):
    """Binary file handle for a source file: remote (streamed) or downloaded."""
    if stream:
        fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
        return fs.open(_hf_uri(src_cfg, path), "rb", block_size=8 * 1024 * 1024)
    local = hub.download(src_cfg["hf_repo"], path, local_dir)
    return open(local, "rb")


def _iter_parquet_rows(handle, id_field, text_field):
    """(native_id, text) pairs from a parquet file, batch by batch."""
    parquet = pq.ParquetFile(handle)
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=[id_field, text_field]):
        ids = batch.column(id_field).to_pylist()
        texts = batch.column(text_field).to_pylist()
        yield from zip(ids, texts)


def _iter_jsonl_zst_rows(handle, id_field, text_field):
    """(native_id, text) pairs from a zstd-compressed JSON-lines file."""
    reader = zstandard.ZstdDecompressor().stream_reader(handle)
    for line in io.TextIOWrapper(reader, encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        yield record[id_field], record[text_field]


def iter_file(name, src_cfg, path, stream=False, local_dir="/tmp/hindi-lm/src"):
    """Yield normalized-schema docs from one source file. Deletes the download when done."""
    _check_allowed(path)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    handle = _open_remote(src_cfg, path, stream, local_dir)
    try:
        if src_cfg["format"] == "parquet":
            rows = _iter_parquet_rows(handle, src_cfg["id_field"], src_cfg["text_field"])
        elif src_cfg["format"] == "jsonl.zst":
            rows = _iter_jsonl_zst_rows(handle, src_cfg["id_field"], src_cfg["text_field"])
        else:
            raise ValueError(f"Unknown format {src_cfg['format']}")
        for native_id, text in rows:
            if text:
                yield {"id": f"{name}:{native_id}", "source": name, "text": text}
    finally:
        handle.close()
        if not stream:
            local = Path(local_dir) / path
            if local.exists():
                local.unlink()


def enabled_sources(data_cfg):
    """(name, cfg) for every source we are allowed to read, in priority order."""
    sources = [
        (name, cfg) for name, cfg in data_cfg["sources"].items()
        if name != "excluded" and cfg.get("enabled", True)
    ]
    return sorted(sources, key=lambda item: item[1]["priority"])


def count_parquet_rows(src_cfg, path):
    """Row count from the parquet footer only (a few KB read), for size estimates."""
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    with fs.open(_hf_uri(src_cfg, path), "rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows
