"""Stage 4 driver: sample -> train 8 tokenizers -> morph words -> evaluate -> select.

    python -m tokenizer.sweep

Each step is resumable through the Hub (sample and trained models are uploaded as they finish).
Outputs land in runtime.out_dir: tokenizer_sweep.csv, plots/, tokenizer_examples.md,
stage4_selection.md, morph_words.tsv, and the chosen .model/.vocab.
"""

import json
import shutil
from pathlib import Path

from data import hub
from data.config import REPO_ROOT, load_config
from tokenizer import evaluate, morph_words, sample, select, train


def main():
    cfg = load_config("configs/tokenizer.yaml")
    out_dir = Path(cfg["runtime"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path, val_path, sample_info = sample.fetch_or_build(cfg)
    print(f"[sweep] sample: {sample_info}", flush=True)
    infos = train.train_grid(cfg, sample_path)
    morph_words.build(cfg, sample_path)
    rows, eval_info = evaluate.evaluate_all(cfg, infos, val_path)
    evaluate.write_outputs(cfg, rows, out_dir)

    chosen, log = select.select(rows, cfg["selection"])
    failed = [i for i in infos if "error" in i]
    header = ["# Stage 4 selection", "", f"Sample: `{json.dumps(sample_info)}`",
              f"Eval: `{json.dumps(eval_info)}`", ""]
    if failed:
        header += ["Failed trainings:", ""] + [f"- {i['name']}: {i['error']}" for i in failed] + [""]
    (out_dir / "stage4_selection.md").write_text("\n".join(header) + log, encoding="utf-8")
    print(log, flush=True)

    shutil.copy(REPO_ROOT / cfg["morphology"]["out_path"], out_dir / "morph_words.tsv")
    model_dir = REPO_ROOT / cfg["sweep"]["out_dir"]
    for ext in (".model", ".vocab"):
        shutil.copy(model_dir / f"{chosen['name']}{ext}", out_dir / f"{chosen['name']}{ext}")
    recommended = out_dir / "recommended.json"
    recommended.write_text(json.dumps({"name": chosen["name"], "metrics": chosen}, indent=1))
    hub.upload_file(cfg["hf"]["tokenizer_repo"], recommended, "recommended.json", repo_type="model")


if __name__ == "__main__":
    main()
