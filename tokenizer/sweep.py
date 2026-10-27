"""Stage 4 driver (v2): sample -> ladder -> train 8 tokenizers -> morph words -> evaluate -> select.

    python -m tokenizer.sweep [--max-hours 8.5]

1. Nested samples of train (5%, 15%, 30%; 50% only if needed) and the val docs.
2. Ladder: one unigram-8192 per sample size; the config rule picks the sample size.
3. Grid: {bpe, unigram} x 4 vocab sizes on the chosen sample; the D4 rule picks the tokenizer.
Every trained model is uploaded as it finishes, so a run cut off by the time budget resumes.
Outputs land in runtime.out_dir: tokenizer_ladder.csv, tokenizer_sweep.csv, plots/,
tokenizer_examples.md, stage4_selection.md, morph_words.tsv, and the chosen .model/.vocab.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from data import hub
from data.config import REPO_ROOT, load_config
from tokenizer import evaluate, ladder, morph_words, sample, select, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=8.5, help="stop starting new trainings after this")
    args = parser.parse_args()
    cfg = load_config("configs/tokenizer.yaml")
    out_dir = Path(cfg["runtime"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def before_train(name):
        hours = (time.time() - started) / 3600
        if hours > args.max_hours:
            raise SystemExit(f"INCOMPLETE: {hours:.1f} h used, budget {args.max_hours} h; "
                             f"{name} not started. Run again to resume (finished models are on the Hub).")

    paths, val_path, sample_info = sample.build(cfg)
    print(f"[sweep] sample: {json.dumps(sample_info)}", flush=True)
    morph_words.build(cfg, paths[0])  # counted on the smallest sample, the same list for every model

    k, rung_info, ladder_log = ladder.run(cfg, paths, sample_info, val_path, out_dir, before_train)
    print(ladder_log, flush=True)
    files, size = sample.rung_input(paths, sample_info, k)
    lcfg = cfg["ladder"]
    reuse = {(lcfg["model_type"], lcfg["vocab_size"]): rung_info}
    infos = train.train_grid(cfg, files, size, reuse, before_train)
    rows, eval_info = evaluate.evaluate_all(cfg, infos, val_path)
    evaluate.write_outputs(cfg, rows, out_dir)

    chosen, log = select.select(rows, cfg["selection"])
    failed = [i for i in infos if "error" in i]
    header = ["# Stage 4 selection (v2)", "", f"Sample: `{json.dumps(sample_info)}`",
              f"Eval: `{json.dumps(eval_info)}`", "", ladder_log, "",
              f"## Grid: trained on the {size['pct_of_train_bytes']:.1f}% sample", ""]
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
    hub.upload_file(cfg["hf"]["tokenizer_repo"], recommended, f"{cfg['hf']['sweep_prefix']}recommended.json",
                    repo_type="model")
    print(f"[sweep] done in {(time.time() - started) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
