"""Stage 7a (D8): inspect the frozen base tokenizer and show why Latin input can't use it.

    python -m tokenizer.stage7a [--out /kaggle/working/out/d7]

Writes stage7a_inspect.json: model facts (sha256, type, vocab, byte_fallback,
character_coverage, special IDs), the base pieces containing ASCII letters, the probe
table (pieces, count, unk and byte pieces per probe) and the HinglishTokenizer numbers
(vocab size, allowed output IDs). Needs sentencepiece; changes nothing.
"""

import argparse
import json
import re
from pathlib import Path

from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer

PROBES = [
    "kal meeting cancel ho gayi kya",
    "main ghar ja raha hu",
    "mai ghr jaa rha hun",
    "tum kaha ho?",
    "wo aayi ki nahi",
    "rahul ki kitab",
    "mujhe 7 baje uthna h",
    "kya baat hai yaar!!",
    "Office mein Wi-Fi nahi chal raha",
    "aaj bahut garmi hai",
    "आज बहुत गर्मी है",
]


def trainer_spec(sp) -> dict:
    """model_type, vocab_size, byte_fallback, character_coverage, ... from the serialized proto."""
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(sp.serialized_model_proto())
    t = m.trainer_spec
    return {"model_type": pb.TrainerSpec.ModelType.Name(t.model_type), "vocab_size": t.vocab_size,
            "byte_fallback": t.byte_fallback, "character_coverage": t.character_coverage,
            "split_digits": t.split_digits, "normalization_rule_name": m.normalizer_spec.name,
            "unk_id": t.unk_id, "bos_id": t.bos_id, "eos_id": t.eos_id, "pad_id": t.pad_id}


def inspect(tok: HinglishTokenizer) -> dict:
    """All Stage 7a facts as one dict."""
    sp = tok.sp
    from sentencepiece import sentencepiece_model_pb2 as pb

    m = pb.ModelProto()
    m.ParseFromString(sp.serialized_model_proto())
    kinds = {"control": [], "user_defined": [], "byte": [], "unknown": []}
    for i, piece in enumerate(m.pieces):
        kind = pb.ModelProto.SentencePiece.Type.Name(piece.type).lower()
        if kind in kinds:
            kinds[kind].append(i)
    special = set(kinds["control"]) | set(kinds["user_defined"]) | set(kinds["byte"]) | set(kinds["unknown"])
    latin_normal = [i for i in range(sp.get_piece_size()) if i not in special
                    and re.search(r"[A-Za-z]", sp.id_to_piece(i))]
    latin_bytes = [i for i in kinds["byte"] if tok.base_piece_has_ascii_letter(i)]
    probes = []
    for text in PROBES:
        ids = sp.encode(text)
        probes.append({"text": text, "pieces": [sp.id_to_piece(i) for i in ids], "n_pieces": len(ids),
                       "n_unk": sum(sp.is_unknown(i) for i in ids), "n_byte": sum(sp.is_byte(i) for i in ids),
                       "chars": len(text), "rom_ids_len": len(tok.encode_rom(text))})
    allowed = tok.output_allowed_ids()
    return {
        "model_path": str(tok.base_model), "sha256": tok.base_sha256, "trainer_spec": trainer_spec(sp),
        "piece_size": sp.get_piece_size(),
        "ids": {"unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id(), "pad": sp.pad_id()},
        "pieces_0_25": [sp.id_to_piece(i) for i in range(25)],
        "kind_counts": {k: len(v) for k, v in kinds.items()},
        "latin_pieces_normal": {"count": len(latin_normal), "examples": [sp.id_to_piece(i) for i in latin_normal[:30]]},
        "latin_pieces_byte": {"count": len(latin_bytes)},
        "latin_pieces_naive_regex": sum(bool(re.search(r"[A-Za-z]", sp.id_to_piece(i)))
                                        for i in range(sp.get_piece_size())),
        "probes": probes,
        "hinglish": {"vocab_size": tok.vocab_size, "n_new": tok.vocab_size - tok.base_vocab_size,
                     "rom_id": tok.rom_id, "dev_id": tok.dev_id, "eos_id": tok.eos_id,
                     "n_output_allowed": len(allowed), "n_output_banned": tok.base_vocab_size - len(allowed)},
    }


def main() -> None:
    """Run the inspection and write stage7a_inspect.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/out/d7")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = inspect(HinglishTokenizer.load(DEFAULT_EXT))
    (out / "stage7a_inspect.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
