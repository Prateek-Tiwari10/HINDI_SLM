"""D8: Hinglish tokenizer = the frozen base SentencePiece model + a character-level Latin table.

Base IDs 0..10239 are the unchanged `hi_unigram_10240` pieces, so the pretrained weights
stay valid. 64 new IDs are appended from 10240 (see TABLE): <rom>/<dev> markers, a-z,
0-9, 12 punctuation marks, <lsp> (Latin space), <lunk> (any other Latin-side character)
and 12 never-used <reserved_i> rows that pad the vocab to 10,304 = 64 × 161.

Training pair (build_pair):
    input_ids = [<rom>] + rom_ids + [<dev>] + dev_ids + [eos]
    labels    = -100 up to and including <dev>, then dev_ids + [eos]
labels are aligned with input_ids, not pre-shifted. HindiLlama.forward has no internal
shift; pretraining feeds x = w[:, :-1], y = w[:, 1:] (train/data.py), and fine-tuning
does the same with x = input_ids[:-1], y = labels[1:]. The logits at <dev> then predict
the first Devanagari token. -100 is F.cross_entropy's default ignore_index.

The Latin side is normalized by normalize_rom (NFKC, lowercase, strip accents, collapse
whitespace); the Devanagari side by data.normalize.normalize, exactly as in pretraining.
"""

import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path

from data.normalize import normalize

VERSION = "hinglish-v1"
BASE_VOCAB = 10240
IGNORE = -100
LETTERS = list(string.ascii_lowercase)
DIGITS = list(string.digits)
PUNCT = list(".,?!'\"-:;()/")
TABLE = (["<rom>", "<dev>"] + LETTERS + DIGITS + PUNCT + ["<lsp>", "<lunk>"]
         + [f"<reserved_{i}>" for i in range(12)])          # 64 entries -> IDs 10240..10303
LUNK_TEXT = "�"                                         # how decode_rom shows <lunk>
REPO_ROOT = Path(__file__).resolve().parent.parent
_ASCII_LETTER_BYTES = set(range(0x41, 0x5B)) | set(range(0x61, 0x7B))


def normalize_rom(text: str) -> str:
    """NFKC -> lowercase -> strip accents (NFD, drop combining marks) -> collapse whitespace -> strip."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root when inside it, so tokenizer_ext.json works on any machine."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class HinglishTokenizer:
    """Encode romanized input and Devanagari output into one 10,304-token ID space.

    The SentencePiece processor is loaded lazily, so the table, save() and the Latin side
    work without sentencepiece installed.
    """

    def __init__(self, base_model: str | Path, expected_sha256: str | None = None) -> None:
        self.base_model = Path(base_model)
        self.base_sha256 = sha256_file(self.base_model)
        if expected_sha256 is not None and expected_sha256 != self.base_sha256:
            raise ValueError(f"base tokenizer sha256 mismatch for {self.base_model}: "
                             f"file is {self.base_sha256}, tokenizer_ext.json expects {expected_sha256}")
        self._sp = None
        self.token_to_id = {tok: BASE_VOCAB + i for i, tok in enumerate(TABLE)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    # ---------- base model ----------
    @property
    def sp(self):
        """The base SentencePiece processor (hi_unigram_10240), loaded on first use."""
        if self._sp is None:
            import sentencepiece as spm

            self._sp = spm.SentencePieceProcessor(model_file=str(self.base_model))
            assert self._sp.get_piece_size() == BASE_VOCAB, f"base vocab is {self._sp.get_piece_size()}, not {BASE_VOCAB}"
        return self._sp

    @property
    def base_vocab_size(self) -> int:
        """10240: IDs below this are base SentencePiece pieces."""
        return BASE_VOCAB

    @property
    def vocab_size(self) -> int:
        """10304: base + the 64-entry Latin table."""
        return BASE_VOCAB + len(TABLE)

    @property
    def version(self) -> str:
        """Extension format version stored in tokenizer_ext.json and fine-tune checkpoints."""
        return VERSION

    @property
    def rom_id(self) -> int:
        """ID of <rom>, which opens the romanized input."""
        return self.token_to_id["<rom>"]

    @property
    def dev_id(self) -> int:
        """ID of <dev>, which separates input from the Devanagari target."""
        return self.token_to_id["<dev>"]

    @property
    def eos_id(self) -> int:
        """The base eos (</s>, 3): the document-end token of pretraining, reused as end of output."""
        return self.sp.eos_id()

    # ---------- encode / decode ----------
    def encode_rom(self, text: str) -> list[int]:
        """Normalize, then one ID per character; space -> <lsp>, anything outside the table -> <lunk>."""
        lsp, lunk = self.token_to_id["<lsp>"], self.token_to_id["<lunk>"]
        out = []
        for ch in normalize_rom(text):
            if ch == " ":
                out.append(lsp)
            elif ch in self.token_to_id and not ch.startswith("<"):
                out.append(self.token_to_id[ch])
            else:
                out.append(lunk)
        return out

    def encode_dev(self, text: str) -> list[int]:
        """Base SentencePiece IDs of the pretraining-normalized text (no bos/eos)."""
        return self.sp.encode(normalize(text))

    def build_pair(self, rom: str, dev: str) -> tuple[list[int], list[int]]:
        """(input_ids, labels) for one training pair; see the module docstring for the layout."""
        rom_ids, dev_ids = self.encode_rom(rom), self.encode_dev(dev)
        input_ids = [self.rom_id] + rom_ids + [self.dev_id] + dev_ids + [self.eos_id]
        labels = [IGNORE] * (len(rom_ids) + 2) + dev_ids + [self.eos_id]
        return input_ids, labels

    def decode_dev(self, ids: list[int]) -> str:
        """Devanagari text: drops IDs >= 10240 and every base control / special piece."""
        sp = self.sp
        keep = [i for i in ids if i < BASE_VOCAB and not (sp.is_control(i) or sp.is_unknown(i)
                                                          or sp.id_to_piece(i).startswith("<|"))]
        return sp.decode(keep)

    def decode_rom(self, ids: list[int]) -> str:
        """Latin text back from table IDs (debugging); <lunk> shows as U+FFFD, other IDs are skipped."""
        out = []
        for i in ids:
            tok = self.id_to_token.get(i)
            if tok == "<lsp>":
                out.append(" ")
            elif tok == "<lunk>":
                out.append(LUNK_TEXT)
            elif tok is not None and not tok.startswith("<"):
                out.append(tok)
        return "".join(out)

    def base_piece_has_ascii_letter(self, i: int) -> bool:
        """True if base piece i would print ASCII letters.

        Byte-fallback pieces (<0xNN>) are judged by the byte they stand for, not by their
        name: <0x61> ('a') counts, <0xE0> (a UTF-8 lead byte of Devanagari) does not.
        Control and chat/reserved pieces are judged by name (they all contain letters).
        """
        sp = self.sp
        if sp.is_byte(i):
            return int(sp.id_to_piece(i)[1:-1], 16) in _ASCII_LETTER_BYTES
        return re.search(r"[A-Za-z]", sp.id_to_piece(i)) is not None

    def output_allowed_ids(self) -> list[int]:
        """Base IDs the fine-tuned model may emit: no unk/bos/pad, no piece with ASCII letters; plus eos."""
        sp = self.sp
        banned = {sp.unk_id(), sp.bos_id(), sp.pad_id()}
        allowed = [i for i in range(BASE_VOCAB) if i not in banned and not self.base_piece_has_ascii_letter(i)]
        if self.eos_id not in allowed:
            allowed.append(self.eos_id)
        return sorted(allowed)

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        """Everything needed to rebuild this tokenizer and check it against the base model."""
        return {
            "version": VERSION,
            "base_model": _repo_relative(self.base_model),
            "base_sha256": self.base_sha256,
            "base_vocab_size": BASE_VOCAB,
            "vocab_size": self.vocab_size,
            "table": self.token_to_id,
            "special_ids": {"rom": self.rom_id, "dev": self.dev_id, "eos": 3, "bos": 2, "pad": 0, "unk": 1,
                            "lsp": self.token_to_id["<lsp>"], "lunk": self.token_to_id["<lunk>"]},
            "normalization": {"rom": "NFKC, lowercase, strip accents (NFD, drop combining), collapse whitespace, strip",
                              "dev": "data.normalize.normalize (pretraining)"},
        }

    def save(self, path: str | Path) -> None:
        """Write tokenizer_ext.json."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path, base_model: str | Path | None = None) -> "HinglishTokenizer":
        """Rebuild from tokenizer_ext.json; raise if the base model's sha256 or the table differs.

        base_model overrides the stored path (e.g. a copy downloaded on Kaggle); the
        sha256 check still applies.
        """
        meta = json.loads(Path(path).read_text())
        stored = Path(meta["base_model"])
        if not stored.is_absolute():
            stored = REPO_ROOT / stored
        if meta.get("version") != VERSION:
            raise ValueError(f"tokenizer_ext version {meta.get('version')!r}, expected {VERSION!r}")
        tok = cls(base_model or stored, expected_sha256=meta["base_sha256"])
        if meta["table"] != tok.token_to_id or meta["vocab_size"] != tok.vocab_size:
            raise ValueError("tokenizer_ext.json table does not match this code's TABLE")
        return tok


DEFAULT_EXT = Path(__file__).with_name("tokenizer_ext.json")
DEFAULT_BASE = Path(__file__).with_name("hi_unigram_10240.model")
