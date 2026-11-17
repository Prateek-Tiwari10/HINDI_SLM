"""D10: Devanagari sentence -> Hinglish (Latin) romanization, for synthetic training pairs.

Each Devanagari word gets its spelling from, in order:
  1. SHORTCUTS: chat spellings of very common words (है -> hai/h/he), sampled by weight.
  2. the lexicon: human-attested spellings. A word in the Dakshina train lexicon uses only its
     Dakshina spellings; Aksharantar train fills in words Dakshina lacks (filtered, see
     stage7c.build_lexicon). Sampled with p proportional to count^2 (D10, romanize-v2).
  3. rule_romanize(): character mapping with final and medial schwa deletion. Fallback only.
Then one noise level per sentence (clean / mild / heavy) perturbs the spellings.

Digits are kept (Devanagari digits become ASCII, the only digits the Latin table has).
Punctuation in the 7a table passes through, the danda becomes '.' with probability
danda_to_dot_p (human Hinglish ends sentences with '.', see stage7c_RESULTS.md), and any
other character is dropped. The output contains only [a-z0-9], the 7a punctuation and spaces.
"""

import random
import re
import unicodedata
from dataclasses import dataclass, field

from tokenizer.hinglish import PUNCT

GENERATOR_VERSION = "romanize-v2"
SOURCES = ("shortcut", "lexicon", "rules")
NOISE_LEVELS = ("clean", "mild", "heavy")

# ------------------------------------------------------------------ chat shortcuts
# Weighted variants of very common words, as people type them in chat.
SHORTCUTS = {
    "है": {"hai": 6, "h": 2, "he": 2},
    "हैं": {"hain": 5, "hai": 3, "h": 1},
    "नहीं": {"nahi": 5, "nhi": 3, "nahin": 1, "nai": 1},
    "मैं": {"main": 5, "mai": 3, "me": 2},
    "में": {"men": 4, "me": 3, "mein": 3, "main": 1},   # Dakshina dev humans: men 43%, me 28%, mein 26%
    "हूँ": {"hu": 3, "hun": 3, "hoon": 2},
    "हूं": {"hu": 3, "hun": 3, "hoon": 2},
    "क्या": {"kya": 8, "kia": 1},
    "ठीक": {"thik": 4, "theek": 4},
    "भी": {"bhi": 1},
    "तो": {"to": 3, "toh": 1},
    "रहा": {"raha": 6, "rha": 4},
    "रही": {"rahi": 6, "rhi": 4},
    "रहे": {"rahe": 6, "rhe": 4},
    "गया": {"gaya": 7, "gya": 3},
    "गई": {"gayi": 5, "gai": 4, "gyi": 2},
    "कर": {"kar": 8, "kr": 2},
    "और": {"aur": 7, "or": 3},
    "कि": {"ki": 7, "k": 1},
    "यह": {"yeh": 5, "ye": 4, "yah": 1},
    "वह": {"woh": 5, "vo": 3, "wo": 3, "vah": 1},
    "ये": {"ye": 6, "yeh": 3},
    "वो": {"wo": 5, "vo": 4, "woh": 2},
    "मुझे": {"mujhe": 8, "mjhe": 2},
    "आप": {"aap": 7, "ap": 3},
    "हम": {"hum": 8, "ham": 2},
    "कैसे": {"kaise": 8, "kese": 2},
    "क्यों": {"kyu": 4, "kyon": 3, "kyun": 3},
    "बहुत": {"bahut": 6, "bohot": 3, "bhut": 2},
    "अच्छा": {"accha": 5, "acha": 4, "achha": 3},
    "यार": {"yaar": 6, "yar": 4},
}

# ------------------------------------------------------------------ rule fallback tables
CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
NUKTA = "़"
NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y",
                    "र": "r", "न": "n", "ल": "l"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ॠ": "ri", "ऌ": "li",
          "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o"}
MATRAS = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "ॄ": "ri", "ॢ": "li",
          "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o"}
FINAL_SHORT = {"aa": "a", "ee": "i", "oo": "u"}     # word-final long vowels are usually typed short
HALANT, ANUSVARA, CHANDRABINDU, VISARGA, OM = "्", "ं", "ँ", "ः", "ॐ"
LABIALS = {"p", "ph", "b", "bh", "m"}
CLUSTER_KEEPS_SCHWA = {"r", "y", "v", "n"}
DEV_DIGITS = {chr(0x0966 + d): str(d) for d in range(10)}
DANDAS = {"।", "॥"}

# A Devanagari word unit (letters, signs, joiners; no dandas or digits), a digit run, or one other character.
DEV_UNIT = "[ऀ-ॣ॰-ॿ‌‍]"
_PIECES = re.compile(f"({DEV_UNIT}+)|([0-9०-९]+)|(.)")
_CONSONANT_RE = "[bcdfghjklmnpqrstvwxyz]"
# Common verb forms whose first 'a' heavy-noise typists drop: raha -> rha, karna -> krna.
_VERB_DROP = re.compile(rf"^({_CONSONANT_RE}{{1,2}})a({_CONSONANT_RE}[a-z]*)$")
_VERB_ENDINGS = ("na", "ne", "ni", "ta", "te", "ti", "ya", "ye", "yi", "ha", "he", "hi")
_PUNCT = set(PUNCT)


# ------------------------------------------------------------------ rule romanizer
def _aksharas(word):
    """Split a Devanagari word into aksharas: {cons: [latin...], vowel: str|None, inherent: bool, coda: str}.

    vowel None = no vowel (halant, or a deleted schwa). coda holds 'N' (anusvara or
    chandrabindu, resolved to n/m at render time) and 'h' (visarga).
    """
    out, pending_halant = [], False
    chars = unicodedata.normalize("NFD", word)      # precomposed nukta letters -> base + nukta
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in CONSONANTS:
            lat = CONSONANTS[ch]
            if i + 1 < len(chars) and chars[i + 1] == NUKTA:
                lat = NUKTA_CONSONANTS.get(ch, lat)
                i += 1
            if pending_halant and out and out[-1]["cons"]:
                if ch == "ञ" and out[-1]["cons"][-1] == "j":     # ज्ञ -> gy (gyaan)
                    out[-1]["cons"][-1] = "gy"
                else:
                    out[-1]["cons"].append(lat)
                out[-1]["vowel"], out[-1]["inherent"] = "a", True
            else:
                out.append({"cons": [lat], "vowel": "a", "inherent": True, "coda": ""})
            pending_halant = False
        elif ch == HALANT:
            if out and out[-1]["cons"]:
                out[-1]["vowel"], out[-1]["inherent"] = None, False
                pending_halant = True
        elif ch in MATRAS:
            if out and out[-1]["cons"] and not pending_halant:
                out[-1]["vowel"], out[-1]["inherent"] = MATRAS[ch], False
        elif ch in VOWELS:
            out.append({"cons": [], "vowel": VOWELS[ch], "inherent": False, "coda": ""})
            pending_halant = False
        elif ch in (ANUSVARA, CHANDRABINDU):
            if out:
                out[-1]["coda"] += "N"
        elif ch == VISARGA:
            if out:
                out[-1]["coda"] += "h"
        elif ch == OM:
            out.append({"cons": [], "vowel": "om", "inherent": False, "coda": ""})
            pending_halant = False
        # anything else (avagraha, joiners, stray nukta) is skipped
        i += 1
    return out


def _delete_schwas(ak):
    """Final schwa deletion, then the medial V C a C V -> V C C V rule applied right to left."""
    n = len(ak)
    if n >= 2:
        last = ak[-1]
        keep_ya = last["cons"] == ["y"] and ak[-2]["vowel"] in ("i", "ee")       # priya, not priy
        # After a cluster the schwa stays when it ends in r/y/v/n (mitra, karya, krishna), else it
        # goes (dost, film, dharm).
        cluster_keeps = len(last["cons"]) > 1 and last["cons"][-1] in CLUSTER_KEEPS_SCHWA
        if last["inherent"] and not last["coda"] and not keep_ya and not cluster_keeps:
            last["vowel"] = None
    for i in range(n - 2, 0, -1):
        a, prev, nxt = ak[i], ak[i - 1], ak[i + 1]
        if not (a["inherent"] and a["vowel"] == "a" and len(a["cons"]) == 1 and not a["coda"]):
            continue
        if prev["vowel"] is None or len(nxt["cons"]) != 1 or nxt["vowel"] is None:
            continue
        a["vowel"] = None


def _render(ak):
    parts = []
    n = len(ak)
    for i, a in enumerate(ak):
        s = "".join(a["cons"])
        v = a["vowel"]
        if v is not None:
            if not a["cons"] and v == "e" and i > 0 and ak[i - 1]["vowel"] is not None:
                v = "ye"                                   # लिए -> liye, गए -> gaye
            if i == n - 1 and not a["coda"]:
                v = FINAL_SHORT.get(v, v)                  # but माँ -> maan, हूँ -> hoon
            s += v
        for c in a["coda"]:
            if c == "N":
                nxt = ak[i + 1]["cons"][0] if i + 1 < n and ak[i + 1]["cons"] else ""
                s += "m" if nxt in LABIALS else "n"
            else:
                s += c
        parts.append(s)
    return "".join(parts)


def rule_romanize(word):
    """Deterministic rule romanization of one Devanagari word (no noise)."""
    ak = _aksharas(word)
    _delete_schwas(ak)
    return _render(ak)


# ------------------------------------------------------------------ noise
def _mild_op(word, rng):
    """One mild typo on the word (all occurrences): aa->a, ee->i, oo->u, sh->s, w<->v."""
    ops = [(a, b) for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("sh", "s"), ("w", "v"), ("v", "w"))
           if a in word]
    if not ops:
        return word
    a, b = rng.choice(ops)
    return word.replace(a, b)


def apply_noise(word, level, rng, cfg):
    """Perturb one romanized word according to the sentence's noise level."""
    if level == "clean" or not word.isalpha():
        return word
    p = cfg["mild_p"] if level == "mild" else cfg["heavy_p"]
    if rng.random() < p:
        word = _mild_op(word, rng)
    if level == "heavy" and rng.random() < cfg["heavy_verb_drop_p"]:
        m = _VERB_DROP.match(word)
        if m and m.group(2).endswith(_VERB_ENDINGS):
            word = m.group(1) + m.group(2)
    return word


# ------------------------------------------------------------------ sentence romanizer
@dataclass
class Romanization:
    rom: str
    noise_level: str
    tokens: list = field(default_factory=list)     # per whitespace token: (rom_token, [source per word unit])
    units: list = field(default_factory=list)      # per word unit: (devanagari, spelling before noise, after noise, source)

    def source_counts(self):
        counts = dict.fromkeys(SOURCES, 0)
        for _, sources in self.tokens:
            for s in sources:
                counts[s] += 1
        return counts

    def source_mix(self):
        counts = self.source_counts()
        total = sum(counts.values())
        return {k: round(v / total, 4) if total else 0.0 for k, v in counts.items()}


class Romanizer:
    """Sentence romanizer: shortcuts -> lexicon -> rules, then sentence-level noise.

    lexicon: {devanagari_word: (spellings, cumulative_weights)}, see build_sampler().
    use_shortcuts / use_lexicon = False gives the rule-only baseline of the realism check.
    """

    def __init__(self, lexicon, cfg, use_shortcuts=True, use_lexicon=True):
        self.lexicon = lexicon if use_lexicon else {}
        self.shortcuts = build_sampler(SHORTCUTS) if use_shortcuts else {}
        self.cfg = cfg
        levels = cfg["noise_levels"]
        self.levels, self.level_weights = list(levels), [levels[k] for k in levels]

    def word(self, unit, rng):
        """(latin spelling, source) of one Devanagari word unit, before noise."""
        for table, source in ((self.shortcuts, "shortcut"), (self.lexicon, "lexicon")):
            entry = table.get(unit)
            if entry is not None:
                spellings, cum = entry
                return rng.choices(spellings, cum_weights=cum)[0], source
        return rule_romanize(unit), "rules"

    def sentence(self, text, rng, level=None):
        """Romanize one (normalized) Devanagari sentence. level=None samples the noise level."""
        if level is None:
            level = rng.choices(self.levels, weights=self.level_weights)[0]
        out_tokens, units = [], []
        for token in text.split():
            pieces, sources = [], []
            for unit, digits, other in _PIECES.findall(token):
                if unit:
                    word, source = self.word(unit, rng)
                    if word:
                        noisy = apply_noise(word, level, rng, self.cfg)
                        pieces.append(noisy)
                        sources.append(source)
                        units.append((unit, word, noisy, source))
                elif digits:
                    pieces.append("".join(DEV_DIGITS.get(c, c) for c in digits))
                elif other in _PUNCT:
                    pieces.append(other)
                elif other in DANDAS and rng.random() < self.cfg["danda_to_dot_p"]:
                    pieces.append(".")
            rom_token = "".join(pieces)
            if rom_token:
                out_tokens.append((rom_token, sources))
        return Romanization(" ".join(t for t, _ in out_tokens), level, out_tokens, units)


def build_sampler(table):
    """{word: {spelling: weight}} -> {word: (spellings, cumulative weights)} for rng.choices."""
    out = {}
    for word, variants in table.items():
        spellings, cum, total = [], [], 0
        for spelling, weight in sorted(variants.items(), key=lambda kv: (-kv[1], kv[0])):
            total += weight
            spellings.append(spelling)
            cum.append(total)
        out[word] = (spellings, cum)
    return out


def seeded_rng(seed, *keys):
    """A random.Random whose state depends only on seed and keys (stable across processes)."""
    return random.Random(f"{seed}:" + ":".join(map(str, keys)))
