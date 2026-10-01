"""Text normalization (D3). Shared by the data pipeline and, later, by serving.

normalize() applies, in order:
  1. html.unescape (repeated until stable, so double-escaped '&amp;lt;' settles)
  2. Unicode NFC (not NFKC). Nukta letters become base + nukta; that is expected.
  3. Remove control characters (category C*) except \\n, \\t, ZWJ and ZWNJ.
  4. Remove ZWJ (U+200D) / ZWNJ (U+200C) unless both neighbours are Devanagari.
  5. A '|' after Devanagari text (optionally after spaces) becomes the danda '।'.
  6. Collapse spaces/tabs to one space, 3+ newlines to 2, strip.
  7. NFC once more, so removals in 3-4 cannot leave an uncomposed pair behind.
Devanagari digits are NOT converted to ASCII (D3).

Control-character removal runs before the joiner and pipe steps (the decision log
lists it after them). Removing a control character can make a joiner or a pipe
newly adjacent to Devanagari. Doing it first means the joiner and pipe steps see
the final neighbours, which makes normalize() idempotent. On normal text the
output is the same either way.
"""

import html
import re
import sys
import unicodedata
from functools import lru_cache

ZWNJ = "‌"
ZWJ = "‍"
DEVANAGARI = "ऀ-ॿ"
DANDA = "।"
MAX_UNESCAPE_ROUNDS = 3

# A joiner is dropped when its left or its right neighbour is not Devanagari.
_STRAY_JOINER = re.compile(
    rf"(?<![{DEVANAGARI}])[{ZWNJ}{ZWJ}]|[{ZWNJ}{ZWJ}](?![{DEVANAGARI}])"
)
# Spaces/tabs then one or more pipes, right after a Devanagari character.
_PIPE_AFTER_DEVANAGARI = re.compile(rf"(?<=[{DEVANAGARI}])[ \t]*\|+")
_SPACES = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


@lru_cache(maxsize=1)
def _control_char_pattern():
    """Regex matching every C* code point except \\n, \\t, ZWJ, ZWNJ.

    Built once by scanning the code space into ranges, so removal runs in C
    instead of a Python loop per character.
    """
    keep = {"\n", "\t", ZWJ, ZWNJ}
    ranges = []
    start = None
    for cp in range(sys.maxunicode + 1):
        ch = chr(cp)
        is_control = unicodedata.category(ch).startswith("C") and ch not in keep
        if is_control and start is None:
            start = cp
        elif not is_control and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, sys.maxunicode))
    parts = [
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in ranges
    ]
    return re.compile("[" + "".join(parts) + "]")


def _unescape_html(text):
    """html.unescape until the text stops changing (bounded)."""
    for _ in range(MAX_UNESCAPE_ROUNDS):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def _remove_control_chars(text):
    return _control_char_pattern().sub("", text)


def _remove_stray_joiners(text):
    return _STRAY_JOINER.sub("", text)


def _pipe_to_danda(text):
    """'है |' -> 'है।' and 'है ||' -> 'है।।'. Repeats until stable for chains like '| |'."""
    def replace(match):
        return DANDA * match.group(0).count("|")

    while True:
        new_text = _PIPE_AFTER_DEVANAGARI.sub(replace, text)
        if new_text == text:
            return text
        text = new_text


def _collapse_whitespace(text):
    text = _SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def normalize(text):
    """Normalize one document. Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = _unescape_html(text)
    text = unicodedata.normalize("NFC", text)
    text = _remove_control_chars(text)
    text = _remove_stray_joiners(text)
    text = _pipe_to_danda(text)
    text = _collapse_whitespace(text)
    return unicodedata.normalize("NFC", text)
