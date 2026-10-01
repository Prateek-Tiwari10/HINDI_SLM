import pytest

from data.normalize import normalize

ZWNJ = "‌"


def test_composed_and_decomposed_nukta_match():
    composed = "क़"            # क़ as one code point
    decomposed = "क़"    # क + nukta
    assert normalize(composed) == normalize(decomposed)


def test_zwnj_kept_between_devanagari():
    text = f"क{ZWNJ}ष"
    assert normalize(text) == text


def test_zwnj_removed_next_to_latin():
    assert normalize(f"a{ZWNJ}b") == "ab"
    assert normalize(f"क{ZWNJ}b") == "कb"


def test_pipe_after_devanagari_becomes_danda():
    assert normalize("यह सही है |") == "यह सही है।"


def test_pipe_between_latin_unchanged():
    assert normalize("a | b") == "a | b"


def test_devanagari_digits_kept():
    assert normalize("१२३ 123") == "१२३ 123"


def test_whitespace_collapsed():
    assert normalize("  a \t b\n\n\n\nc  ") == "a b\n\nc"


def test_control_chars_removed_but_newlines_kept():
    assert normalize("a\x00b\x07c\nd\te") == "abc\nd e"


@pytest.mark.parametrize("text", [
    "यह सही है |",
    "है | | आगे",
    "क\x01|",
    "क‍\x01ष",
    "&amp;lt;p&amp;gt; हिंदी",
    "क\x01़",
    "नमस्ते  दुनिया\n\n\n\nफिर  मिलेंगे ||",
    "mixed English और हिंदी text | here",
])
def test_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once
