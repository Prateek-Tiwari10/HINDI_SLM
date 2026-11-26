"""Demo server tests, against the real pinned model (serve/MODEL_ID).

The expected outputs are the CPU outputs recorded by D7 Step 6 in the release's examples.json.
"""

import json
import os
import re

os.environ["RATE_LIMIT_PER_MIN"] = "1000"     # the 429 path is tested with its own limiter below

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from finetune.translit import split_input  # noqa: E402
from serve.app import MAX_INPUT_CHARS, app  # noqa: E402
from serve.model_ref import model_dir  # noqa: E402
from serve.ratelimit import RateLimiter, client_key  # noqa: E402

LATIN = re.compile(r"[A-Za-z]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def post(client, text):
    return client.post("/api/transliterate", json={"text": text})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "@" in body["model"]


def test_page_and_examples(client):
    assert client.get("/").status_code == 200 and "Transliterate" in client.get("/").text
    ex = client.get("/api/examples").json()
    assert len(ex) >= 5 and all(e["category"] in ("best", "ambiguous") for e in ex)


def test_known_pairs_match_step6_cpu_outputs(client):
    ex = json.loads((model_dir() / "examples.json").read_text())
    known = [(e["rom"], e["hyp_cpu"]) for e in ex if e["category"] == "best"]
    assert len(known) >= 3
    for rom, expected in known:
        r = post(client, rom)
        assert r.status_code == 200
        body = r.json()
        assert body["output"] == expected, rom
        assert body["latency_ms"] > 0


def test_simple_sentence(client):
    out = post(client, "kya haal hai").json()["output"]
    assert DEVANAGARI.search(out) and not LATIN.search(out)


def test_long_input_is_split_and_answered(client):
    text = ("aaj subah main office jaldi pahuncha kyunki kal raat baarish bahut tez thi aur sadak par pani bhara tha. "
            * 4).strip()
    assert len(text) <= MAX_INPUT_CHARS and len(split_input(text)) > 1
    r = post(client, text)
    assert r.status_code == 200
    out = r.json()["output"]
    assert not LATIN.search(out) and len(out.split()) >= len(text.split()) // 2


@pytest.mark.parametrize("payload", [{"text": "a" * (MAX_INPUT_CHARS + 1)}, {"text": ""}, {"text": "नमस्ते दुनिया"},
                                     {"text": "  123 ?! "}, {}, {"text": 5}])
def test_bad_input_is_422(client, payload):
    assert client.post("/api/transliterate", json=payload).status_code == 422


def test_rate_limit_returns_429(client):
    saved = app.state.limiter
    app.state.limiter = RateLimiter(2, 60.0)
    try:
        codes = [post(client, "theek hai").status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        r = post(client, "theek hai")
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    finally:
        app.state.limiter = saved


def test_rate_limiter_window():
    rl = RateLimiter(2, 60.0)
    assert rl.check("a", 0) is None and rl.check("a", 1) is None
    assert rl.check("a", 2) == 58.0
    assert rl.check("b", 2) is None                      # per client
    assert rl.check("a", 60.5) is None                   # first hit left the window


def test_client_key_uses_last_forwarded_address():
    assert client_key({"x-forwarded-for": "6.6.6.6, 203.0.113.9"}, "10.0.0.1") == "203.0.113.9"
    assert client_key({}, "10.0.0.1") == "10.0.0.1"
