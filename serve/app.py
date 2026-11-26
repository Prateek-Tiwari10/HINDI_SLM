"""Live demo: Hinglish -> Devanagari with the 61M model (D7), CPU only.

    uvicorn serve.app:app --host 0.0.0.0 --port 8000 --workers 1

GET  /                    one static page (textbox, examples, latency)
POST /api/transliterate   {"text": "..."} -> {"output": "...", "latency_ms": 123.4}
GET  /api/examples        the Step 6 examples shown as chips
GET  /health              liveness + which model / build is running
GET  /docs                Swagger UI (FastAPI built-in)

The model + tokenizer load once at startup (lifespan). Inference is the exact Step 6 path
(finetune/translit.py): normalized Latin input split into <= 160-char chunks, greedy decoding with the
KV cache and the Devanagari-only output mask. One inference at a time per replica (a lock), because a
0.5 vCPU replica gains nothing from running two at once.

Settings (env): MODEL_ID (owner/repo@revision; default serve/MODEL_ID), MAX_INPUT_CHARS (500),
RATE_LIMIT_PER_MIN (30 per client), TORCH_THREADS (1), GIT_SHA (set by the image build).
"""

import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from finetune.translit import load_release
from serve.model_ref import model_dir, model_id
from serve.ratelimit import RateLimiter, client_key

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "1"))
GIT_SHA = os.environ.get("GIT_SHA", "dev")
STATIC = Path(__file__).with_name("static")


# ---------------------------------------------------------------- JSON logs
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname, "msg": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


log = logging.getLogger("serve")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(JsonFormatter())
    log.addHandler(_h)
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    log.propagate = False


def jlog(msg: str, **fields) -> None:
    log.info(msg, extra={"fields": fields})


# ---------------------------------------------------------------- model lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(TORCH_THREADS)
    t0 = time.perf_counter()
    ref = model_id()
    path = model_dir(ref)
    app.state.translit = load_release(path, "cpu")
    examples = path / "examples.json"
    app.state.examples = json.loads(examples.read_text()) if examples.exists() else []
    app.state.lock = threading.Lock()
    app.state.limiter = RateLimiter(RATE_LIMIT_PER_MIN, 60.0)
    app.state.model_id = ref
    app.state.translit("namaste")                       # warm-up: first call pays one-time allocation costs
    jlog("startup", model_id=ref, load_s=round(time.perf_counter() - t0, 2), threads=TORCH_THREADS, git_sha=GIT_SHA)
    yield


app = FastAPI(title="Hinglish → Devanagari", version=GIT_SHA, lifespan=lifespan,
              description="A 61M-parameter Hindi model trained from scratch, fine-tuned to turn Hindi typed in "
                          "Latin letters into Devanagari. Code: https://github.com/Prateek-Tiwari10/HINDI_SLM")


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        jlog("request", method=request.method, path=request.url.path, status=status,
             latency_ms=round((time.perf_counter() - t0) * 1000, 1))


# ---------------------------------------------------------------- API
class TransliterateIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS,
                      examples=["kal meeting cancel ho gayi kya"],
                      description=f"Hindi typed in Latin letters, up to {MAX_INPUT_CHARS} characters.")

    @field_validator("text")
    @classmethod
    def has_latin_letters(cls, v: str) -> str:
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("text must contain Latin letters (Hindi typed in English letters)")
        return v


class TransliterateOut(BaseModel):
    output: str = Field(..., examples=["कल मीटिंग कैंसिल हो गई क्या"])
    latency_ms: float = Field(..., description="Server time for this request, including any wait for the model.")


@app.post("/api/transliterate", response_model=TransliterateOut)
def transliterate(body: TransliterateIn, request: Request) -> TransliterateOut:
    t0 = time.perf_counter()
    retry = request.app.state.limiter.check(client_key(request.headers, request.client.host if request.client else None))
    if retry is not None:
        raise HTTPException(429, detail=f"rate limit: {RATE_LIMIT_PER_MIN} requests per minute",
                            headers={"Retry-After": str(int(retry) + 1)})
    with request.app.state.lock:
        r = request.app.state.translit(body.text)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    jlog("transliterate", chars=len(body.text), chunks=r.chunks, new_tokens=r.new_tokens,
         model_ms=round(r.seconds * 1000, 1), latency_ms=ms)
    return TransliterateOut(output=r.output, latency_ms=ms)


@app.get("/api/examples")
def examples(request: Request) -> list[dict]:
    """Step 6 test examples (Dakshina test, clean part) that the model gets right: best + ambiguous."""
    return [{"rom": e["rom"], "category": e["category"]} for e in request.app.state.examples
            if e.get("category") in ("best", "ambiguous")]


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model": request.app.state.model_id, "version": GIT_SHA}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled", extra={"fields": {"path": request.url.path}})
    return JSONResponse({"detail": "internal error"}, status_code=500)
