import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pii_masking.infer.gguf_infer import GGUFModel
from pii_masking.utils.post_processing import normalize_entities
from services.backend.common.schema import RedactIn, RedactOut

GGUF_PATH = os.getenv("GGUF_PATH")  # e.g. /models/mistral7b-pii-f16.gguf
N_CTX = int(os.getenv("N_CTX", "2048"))
THREADS = int(os.getenv("THREADS", str(os.cpu_count() or 4)))
SYSTEM = os.getenv("SYSTEM_PROMPT", "You are a data privacy assistant.")

app = FastAPI(title="PII Redaction (CPU/GGUF)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

@app.on_event("startup")
def _load():
    global _model
    assert GGUF_PATH and os.path.isfile(GGUF_PATH), f"Missing GGUF_PATH: {GGUF_PATH}"
    _model = GGUFModel(GGUF_PATH, n_ctx=N_CTX, n_threads=THREADS)

@app.get("/")
def root():
    return {"backend": "cpu-gguf", "gguf": GGUF_PATH, "n_ctx": N_CTX, "threads": THREADS}

@app.post("/redact", response_model=RedactOut)
def redact(x: RedactIn):
    max_new = x.max_new_tokens or 256
    raw = _model.generate(SYSTEM, x.text, max_new_tokens=max_new)
    norm = normalize_entities(raw, system=SYSTEM, user_text=x.text)
    return RedactOut(normalized=norm)
