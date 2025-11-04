import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pii_masking.infer.hf_infer import HFModel
from pii_masking.utils.post_processing import normalize_entities
from services.backend.common.schema import RedactIn, RedactOut

HF_DIR = os.getenv("HF_DIR")  # e.g. /models/merged_pii_model
SYSTEM = os.getenv("SYSTEM_PROMPT", "You are a data privacy assistant.")

app = FastAPI(title="PII Redaction (GPU/HF)")
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
    assert HF_DIR and os.path.isdir(HF_DIR), f"Missing HF_DIR: {HF_DIR}"
    _model = HFModel(HF_DIR)

@app.get("/")
def root():
    return {"backend": "gpu-hf", "model_dir": HF_DIR}

@app.post("/redact", response_model=RedactOut)
def redact(x: RedactIn):
    max_new = x.max_new_tokens or 256
    raw = _model.generate(SYSTEM, x.text, max_new_tokens=max_new)
    norm = normalize_entities(raw, system=SYSTEM, user_text=x.text)
    return RedactOut(normalized=norm)
