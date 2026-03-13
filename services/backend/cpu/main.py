import os
import threading
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pii_masking.infer.gguf_infer import GGUFModel
from pii_masking.utils.post_processing import normalize_entities
from services.backend.common.schema import RedactIn, RedactOut

GGUF_PATH = os.getenv("GGUF_PATH")  # e.g. /models/gguf/quantized/mistral7b-pii-Q5_K_M.gguf
N_CTX = int(os.getenv("N_CTX", "2048"))
THREADS = int(os.getenv("THREADS", str(os.cpu_count() or 4)))
GGUF_SCAN_DIRS = [p.strip() for p in os.getenv("GGUF_SCAN_DIRS", "").split(",") if p.strip()]
EVAL_RUNS_DIR = Path(
    os.getenv("EVAL_RUNS_DIR", str(Path(__file__).resolve().parents[3] / "src" / "pii_masking" / "eval" / "eval_runs"))
)
SYSTEM = os.getenv(
    "PII_SYSTEM_PROMPT",
    "You are a PII redaction assistant. Replace PII with bracketed tags only. "
    "Use only these tags: [NAME], [ADDRESS], [CARDNUMBER], [PHONENUMBER], [DATE], "
    "[EMAIL], [URL], [USERNAME], [IP], [IPV4], [IPV6], [ACCOUNTNUMBER], [OTHERPII]. "
    "Preserve all non-PII text exactly. Output only the redacted text.",
)

app = FastAPI(title="PII Redaction (CPU/GGUF)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_models_lock = threading.Lock()
_model_cache: dict[str, GGUFModel] = {}
_model_infer_locks: dict[str, threading.Lock] = {}
_allowed_models: list[str] = []


def _default_scan_dirs() -> list[str]:
    if GGUF_SCAN_DIRS:
        return GGUF_SCAN_DIRS
    if not GGUF_PATH:
        return []

    path = Path(GGUF_PATH).resolve()
    dirs = [path.parent]
    if path.parent.name in {"quantized", "f16"}:
        dirs.append(path.parent.parent)
    return [str(d) for d in dirs if d.exists()]


def _scan_models() -> list[str]:
    out = set()
    if GGUF_PATH and os.path.isfile(GGUF_PATH):
        out.add(str(Path(GGUF_PATH).resolve()))
    for d in _default_scan_dirs():
        p = Path(d)
        if p.is_dir():
            for f in p.rglob("*.gguf"):
                out.add(str(f.resolve()))
    return sorted(out)


def _refresh_allowed_models() -> list[str]:
    global _allowed_models
    _allowed_models = _scan_models()
    return _allowed_models


def _default_model_path() -> Optional[str]:
    allowed = _refresh_allowed_models()
    if GGUF_PATH and os.path.isfile(GGUF_PATH):
        return str(Path(GGUF_PATH).resolve())
    if allowed:
        return allowed[0]
    return None


def _get_model(model_path: str) -> GGUFModel:
    rp = str(Path(model_path).resolve())
    with _models_lock:
        if rp not in _model_cache:
            _model_cache[rp] = GGUFModel(rp, n_ctx=N_CTX, n_threads=THREADS)
            _model_infer_locks[rp] = threading.Lock()
        return _model_cache[rp]


def _get_model_infer_lock(model_path: str) -> threading.Lock:
    rp = str(Path(model_path).resolve())
    with _models_lock:
        if rp not in _model_infer_locks:
            _model_infer_locks[rp] = threading.Lock()
        return _model_infer_locks[rp]


def _eval_file(path: Path) -> Path:
    rp = path.resolve()
    base = EVAL_RUNS_DIR.resolve()
    if base not in rp.parents and rp != base:
        raise HTTPException(status_code=400, detail="Invalid eval path.")
    return rp


def _read_json(path: Path):
    p = _eval_file(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Eval file not found: {p.name}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {p.name}: {e}") from e


def _summary_path_for_run(run_name: str) -> Path:
    if run_name == "hf_merged":
        return EVAL_RUNS_DIR / "hf_merged_summary.json"
    return EVAL_RUNS_DIR / f"{run_name}_summary.json"


def _norm_row(row: dict) -> dict:
    return {
        "run_name": row.get("run_name"),
        "model_type": row.get("model_type"),
        "gguf_path": row.get("gguf_path"),
        "micro_f1": row.get("micro_f1"),
        "macro_f1": row.get("macro_f1"),
        "exact_match_rate": row.get("exact_match_rate"),
        "avg_latency_ms": row.get("avg_latency_ms"),
        "p95_latency_ms": row.get("p95_latency_ms"),
    }

@app.on_event("startup")
def _load():
    default_model = _default_model_path()
    assert default_model, f"No GGUF models found. GGUF_PATH={GGUF_PATH}, GGUF_SCAN_DIRS={GGUF_SCAN_DIRS}"
    _get_model(default_model)

@app.get("/")
def root():
    model_path = _default_model_path()
    return {
        "backend": "cpu-gguf",
        "mode": "single-model-ready",
        "gguf": model_path,
        "model_name": os.path.basename(model_path) if model_path else None,
        "n_ctx": N_CTX,
        "threads": THREADS,
    }


@app.get("/models")
def models():
    allowed = _refresh_allowed_models()
    return {"default_model": _default_model_path(), "models": allowed}


@app.get("/eval/leaderboard")
def eval_leaderboard():
    payload = _read_json(EVAL_RUNS_DIR / "leaderboard.json")
    rows = [_norm_row(r) for r in payload.get("rows", [])]
    return {"dataset": payload.get("dataset"), "rows": rows}


@app.get("/eval/summary/{run_name}")
def eval_summary(run_name: str):
    if not run_name:
        raise HTTPException(status_code=400, detail="run_name is required.")
    return _read_json(_summary_path_for_run(run_name))

@app.post("/redact", response_model=RedactOut)
def redact(x: RedactIn):
    allowed = _refresh_allowed_models()
    selected = x.model_path or _default_model_path()
    if not selected:
        raise HTTPException(status_code=400, detail="No model path provided.")
    selected = str(Path(selected).resolve())
    if selected not in allowed:
        raise HTTPException(status_code=400, detail=f"Model not in allowed list: {selected}")

    max_new = x.max_new_tokens or 256
    model = _get_model(selected)
    infer_lock = _get_model_infer_lock(selected)
    # llama.cpp python bindings are not safe for concurrent generation on the same model instance.
    t0 = time.perf_counter()
    with infer_lock:
        raw = model.generate(SYSTEM, x.text, max_new_tokens=max_new)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    norm = normalize_entities(raw, system=SYSTEM, user_text=x.text)
    return RedactOut(
        normalized=norm,
        latency_ms=latency_ms,
        tag_count=norm.count("["),
        model_name=os.path.basename(selected),
        model_path=selected,
        max_new_tokens=max_new,
    )
