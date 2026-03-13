from pydantic import BaseModel

class RedactIn(BaseModel):
    text: str
    max_new_tokens: int | None = None
    model_path: str | None = None

class RedactOut(BaseModel):
    normalized: str
    latency_ms: float | None = None
    tag_count: int | None = None
    model_name: str | None = None
    model_path: str | None = None
    max_new_tokens: int | None = None
