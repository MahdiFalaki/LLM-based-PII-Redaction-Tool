from pydantic import BaseModel

class RedactIn(BaseModel):
    text: str
    max_new_tokens: int | None = None

class RedactOut(BaseModel):
    normalized: str
