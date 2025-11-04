import re
from pii_masking.scripts.utils.post_processing import CANON  # package import

BRACKETED_TAG = re.compile(r"\[([A-Za-z0-9_]+)\]")

def normalize_reference(text: str) -> str:
    def repl(m):
        tag = m.group(1).upper()
        tag = CANON.get(tag, tag)
        return f"[{tag}]"
    return BRACKETED_TAG.sub(repl, text)
