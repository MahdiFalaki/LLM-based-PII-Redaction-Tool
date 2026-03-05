# src/pii_masking/utils/post_processing.py
import re
import os
from typing import Optional
from pii_masking.utils.tag_profiles import CANON, get_tag_profile, rewrite_bracketed_tags

RE_CC_IN_INPUT = re.compile(r"(?:\d[ -]?){13,19}")
RE_CREDIT_CARD_WORDING = re.compile(r"credit\s*card", re.IGNORECASE)
RE_CC_KEYWORDS_IN_INPUT = re.compile(
    r"\b(credit\s*card|visa|mastercard|amex|american\s*express|card\s*number|cvv)\b",
    re.IGNORECASE,
)

RE_ADDRESS_BLOCK = re.compile(
    r"""
    \[BUILDINGNUMBER\]\s+\[STREET\]
    (?:\s*,\s*\[CITY\])?
    (?:\s*,\s*\[STATE\])?
    (?:\s*,\s*\[ZIPCODE\])?
    """,
    re.VERBOSE,
)
RE_REDUNDANT_ADDRESS = re.compile(r"(?:\[ADDRESS\](?:\s*,?\s*)){2,}")

def strip_to_last_assistant_segment(text: str) -> str:
    parts = text.split("[/INST]")
    tail = parts[-1] if parts else text
    tail = tail.replace("<s>", "").replace("</s>", "")
    tail = re.sub(r"\[(?:\/)?INST\]", "", tail)
    return tail.strip()

def strip_leading_mask_instruction(text: str) -> str:
    return re.sub(r"^\s*mask\s+all\s+pii:\s*", "", text, flags=re.IGNORECASE)

def canonicalize_tags(text: str) -> str:
    return rewrite_bracketed_tags(text, profile=get_tag_profile())

def collapse_address(text: str) -> str:
    t = RE_ADDRESS_BLOCK.sub("[ADDRESS]", text)
    t = RE_REDUNDANT_ADDRESS.sub("[ADDRESS] ", t)
    return re.sub(r"\s+", " ", t).strip()

def override_credit_card(user_text: str, normalized_text: str) -> str:
    if not RE_CC_IN_INPUT.search(user_text or ""):
        return normalized_text
    if not RE_CC_KEYWORDS_IN_INPUT.search(user_text or ""):
        return normalized_text

    # also treat PHONEIMEI as a phone-like mis-tag here
    PHONE_LIKE = ("[PHONENUMBER]", "[PHONEIMEI]")

    m = RE_CREDIT_CARD_WORDING.search(normalized_text)
    if m:
        after = normalized_text[m.end():]
        for ph in PHONE_LIKE:
            if ph in after:
                return normalized_text[:m.end()] + after.replace(ph, "[CREDITCARDNUMBER]", 1)

    # fallback: single phone-like tag anywhere
    for ph in PHONE_LIKE:
        if normalized_text.count(ph) == 1 and "[CREDITCARDNUMBER]" not in normalized_text:
            return normalized_text.replace(ph, "[CREDITCARDNUMBER]")

    return normalized_text

def normalize_reference(text: str) -> str:
    t = canonicalize_tags(text)
    if os.getenv("PII_COLLAPSE_ADDRESS", "1") == "1":
        t = collapse_address(t)
    return t

def normalize_entities(model_text: str, system: Optional[str] = None, user_text: Optional[str] = None) -> str:
    t = strip_to_last_assistant_segment(model_text)
    if system:
        ts = t.lstrip()
        if ts.startswith(system):
            t = ts[len(system):].lstrip()
    t = strip_leading_mask_instruction(t)
    t = canonicalize_tags(t)
    if os.getenv("PII_COLLAPSE_ADDRESS", "1") == "1":
        t = collapse_address(t)
    if user_text:
        t = override_credit_card(user_text, t)
    return t.strip()
