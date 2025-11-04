# src/pii_masking/utils/post_processing.py
import re
from typing import Optional

CANON = {
    "FIRSTNAME": "FIRSTNAME",
    "MIDDLENAME": "MIDDLENAME",
    "LASTNAME": "LASTNAME",
    "PREFIX": "PREFIX",
    "DATE": "DATE",
    "TIME": "TIME",
    "PHONEIMEI": "PHONEIMEI",
    "USERNAME": "USERNAME",
    "GENDER": "GENDER",
    "CITY": "ADDRESS",
    "STATE": "ADDRESS",
    "URL": "URL",
    "JOBAREA": "JOBAREA",
    "EMAIL": "EMAIL",
    "JOBTYPE": "JOBTYPE",
    "COMPANYNAME": "COMPANYNAME",
    "JOBTITLE": "JOBTITLE",
    "STREET": "ADDRESS",
    "SECONDARYADDRESS": "SECONDARYADDRESS",
    "COUNTY": "ADDRESS",
    "AGE": "AGE",
    "USERAGENT": "USERAGENT",
    "ACCOUNTNAME": "ACCOUNTNAME",
    "ACCOUNTNUMBER": "ACCOUNTNUMBER",
    "CURRENCYSYMBOL": "CURRENCYSYMBOL",
    "AMOUNT": "MASKEDNUMBER",
    "CREDITCARDISSUER": "MASKEDNUMBER",
    "CREDITCARDNUMBER": "MASKEDNUMBER",
    "CREDITCARDCVV": "MASKEDNUMBER",
    "PHONENUMBER": "PHONENUMBER",
    "SEX": "SEX",
    "IP": "IP",
    "ETHEREUMADDRESS": "ETHEREUMADDRESS",
    "BITCOINADDRESS": "BITCOINADDRESS",
    "IBAN": "IBAN",
    "VEHICLEVRM": "VEHICLEVRM",
    "DOB": "DOB",
    "PIN": "PIN",
    "CURRENCY": "CURRENCY",
    "PASSWORD": "PASSWORD",
    "CURRENCYNAME": "CURRENCYNAME",
    "LITECOINADDRESS": "LITECOINADDRESS",
    "CURRENCYCODE": "CURRENCYCODE",
    "BUILDINGNUMBER": "ADDRESS",
    "ORDINALDIRECTION": "ORDINALDIRECTION",
    "MASKEDNUMBER": "MASKEDNUMBER",
    "ZIPCODE": "ZIPCODE",
    "BIC": "BIC",
    "IPV4": "IPV4",
    "IPV6": "IPV6",
    "MAC": "MAC",
    "NEARBYGPSCOORDINATE": "NEARBYGPSCOORDINATE",
    "VEHICLEVIN": "VEHICLEVIN",
    "EYECOLOR": "EYECOLOR",
    "HEIGHT": "HEIGHT",
    "SSN": "SSN",
}

BRACKETED_TAG = re.compile(r"\[([A-Za-z0-9_]+)\]")

RE_CC_IN_INPUT = re.compile(r"(?:\d[ -]?){13,19}")
RE_CREDIT_CARD_WORDING = re.compile(r"credit\s*card", re.IGNORECASE)

RE_ADDRESS_BLOCK = re.compile(
    r"""
    \[BUILDINGNUMBER\]\s+\[STREET\]
    (?:\s*,\s*\[CITY\])?
    (?:\s*,\s*\[STATE\])?
    (?:\s*,\s*\[ZIPCODE\])?
    """,
    re.VERBOSE,
)

def strip_to_last_assistant_segment(text: str) -> str:
    parts = text.split("[/INST]")
    tail = parts[-1] if parts else text
    tail = tail.replace("<s>", "").replace("</s>", "")
    tail = re.sub(r"\[(?:\/)?INST\]", "", tail)
    return tail.strip()

def strip_leading_mask_instruction(text: str) -> str:
    return re.sub(r"^\s*mask\s+all\s+pii:\s*", "", text, flags=re.IGNORECASE)

def canonicalize_tags(text: str) -> str:
    def repl(m):
        tag = CANON.get(m.group(1).upper(), m.group(1).upper())
        return f"[{tag}]"
    return BRACKETED_TAG.sub(repl, text)

def collapse_address(text: str) -> str:
    return RE_ADDRESS_BLOCK.sub("[ADDRESS]", text)

def override_credit_card(user_text: str, normalized_text: str) -> str:
    if not RE_CC_IN_INPUT.search(user_text or ""):
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
    return canonicalize_tags(text)

def normalize_entities(model_text: str, system: Optional[str] = None, user_text: Optional[str] = None) -> str:
    t = strip_to_last_assistant_segment(model_text)
    if system:
        ts = t.lstrip()
        if ts.startswith(system):
            t = ts[len(system):].lstrip()
    t = strip_leading_mask_instruction(t)
    t = canonicalize_tags(t)
    t = collapse_address(t)
    if user_text:
        t = override_credit_card(user_text, t)
    return t.strip()
