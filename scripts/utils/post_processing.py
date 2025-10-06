# post_processing.py
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

def strip_system_echo(text: str, system: Optional[str] = None) -> str:
    t = text.replace("<s>", "").replace("</s>", "").strip()
    if system:
        ts = t.lstrip()
        if ts.startswith(system):
            t = ts[len(system):].lstrip()
    return t

def canonicalize_tags(text: str) -> str:
    def repl(m):
        tag = m.group(1).upper()
        tag = CANON.get(tag, tag)
        return f"[{tag}]"
    return BRACKETED_TAG.sub(repl, text)

def collapse_address(text: str) -> str:
    return RE_ADDRESS_BLOCK.sub("[ADDRESS]", text)

def override_credit_card(user_text: str, normalized_text: str) -> str:
    if not RE_CC_IN_INPUT.search(user_text):
        return normalized_text
    m = RE_CREDIT_CARD_WORDING.search(normalized_text)
    if m:
        after = normalized_text[m.end():]
        replaced_after = after.replace("[PHONENUMBER]", "[CREDITCARDNUMBER]", 1)
        if after != replaced_after:
            return normalized_text[:m.end()] + replaced_after
    if normalized_text.count("[PHONENUMBER]") == 1 and "[CREDITCARDNUMBER]" not in normalized_text:
        return normalized_text.replace("[PHONENUMBER]", "[CREDITCARDNUMBER]")
    return normalized_text

# --- NEW: for ground-truth normalization ---
def normalize_reference(text: str) -> str:
    import re
    BRACKETED_TAG = re.compile(r"\[([A-Za-z0-9_]+)\]")
    def repl(m):
        tag = m.group(1).upper()
        tag = CANON.get(tag, tag)
        return f"[{tag}]"
    return BRACKETED_TAG.sub(repl, text)

# --- For predictions ---
def normalize_entities(model_text: str, system: Optional[str] = None, user_text: Optional[str] = None) -> str:
    t = strip_system_echo(model_text, system=system)
    t = canonicalize_tags(t)
    t = collapse_address(t)
    if user_text:
        t = override_credit_card(user_text, t)
    return t
