import os
import re
from typing import Optional

BRACKETED_TAG = re.compile(r"\[([A-Za-z0-9_]+)\]")

PROFILE_FULL = "full"
PROFILE_BASIC = "basic"

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
    "AMOUNT": "AMOUNT",
    "CREDITCARDISSUER": "CREDITCARDNUMBER",
    "CREDITCARDNUMBER": "CREDITCARDNUMBER",
    "CREDITCARDCVV": "CREDITCARDNUMBER",
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

BASIC_MAP = {
    "FIRSTNAME": "NAME",
    "MIDDLENAME": "NAME",
    "LASTNAME": "NAME",
    "PREFIX": "NAME",
    "ADDRESS": "ADDRESS",
    "SECONDARYADDRESS": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "CREDITCARDNUMBER": "CARDNUMBER",
    "MASKEDNUMBER": "CARDNUMBER",
    "PHONENUMBER": "PHONENUMBER",
    "PHONEIMEI": "PHONENUMBER",
    "DOB": "DATE",
}

BASIC_KEEP = {
    "NAME",
    "ADDRESS",
    "CARDNUMBER",
    "PHONENUMBER",
    "DATE",
    "EMAIL",
    "URL",
    "USERNAME",
    "IP",
    "IPV4",
    "IPV6",
    "ACCOUNTNUMBER",
}


def get_tag_profile() -> str:
    profile = os.getenv("PII_TAG_PROFILE", PROFILE_FULL).strip().lower()
    if profile not in {PROFILE_FULL, PROFILE_BASIC}:
        return PROFILE_FULL
    return profile


def canonicalize_tag(raw_tag: str) -> str:
    tag = raw_tag.upper()
    return CANON.get(tag, tag)


def project_tag(tag: str, profile: str) -> str:
    if profile == PROFILE_BASIC:
        if tag in BASIC_MAP:
            return BASIC_MAP[tag]
        if tag in BASIC_KEEP:
            return tag
        return "OTHERPII"
    return tag


def rewrite_bracketed_tags(text: str, profile: Optional[str] = None) -> str:
    active_profile = profile or PROFILE_FULL

    def repl(m):
        tag = canonicalize_tag(m.group(1))
        tag = project_tag(tag, active_profile)
        return f"[{tag}]"

    return BRACKETED_TAG.sub(repl, text)
