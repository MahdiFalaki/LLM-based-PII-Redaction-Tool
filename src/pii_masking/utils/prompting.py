# src/pii_masking/utils/prompting.py
def mistral_inst(system: str, user: str) -> str:
    return f"<s>[INST] {system}\n\n{user} [/INST]"

def mistral_messages(system: str, user: str):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
