# examples/pii_masking/prompting.py
def mistral_inst(system: str, user: str) -> str:
    # plain string for llama.cpp “completion” API
    return f"<s>[INST] {system}\n\n{user} [/INST]"

def mistral_messages(system: str, user: str):
    # messages for Hugging Face chat template
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
