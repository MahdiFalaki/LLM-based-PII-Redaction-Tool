import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pii_masking.config.config import DEFAULT_HF_DIR, SYSTEM_PROMPT
from pii_masking.scripts.utils.post_processing import normalize_entities
from pii_masking.scripts.utils.prompting import mistral_messages

MODEL_DIR = DEFAULT_HF_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Loading {MODEL_DIR} on {device}…")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=dtype, device_map="auto" if device=="cuda" else None, low_cpu_mem_usage=True
).to(device)

tests = [
  "Mask all PII: John Smith lives at 123 Main Street, Toronto. His phone number is +1 416 555 9823.",
  "Mask all PII: Jean Dupont habite au 12 rue de la Paix, Paris. Son numéro de carte bancaire est 4532 9483 0294 5521.",
  "Mask all PII: Hans Müller wohnt in der Berliner Straße 45, Berlin. Seine E-Mail ist hans.mueller@example.com.",
  "Mask all PII: Giulia Rossi vive in Via Roma 88, Milano. Il suo numero IMEI è 06-184755-866851-3.",
]

for i, text in enumerate(tests, 1):
    msgs = mistral_messages(SYSTEM_PROMPT, text)
    enc = tok.apply_chat_template(msgs, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(enc, max_new_tokens=128, temperature=0.0, do_sample=False)
    raw = tok.decode(out[0], skip_special_tokens=True)
    norm = normalize_entities(raw, system=SYSTEM_PROMPT, user_text=text)
    print(f"\n=== Example {i} ===")
    print("RAW:", raw)
    print("NORM:", norm)
