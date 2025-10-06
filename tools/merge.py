import os, time, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.2"  # base model on HF hub
ROOT = Path("/outputs/pii_masking_mistral")  # Axolotl outputs
OUT  = Path("/outputs/pii_masking_mistral/merged_pii_model")            # where to save merged

def is_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").is_file() and (
        (p / "adapter_model.safetensors").is_file()
        or (p / "adapter_model.bin").is_file()
        or (p / "pytorch_model.bin").is_file()
    )

# Find the newest valid adapter folder under ROOT
candidates = []
for dirpath, _, _ in os.walk(ROOT):
    d = Path(dirpath)
    if is_adapter_dir(d):
        mt = 0.0
        for f in d.glob("adapter_model*"):
            try:
                mt = max(mt, f.stat().st_mtime)
            except FileNotFoundError:
                pass
        candidates.append((mt, d))

if not candidates:
    raise FileNotFoundError(
        f"No adapter found under {ROOT}. Need adapter_config.json + adapter_model.*"
    )

candidates.sort(reverse=True)
ADAPTER = candidates[0][1]
print(f"Using adapter directory: {ADAPTER}")
print(f"Adapter modified: {time.ctime(candidates[0][0])}")

# Prefer tokenizer from adapter dir if present (it may include special tags or chat template)
tokenizer_src = ADAPTER if (ADAPTER / "tokenizer_config.json").is_file() else BASE
tok = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
print(f"Loaded tokenizer from: {tokenizer_src}")

use_cuda = torch.cuda.is_available()
dtype = torch.bfloat16 if use_cuda else torch.float32

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=dtype,
    device_map="auto" if use_cuda else "cpu",
    low_cpu_mem_usage=True,
)

print("Merging LoRA into base (this may take a bit)...")
with torch.inference_mode():
    merged = PeftModel.from_pretrained(base, str(ADAPTER))
    merged = merged.merge_and_unload()

OUT.mkdir(parents=True, exist_ok=True)

print("Saving merged model (safetensors, sharded)...")
merged.save_pretrained(str(OUT), safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained(str(OUT))

# If Axolotl wrote a chat template, copy it so inference matches training formatting
tmpl = ADAPTER / "chat_template.jinja"
if tmpl.is_file():
    shutil.copy2(tmpl, OUT / "chat_template.jinja")
    print("Copied chat_template.jinja into merged folder.")

print(f"✅ Merged model saved to: {OUT}")
