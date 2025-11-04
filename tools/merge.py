# tools/merge.py
import os, shutil, time
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pii_masking.config.config import BASE_MODEL, DEFAULT_HF_DIR

def is_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").is_file() and any(
        (p / n).is_file() for n in ("adapter_model.safetensors","adapter_model.bin","pytorch_model.bin")
    )

def find_newest_adapter(root: Path) -> Path:
    cands = []
    for dirpath, _, _ in os.walk(root):
        d = Path(dirpath)
        if is_adapter_dir(d):
            mt = 0.0
            for f in d.glob("adapter_model*"):
                try: mt = max(mt, f.stat().st_mtime)
                except FileNotFoundError: pass
            cands.append((mt, d))
    if not cands:
        raise FileNotFoundError(f"No adapter found under {root}")
    cands.sort(reverse=True)
    return cands[0][1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--root", default=str(Path(DEFAULT_HF_DIR).parent))
    ap.add_argument("--out", default=DEFAULT_HF_DIR)
    args = ap.parse_args()

    ROOT = Path(args.root)
    OUT = Path(args.out)
    ADAPTER = find_newest_adapter(ROOT)
    print(f"Using adapter: {ADAPTER} (modified {time.ctime(ADAPTER.stat().st_mtime)})")

    tok_src = ADAPTER if (ADAPTER / "tokenizer_config.json").is_file() else args.base_model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    print(f"Loaded tokenizer from: {tok_src}")

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        low_cpu_mem_usage=True,
    )

    print("Merging LoRA -> base…")
    with torch.inference_mode():
        merged = PeftModel.from_pretrained(base, str(ADAPTER))
        merged = merged.merge_and_unload()

    OUT.mkdir(parents=True, exist_ok=True)
    print("Saving merged model…")
    merged.save_pretrained(str(OUT), safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(str(OUT))

    tmpl = ADAPTER / "chat_template.jinja"
    if tmpl.is_file():
        shutil.copy2(tmpl, OUT / "chat_template.jinja")
        print("Copied chat_template.jinja")

    print(f"✅ Saved merged model to: {OUT}")

if __name__ == "__main__":
    main()
