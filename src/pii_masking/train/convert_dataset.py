# src/pii_masking/train/convert_dataset.py
import json
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "data" / "pii_mask.jsonl"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    out = []
    for ex in ds:
        out.append({
            "system": "You are a data privacy assistant.",
            "instruction": "Mask all personally identifiable information in the given text.",
            "input": ex["source_text"],
            "output": ex["target_text"],
        })

    with OUT.open("w", encoding="utf-8") as f:
        for obj in out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(out)} examples to {OUT}")

if __name__ == "__main__":
    main()
