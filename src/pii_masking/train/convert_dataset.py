# src/pii_masking/train/convert_dataset.py
import json
import os
import random
from pathlib import Path
from datasets import load_dataset
from pii_masking.utils.tag_profiles import get_tag_profile, rewrite_bracketed_tags

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "data" / "pii_mask.jsonl"
CACHE_DIR = Path(os.getenv("PII_DATASETS_CACHE", PROJECT_ROOT / ".cache" / "hf_datasets"))
SYSTEM_PROMPT = os.getenv(
    "PII_SYSTEM_PROMPT",
    "You are a PII redaction assistant. Replace PII with bracketed tags only. "
    "Use only these tags: [NAME], [ADDRESS], [CARDNUMBER], [PHONENUMBER], [DATE], "
    "[EMAIL], [URL], [USERNAME], [IP], [IPV4], [IPV6], [ACCOUNTNUMBER], [OTHERPII]. "
    "Preserve all non-PII text exactly. Output only the redacted text.",
)
INSTRUCTION = "Mask all PII:"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ai4privacy/pii-masking-200k", split="train", cache_dir=str(CACHE_DIR))
    ds_seed = int(os.getenv("PII_DATASET_SEED", "42"))
    max_rows = int(os.getenv("PII_DATASET_MAX_ROWS", "0"))
    tag_profile = get_tag_profile()
    lang_filter_raw = os.getenv("PII_LANG", "").strip().lower()
    lang_filter = {x.strip() for x in lang_filter_raw.split(",") if x.strip()} if lang_filter_raw else set()
    rng = random.Random(ds_seed)
    rows = list(range(len(ds)))
    rng.shuffle(rows)

    out = []
    skipped_lang = 0
    for idx in rows:
        ex = ds[idx]
        ex_lang = (ex.get("language") or "").strip().lower()
        if lang_filter and ex_lang not in lang_filter:
            skipped_lang += 1
            continue
        src = (ex.get("source_text") or "").strip()
        tgt = rewrite_bracketed_tags((ex.get("target_text") or "").strip(), profile=tag_profile)
        if not src or not tgt:
            continue
        out.append({
            # Keep the system guidance in the actual supervised prompt because
            # the alpaca prompter does not use a standalone `system` field.
            "system": SYSTEM_PROMPT,
            "instruction": f"{SYSTEM_PROMPT}\n\n{INSTRUCTION}",
            "input": src,
            "output": tgt,
        })
        if max_rows > 0 and len(out) >= max_rows:
            break

    with OUT.open("w", encoding="utf-8") as f:
        for obj in out:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        f"Saved {len(out)} examples to {OUT} "
        f"(seed={ds_seed}, max_rows={max_rows or 'all'}, "
        f"lang_filter={sorted(lang_filter) if lang_filter else 'all'}, "
        f"tag_profile={tag_profile}, skipped_lang={skipped_lang})"
    )

if __name__ == "__main__":
    main()
