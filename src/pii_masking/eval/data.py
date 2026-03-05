# src/pii_masking/eval/data.py
import json
import os
import random
from pathlib import Path
from datasets import load_dataset

_DEFAULT_CACHE = Path(os.getenv("PII_DATASETS_CACHE", Path.cwd() / ".cache" / "hf_datasets"))

def _load(split: str):
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    return load_dataset("ai4privacy/pii-masking-200k", split=split, cache_dir=str(_DEFAULT_CACHE))

def load_sampled(k: int = 100, seed: int = 42, split: str = "train"):
    if split == "train":
        ds = _load("train")
    else:
        base = _load("train")
        holdout = base.train_test_split(test_size=0.02, seed=seed)
        ds = holdout["test"]
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), k=min(k, len(ds)))
    subset = []
    for i in idxs:
        ex = ds[i]
        src = ex.get("source_text") or ex.get("input") or ex.get("text") or ""
        tgt = ex.get("target_text") or ex.get("output") or ""
        subset.append({"source_text": src, "target_text": tgt})
    return subset, idxs

def load_jsonl_custom(path: str):
    exs, idxs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            src = obj.get("input") or obj.get("source_text") or ""
            tgt = obj.get("output") or obj.get("target_text") or ""
            exs.append({"source_text": src, "target_text": tgt})
            idxs.append(i)
    return exs, idxs
