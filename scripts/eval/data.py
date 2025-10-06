# examples/pii_masking/eval/data.py
import random
from datasets import load_dataset

def load_sampled(dataset_name="ai4privacy/pii-masking-200k",
                 split="test", k=100, seed=42):
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        ds = load_dataset(dataset_name, split="train[-10%:]")
    total = len(ds)
    random.seed(seed)
    idxs = random.sample(range(total), k=min(k, total))
    return ds.select(idxs), idxs
