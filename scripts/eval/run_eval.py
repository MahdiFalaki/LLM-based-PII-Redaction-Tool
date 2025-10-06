# scripts/eval/run_eval.py
import os, sys, json, csv, argparse, random, re
from collections import defaultdict, Counter

# --- bootstrap package root on sys.path ---
import sys, pathlib
PKG_ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../examples
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pii_masking.config import DEFAULT_HF_DIR, DEFAULT_GGUF, SYSTEM_PROMPT, N_CTX, CPU_THREADS
from pii_masking.scripts.utils.post_processing import normalize_entities
from pii_masking.scripts.ref_normalize import normalize_reference
from pii_masking.scripts.models.hf_infer import HFModel
from pii_masking.scripts.models.gguf_infer import GGUFModel
from pii_masking.scripts.eval.data import load_sampled
from pii_masking.scripts.eval.metrics import (
    extract_tag_sequence, pairwise_confusion, merge_confusion,
    per_tag_prf, aggregate_prf, print_confusion, print_prf
)

# --- minimal metrics (self-contained) ---
TAG_RE = re.compile(r"\[([A-Z0-9_]+)\]")

def extract_tag_sequence(text: str):
    """Extract sequence of bracketed tags in order."""
    return TAG_RE.findall(text)

def pairwise_confusion(ref_seq, pred_seq):
    """
    Build confusion counts over pairwise alignment by position.
    If lengths differ, compare up to min length and count leftovers as misses/extras.
    """
    conf = defaultdict(Counter)
    m = min(len(ref_seq), len(pred_seq))
    for i in range(m):
        conf[ref_seq[i]][pred_seq[i]] += 1
    # handle tail (optional): count missed predictions/ref if you want
    # for simplicity we’ll ignore tail mismatches in confusion; PRF below uses sets
    return conf

def merge_confusion(agg, cur):
    for gold, row in cur.items():
        for pred, c in row.items():
            agg.setdefault(gold, Counter())
            agg[gold][pred] += c

def per_tag_prf(ref_seq, pred_seq):
    """
    Per-tag precision/recall/F1 using set membership (bag-of-tags).
    Returns list of (tag, P, R, F1, TP, FP, FN).
    """
    ref_counts = Counter(ref_seq)
    pred_counts = Counter(pred_seq)
    tags = set(ref_counts) | set(pred_counts)
    rows = []
    for t in tags:
        tp = min(ref_counts[t], pred_counts[t])
        fp = max(0, pred_counts[t] - tp)
        fn = max(0, ref_counts[t] - tp)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r / (p + r) if (p + r) else 0.0
        rows.append((t, p, r, f1, tp, fp, fn))
    return rows

def aggregate_prf(rows):
    """
    Macro-average P/R/F1 across tags.
    rows is list of (tag, P, R, F1, TP, FP, FN)
    """
    if not rows:
        return {"macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0}
    p = sum(r[1] for r in rows) / len(rows)
    r = sum(r[2] for r in rows) / len(rows)
    f1 = sum(r[3] for r in rows) / len(rows)
    return {"macro_p": p, "macro_r": r, "macro_f1": f1}

def print_confusion(conf, title):
    print(f"\n=== Confusion: {title} ===")
    # collect label set
    labels = sorted(set(conf.keys()) | set(x for row in conf.values() for x in row.keys()))
    # header
    print("gold\\pred," + ",".join(labels))
    for g in labels:
        row = [str(conf.get(g, {}).get(p, 0)) for p in labels]
        print(f"{g}," + ",".join(row))

def print_prf(agg, title):
    print(f"\n=== Macro P/R/F1: {title} ===")
    print(f"P={agg['macro_p']:.3f} R={agg['macro_r']:.3f} F1={agg['macro_f1']:.3f}")

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train", choices=["train"])  # dataset only has train
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds, idxs = load_sampled(k=args.samples, seed=args.seed, split=args.split)

    print("Loading HF model…")
    hf = HFModel(args.hf_dir)
    print("Loading GGUF model…")
    gg = GGUFModel(args.gguf, n_ctx=N_CTX, n_threads=CPU_THREADS)

    rows = []
    conf_hf, conf_gg = {}, {}
    prf_hf_all, prf_gg_all = [], []

    for i, ex in enumerate(ds):
        src = ex["source_text"]
        ref = ex["target_text"]
        ref_norm = normalize_reference(ref)

        pred_hf_raw  = hf.generate(SYSTEM_PROMPT, src)
        pred_hf_norm = normalize_entities(pred_hf_raw, system=SYSTEM_PROMPT, user_text=src)

        pred_gg_raw  = gg.generate(SYSTEM_PROMPT, src)
        pred_gg_norm = normalize_entities(pred_gg_raw, system=SYSTEM_PROMPT, user_text=src)

        ref_seq = extract_tag_sequence(ref_norm)
        hf_seq  = extract_tag_sequence(pred_hf_norm)
        gg_seq  = extract_tag_sequence(pred_gg_norm)

        merge_confusion(conf_hf, pairwise_confusion(ref_seq, hf_seq))
        merge_confusion(conf_gg, pairwise_confusion(ref_seq, gg_seq))

        prf_hf_all.extend(per_tag_prf(ref_seq, hf_seq))
        prf_gg_all.extend(per_tag_prf(ref_seq, gg_seq))

        rows.append({
            "id": int(idxs[i]),
            "source_text": src,
            "target_text_norm": ref_norm,
            "hf_raw":  pred_hf_raw.strip(),
            "hf_norm": pred_hf_norm.strip(),
            "gguf_raw":  pred_gg_raw.strip(),
            "gguf_norm": pred_gg_norm.strip(),
        })

        if (i + 1) % 10 == 0:
            print(f"...{i+1}/{len(ds)}")

    # write outputs
    jsonl_path = os.path.join(args.outdir, "eval_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = os.path.join(args.outdir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # print summaries
    print_confusion(conf_hf, "HF merged (GPU)")
    print_confusion(conf_gg, "GGUF quantized (CPU)")

    print_prf(aggregate_prf(prf_hf_all), "HF merged (GPU)")
    print_prf(aggregate_prf(prf_gg_all), "GGUF quantized (CPU)")

    print(f"\nSaved:\n  {jsonl_path}\n  {csv_path}")

if __name__ == "__main__":
    main()
