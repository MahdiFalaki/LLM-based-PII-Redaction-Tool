# src/pii_masking/eval/run_eval.py
import os, json, csv, argparse
from collections import defaultdict, Counter

# config is optional; fall back if not present
try:
    from pii_masking.config.config import SYSTEM_PROMPT, N_CTX, CPU_THREADS
except Exception:
    SYSTEM_PROMPT = "You are a data privacy assistant."
    N_CTX = 2048
    CPU_THREADS = None  # auto

from pii_masking.utils.post_processing import normalize_entities, normalize_reference
from pii_masking.infer.hf_infer import HFModel
from pii_masking.infer.gguf_infer import GGUFModel
from pii_masking.eval.data import load_sampled, load_jsonl_custom
from pii_masking.utils.metrics import (
    extract_tag_sequence,
    pairwise_confusion,
    merge_confusion,
    per_tag_prf,
    aggregate_prf,
    print_confusion,
)
from pii_masking.utils.plots import save_confusion_heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--samples", type=int, default=100, help="Used only if --jsonl not provided")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--jsonl", default=None, help="Path to custom JSONL (input/output format)")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--plot", action="store_true", help="Save heatmap PNGs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # data
    if args.jsonl:
        ds, idxs = load_jsonl_custom(args.jsonl)
        print(f"[data] loaded {len(ds)} rows from {args.jsonl}")
    else:
        ds, idxs = load_sampled(k=args.samples, seed=args.seed, split=args.split)
        print(f"[data] sampled {len(ds)} rows from ai4privacy/pii-masking-200k")

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
        ref_norm = normalize_reference(ref)  # same CANON as predictions

        pred_hf_raw = hf.generate(SYSTEM_PROMPT, src)
        pred_hf_norm = normalize_entities(pred_hf_raw, system=SYSTEM_PROMPT, user_text=src)

        pred_gg_raw = gg.generate(SYSTEM_PROMPT, src)
        pred_gg_norm = normalize_entities(pred_gg_raw, system=SYSTEM_PROMPT, user_text=src)

        ref_seq = extract_tag_sequence(ref_norm)
        hf_seq = extract_tag_sequence(pred_hf_norm)
        gg_seq = extract_tag_sequence(pred_gg_norm)

        merge_confusion(conf_hf, pairwise_confusion(ref_seq, hf_seq))
        merge_confusion(conf_gg, pairwise_confusion(ref_seq, gg_seq))

        prf_hf_all.extend(per_tag_prf(ref_seq, hf_seq))
        prf_gg_all.extend(per_tag_prf(ref_seq, gg_seq))

        rows.append({
            "id": int(idxs[i]),
            "source_text": src,
            "target_text_norm": ref_norm,
            "hf_raw": pred_hf_raw.strip(),
            "hf_norm": pred_hf_norm.strip(),
            "gguf_raw": pred_gg_raw.strip(),
            "gguf_norm": pred_gg_norm.strip(),
        })

        if (i + 1) % 10 == 0:
            print(f"...{i+1}/{len(ds)}")

    # outputs
    jsonl_path = os.path.join(args.outdir, "eval_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = os.path.join(args.outdir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summaries
    print_confusion(conf_hf, "HF merged (GPU)")
    print_confusion(conf_gg, "GGUF quantized (CPU)")

    agg_hf = aggregate_prf(prf_hf_all)
    agg_gg = aggregate_prf(prf_gg_all)
    print(f"\n=== Macro P/R/F1: HF merged (GPU) ===\nP={agg_hf['macro_p']:.3f} R={agg_hf['macro_r']:.3f} F1={agg_hf['macro_f1']:.3f}")
    print(f"\n=== Macro P/R/F1: GGUF quantized (CPU) ===\nP={agg_gg['macro_p']:.3f} R={agg_gg['macro_r']:.3f} F1={agg_gg['macro_f1']:.3f}")

    if args.plot:
        save_confusion_heatmap(conf_hf, "HF merged (GPU)", os.path.join(args.outdir, "confusion_hf.png"))
        save_confusion_heatmap(conf_gg, "GGUF quantized (CPU)", os.path.join(args.outdir, "confusion_gguf.png"))

    print(f"\nSaved:\n  {jsonl_path}\n  {csv_path}")
    if args.plot:
        print(f"  {os.path.join(args.outdir, 'confusion_hf.png')}\n  {os.path.join(args.outdir, 'confusion_gguf.png')}")

if __name__ == "__main__":
    main()
