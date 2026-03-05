# src/pii_masking/utils/metrics.py
import re
from collections import defaultdict, Counter

TAG_RE = re.compile(r"\[([A-Z0-9_]+)\]")

# minimal alias map; expand if needed
ALIAS = {
    "HUMANITIES": "JOBAREA",
}

def _canonical(t: str, allowed: set | None = None):
    t = ALIAS.get(t.upper(), t.upper())
    if allowed is not None and t not in allowed:
        return None
    return t

def extract_tag_sequence(text: str, allowed: set | None = None):
    seq = []
    for raw in TAG_RE.findall(text):
        t = _canonical(raw, allowed=allowed)
        if t is not None:
            seq.append(t)
    return seq

def pairwise_confusion(ref_seq, pred_seq):
    conf = defaultdict(Counter)
    m = min(len(ref_seq), len(pred_seq))
    for i in range(m):
        conf[ref_seq[i]][pred_seq[i]] += 1
    for i in range(m, len(ref_seq)):
        conf[ref_seq[i]]["<MISSING>"] += 1
    for i in range(m, len(pred_seq)):
        conf["<SPURIOUS>"][pred_seq[i]] += 1
    return conf

def merge_confusion(agg, cur):
    for gold, row in cur.items():
        for pred, c in row.items():
            agg.setdefault(gold, Counter())
            agg[gold][pred] += c

def per_tag_prf(ref_seq, pred_seq):
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
    if not rows:
        return {"macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0, "micro_p": 0.0, "micro_r": 0.0, "micro_f1": 0.0}

    per_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for tag, _p, _r, _f1, tp, fp, fn in rows:
        per_tag[tag]["tp"] += tp
        per_tag[tag]["fp"] += fp
        per_tag[tag]["fn"] += fn

    macro_p = macro_r = macro_f1 = 0.0
    for stats in per_tag.values():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        macro_p += p
        macro_r += r
        macro_f1 += f1

    n = max(1, len(per_tag))
    macro_p /= n
    macro_r /= n
    macro_f1 /= n

    tp_all = sum(v["tp"] for v in per_tag.values())
    fp_all = sum(v["fp"] for v in per_tag.values())
    fn_all = sum(v["fn"] for v in per_tag.values())
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
    }

def print_confusion(conf, title):
    print(f"\n=== Confusion: {title} ===")
    labels = sorted(set(conf.keys()) | set(x for row in conf.values() for x in row.keys()))
    print("gold\\pred," + ",".join(labels))
    for g in labels:
        row = [str(conf.get(g, {}).get(p, 0)) for p in labels]
        print(f"{g}," + ",".join(row))
