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
        return {"macro_p": 0.0, "macro_r": 0.0, "macro_f1": 0.0}
    p = sum(r[1] for r in rows) / len(rows)
    r = sum(r[2] for r in rows) / len(rows)
    f1 = sum(r[3] for r in rows) / len(rows)
    return {"macro_p": p, "macro_r": r, "macro_f1": f1}

def print_confusion(conf, title):
    print(f"\n=== Confusion: {title} ===")
    labels = sorted(set(conf.keys()) | set(x for row in conf.values() for x in row.keys()))
    print("gold\\pred," + ",".join(labels))
    for g in labels:
        row = [str(conf.get(g, {}).get(p, 0)) for p in labels]
        print(f"{g}," + ",".join(row))
