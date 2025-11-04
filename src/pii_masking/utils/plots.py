# src/pii_masking/utils/plots.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _labels_and_matrix(conf):
    labels = sorted(set(conf.keys()) | set(x for row in conf.values() for x in row.keys()))
    mat = [[conf.get(g, {}).get(p, 0) for p in labels] for g in labels]
    return labels, mat

def save_confusion_heatmap(conf, title, outpath, normalize="row"):
    labels, mat = _labels_and_matrix(conf)
    if not labels:
        print(f"[warn] No labels for heatmap: {title}")
        return

    arr = np.array(mat, dtype=float)

    if normalize == "row":
        row_sums = arr.sum(axis=1, keepdims=True)
        norm = np.zeros_like(arr, dtype=float)
        np.divide(arr, row_sums, out=norm, where=row_sums != 0)
        data = norm
    else:
        data = arr

    h = max(5, len(labels) * 0.35)
    w = max(6, len(labels) * 0.35)
    plt.figure(figsize=(w, h))

    im = plt.imshow(data, aspect='auto')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved heatmap: {outpath}")
