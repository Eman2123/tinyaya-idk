"""Train a logistic-regression probe on the anchor language (gold labels) and transfer it.

Usage: python src/probe.py --anchor en --targets es ar id sw
Adds probe_score / probe_known to outputs/labels/<lang>.jsonl and writes
outputs/results/label_quality.csv (RQ1 table).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import load_config, out_path, read_jsonl, write_jsonl


def load_lang(cfg, lang, layer_key):
    labels = read_jsonl(out_path(cfg, "labels", f"{lang}.jsonl"))
    npz = np.load(out_path(cfg, "hidden", f"{lang}.npz"))
    assert list(npz["ids"]) == [r["id"] for r in labels], "id order mismatch"
    return labels, npz[layer_key]


def best_threshold(y_true, scores):
    """Threshold maximizing F1 of the 'unknown' class on anchor validation data."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred_known = scores >= t
        f = f1_score(~y_true, ~pred_known, zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return best_t


def quality(y_gold, y_pred, scores=None):
    d = {
        "acc": accuracy_score(y_gold, y_pred),
        "f1_unknown": f1_score(~y_gold, ~y_pred, zero_division=0),
        "n": len(y_gold),
        "gold_known_rate": float(np.mean(y_gold)),
    }
    if scores is not None and len(set(y_gold)) > 1:
        d["auroc"] = roc_auc_score(y_gold, scores)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    layers = cfg["sampling"]["probe_layers"]
    layer_key = f"layer{layers[cfg['probe']['layer_index']]}"

    a_labels, a_h = load_lang(cfg, args.anchor, layer_key)
    split = np.array([r["split"] for r in a_labels])
    y = np.array([r["gold_known"] for r in a_labels])

    scaler = StandardScaler().fit(a_h[split == "build"])
    clf = LogisticRegression(C=cfg["probe"]["C"], max_iter=2000)
    clf.fit(scaler.transform(a_h[split == "build"]), y[split == "build"])

    val = split == "val"
    thr = best_threshold(y[val], clf.predict_proba(scaler.transform(a_h[val]))[:, 1])
    print(f"anchor threshold = {thr:.2f}")

    table = []
    for lang in [args.anchor] + args.targets:
        labels, h = (a_labels, a_h) if lang == args.anchor else load_lang(cfg, lang, layer_key)
        scores = clf.predict_proba(scaler.transform(h))[:, 1]
        for r, s in zip(labels, scores):
            r["probe_score"] = float(s)
            r["probe_known"] = bool(s >= thr)
        write_jsonl(out_path(cfg, "labels", f"{lang}.jsonl"), labels)

        test = np.array([r["split"] == "test" for r in labels])
        g = np.array([r["gold_known"] for r in labels])
        c = np.array([r["cons_known"] for r in labels])
        p = np.array([r["probe_known"] for r in labels])
        table.append({"lang": lang, "source": "consistency", **quality(g[test], c[test])})
        table.append({"lang": lang, "source": "probe", **quality(g[test], p[test], scores[test])})

    df = pd.DataFrame(table)
    df.to_csv(out_path(cfg, "results", "label_quality.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
