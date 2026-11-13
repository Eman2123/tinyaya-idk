"""Train a logistic-regression probe on the anchor language (gold labels) and transfer it.

Usage: python src/probe.py --anchor en --targets es ar ms
Optional: --layer_index 1 (index into sampling.probe_layers), --C 0.05, --sweep (layer x C table, anchor only)

Design (no leakage):
  * Probe is trained on anchor build+val with 5-fold CV. The anchor build/val rows get out-of-fold
    scores (so R4 anchor labels are not in-sample); anchor test rows and all target-language rows
    get the final model (fit on all anchor build+val).
  * Threshold = Youden J (TPR - FPR) on the out-of-fold anchor scores (balanced, not accuracy-driven).
  * Reported on the TEST split only. Headline metrics: known-class P/R/F1, balanced accuracy, AUROC,
    with bootstrap 95% CI. An 'all_unknown' baseline row is included because accuracy/F1_unknown alone
    are inflated by the small known share.
Adds probe_score / probe_known to outputs/labels/<lang>.jsonl and writes outputs/results/label_quality.csv.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_config, out_path, read_jsonl, write_jsonl


def load_lang(cfg, lang, layer_key):
    labels = read_jsonl(out_path(cfg, "labels", f"{lang}.jsonl"))
    npz = np.load(out_path(cfg, "hidden", f"{lang}.npz"))
    assert list(npz["ids"]) == [r["id"] for r in labels], "id order mismatch"
    return labels, npz[layer_key]


def make_clf(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def youden_threshold(y, scores):
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[np.argmax(tpr - fpr)])


def metrics(y, pred, scores=None):
    """y, pred: bool arrays where True = known."""
    d = {
        "acc": accuracy_score(y, pred),
        "bal_acc": balanced_accuracy_score(y, pred),
        "known_P": precision_score(y, pred, zero_division=0),
        "known_R": recall_score(y, pred, zero_division=0),
        "known_F1": f1_score(y, pred, zero_division=0),
        "f1_unknown": f1_score(~y, ~pred, zero_division=0),
        "pred_known_rate": float(np.mean(pred)),
        "gold_known_rate": float(np.mean(y)),
        "n": len(y),
    }
    d["auroc"] = roc_auc_score(y, scores) if scores is not None and len(set(y)) > 1 else np.nan
    return d


def bootstrap_ci(y, pred, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    f1s, aucs, bals = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        f1s.append(f1_score(y[i], pred[i], zero_division=0))
        bals.append(balanced_accuracy_score(y[i], pred[i]))
        if scores is not None:
            aucs.append(roc_auc_score(y[i], scores[i]))
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
    return {"known_F1_lo": ci(f1s)[0], "known_F1_hi": ci(f1s)[1],
            "bal_acc_lo": ci(bals)[0], "bal_acc_hi": ci(bals)[1],
            "auroc_lo": ci(aucs)[0], "auroc_hi": ci(aucs)[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--layer_index", type=int, default=None)
    ap.add_argument("--C", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="print AUROC (OOF, anchor build+val) for layer x C")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    layers = cfg["sampling"]["probe_layers"]
    li = cfg["probe"]["layer_index"] if args.layer_index is None else args.layer_index
    C = cfg["probe"]["C"] if args.C is None else args.C
    min_agree = cfg["labels"]["consistency_min_agree"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    if args.sweep:
        print("sweep: OOF AUROC on anchor build+val (gold labels)")
        for k, L in enumerate(layers):
            a_labels, a_h = load_lang(cfg, args.anchor, f"layer{L}")
            tr = np.array([r["split"] in ("build", "val") for r in a_labels])
            y = np.array([r["gold_known"] for r in a_labels])
            for c in [0.001, 0.01, 0.05, 0.1, 1.0]:
                oof = cross_val_predict(make_clf(c), a_h[tr], y[tr], cv=skf, method="predict_proba")[:, 1]
                print(f"  layer_index={k} (layer {L}) C={c}: AUROC={roc_auc_score(y[tr], oof):.3f}")
        return

    layer_key = f"layer{layers[li]}"
    print(f"probe: layer {layers[li]} (index {li}), C={C}, class_weight=balanced")
    a_labels, a_h = load_lang(cfg, args.anchor, layer_key)
    split = np.array([r["split"] for r in a_labels])
    y_a = np.array([r["gold_known"] for r in a_labels])
    tr = (split == "build") | (split == "val")

    oof = cross_val_predict(make_clf(C), a_h[tr], y_a[tr], cv=skf, method="predict_proba")[:, 1]
    thr = youden_threshold(y_a[tr], oof)
    clf = make_clf(C).fit(a_h[tr], y_a[tr])
    print(f"anchor OOF AUROC (build+val) = {roc_auc_score(y_a[tr], oof):.3f}   Youden threshold = {thr:.3f}")

    table = []
    for lang in [args.anchor] + args.targets:
        labels, h = (a_labels, a_h) if lang == args.anchor else load_lang(cfg, lang, layer_key)
        scores = clf.predict_proba(h)[:, 1]
        if lang == args.anchor:
            scores[tr] = oof   # out-of-fold for anchor build/val rows
        for r, s in zip(labels, scores):
            r["probe_score"] = float(s)
            r["probe_known"] = bool(s >= thr)
        write_jsonl(out_path(cfg, "labels", f"{lang}.jsonl"), labels)

        test = np.array([r["split"] == "test" for r in labels])
        g = np.array([r["gold_known"] for r in labels])[test]
        agree = np.array([r["agree"] for r in labels])[test]
        p = np.array([r["probe_known"] for r in labels])[test]
        s_te = scores[test]

        rows = [
            ("all_unknown", np.zeros_like(g), None),
            (f"cons>={min_agree}", agree >= min_agree, agree.astype(float)),
            ("cons>=3", agree >= 3, agree.astype(float)),
            ("cons>=5", agree >= 5, agree.astype(float)),
            ("probe", p, s_te),
        ]
        for name, pred, sc in rows:
            table.append({"lang": lang, "source": name, **metrics(g, pred, sc),
                          **(bootstrap_ci(g, pred, sc) if name in ("probe", f"cons>={min_agree}") else {})})

    df = pd.DataFrame(table)
    df.to_csv(out_path(cfg, "results", "label_quality.csv"), index=False)
    show = ["lang", "source", "n", "gold_known_rate", "pred_known_rate", "known_P", "known_R",
            "known_F1", "bal_acc", "auroc", "auroc_lo", "auroc_hi"]
    print(df[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
