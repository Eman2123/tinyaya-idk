"""Aggregate results: means over seeds, paired bootstrap between recipes, transfer matrix.

Usage:
  python src/stats.py summary                      # mean/min/max over seeds per recipe x lang
  python src/stats.py bootstrap --a R2 --b R3 --langs en ar
  python src/stats.py transfer --recipe R3         # train-lang group x test-lang selective accuracy
Model names follow evaluate.py: <recipe>_<trainlangs joined by '-'>_s<seed> (R0 has no suffix).
"""
import argparse
import re
from glob import glob

import numpy as np
import pandas as pd

from common import load_config, out_path, read_jsonl


def load_all(cfg):
    frames = []
    for f in glob(str(out_path(cfg, "results", "eval_*.csv"))):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    parts = df["model"].str.extract(r"^(R\d)(?:_(.+?))?(?:_s(\d+))?$")
    df["recipe"], df["train"], df["seed"] = parts[0], parts[1], parts[2]
    return df


def cmd_summary(cfg):
    df = load_all(cfg)
    metrics = [c for c in ["coverage", "selective_acc", "hallucination_rate", "over_refusal",
                           "abstain_precision", "abstain_recall"] if c in df]
    g = df.groupby(["recipe", "train", "lang"], dropna=False)[metrics].agg(["mean", "min", "max"])
    out = out_path(cfg, "results", "summary.csv")
    g.to_csv(out)
    print(g.round(3).to_string())


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """a, b: per-question 0/1 arrays on the same questions. Returns (mean diff, 95% CI)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def pred_files(cfg, recipe, lang):
    return sorted(glob(str(out_path(cfg, "results", f"preds_{recipe}_*_{lang}.jsonl")))
                  + glob(str(out_path(cfg, "results", f"preds_{recipe}_{lang}.jsonl"))))


def cmd_bootstrap(cfg, a, b, langs):
    for lang in langs:
        fa, fb = pred_files(cfg, a, lang), pred_files(cfg, b, lang)
        if not fa or not fb:
            print(f"{lang}: missing predictions"); continue
        # use the first seed's file for each; extend by looping over seeds if you want per-seed CIs
        ra = {r["id"]: r for r in read_jsonl(fa[0])}
        rb = {r["id"]: r for r in read_jsonl(fb[0])}
        ids = sorted(set(ra) & set(rb))
        # per-question 'good outcome': answered correctly, or abstained when the answer would be wrong is
        # not scored here; we compare accuracy-on-all (correct) and coverage separately
        acc_d, acc_ci = paired_bootstrap([ra[i]["correct"] for i in ids], [rb[i]["correct"] for i in ids])
        cov_d, cov_ci = paired_bootstrap([not ra[i]["refused"] for i in ids], [not rb[i]["refused"] for i in ids])
        print(f"{lang} {a} - {b}: correct diff {acc_d:+.3f} CI {acc_ci[0]:+.3f}..{acc_ci[1]:+.3f}"
              f" | coverage diff {cov_d:+.3f} CI {cov_ci[0]:+.3f}..{cov_ci[1]:+.3f}"
              f"  ({'within noise' if acc_ci[0] <= 0 <= acc_ci[1] else 'significant'})")


def cmd_transfer(cfg, recipe):
    df = load_all(cfg)
    df = df[df["recipe"] == recipe]
    m = df.pivot_table(index="train", columns="lang", values="selective_acc", aggfunc="mean")
    out = out_path(cfg, "results", f"transfer_{recipe}.csv")
    m.to_csv(out)
    print(m.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    b = sub.add_parser("bootstrap"); b.add_argument("--a", required=True); b.add_argument("--b", required=True)
    b.add_argument("--langs", nargs="+", required=True)
    t = sub.add_parser("transfer"); t.add_argument("--recipe", required=True)
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "summary":
        cmd_summary(cfg)
    elif args.cmd == "bootstrap":
        cmd_bootstrap(cfg, args.a, args.b, args.langs)
    else:
        cmd_transfer(cfg, args.recipe)


if __name__ == "__main__":
    main()
