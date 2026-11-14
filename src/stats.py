"""Aggregate results: means over seeds, paired bootstrap between recipes, transfer matrix.

Usage:
  python src/stats.py summary                      # mean/min/max over seeds per recipe x lang
  python src/stats.py bootstrap --a R4 --b R3 --langs en es ar ms [--seeds 1 2 3]
  python src/stats.py transfer --recipe R4 [--metric hallucination_rate]   # train lang x test lang
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


def pred_path(cfg, recipe, train, seed, lang):
    return out_path(cfg, "results", f"preds_{recipe}_{train}_s{seed}_{lang}.jsonl")


def cmd_bootstrap(cfg, a, b, langs, seeds):
    """Paired bootstrap of recipe a minus recipe b, per language, for per-language adapters (trained AND tested on
    that language). Outcomes are averaged over the seeds both recipes have, then questions are resampled, so the
    CI reflects question-level noise; the seed spread is in `summary`."""
    for lang in langs:
        per = {}
        for rec in (a, b):
            by_id = {}
            for sd in seeds:
                f = pred_path(cfg, rec, lang, sd, lang)
                if not f.exists():
                    continue
                for r in read_jsonl(f):
                    by_id.setdefault(r["id"], []).append(r)
            per[rec] = by_id
        ids = sorted(set(per[a]) & set(per[b]))
        if not ids:
            print(f"{lang}: missing predictions for {a} or {b}"); continue
        nseeds = min(min(len(per[a][i]) for i in ids), min(len(per[b][i]) for i in ids))

        def m(rec, fn):
            return [float(np.mean([fn(r) for r in per[rec][i][:nseeds]])) for i in ids]
        metrics = {"correct": lambda r: r["correct"],
                   "hallucination": lambda r: (not r["refused"]) and (not r["correct"]),
                   "coverage": lambda r: not r["refused"]}
        parts = []
        for name, fn in metrics.items():
            d, ci = paired_bootstrap(m(a, fn), m(b, fn))
            tag = "within noise" if ci[0] <= 0 <= ci[1] else "significant"
            parts.append(f"{name} {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] ({tag})")
        print(f"{lang} {a}-{b} (n={len(ids)} questions, {nseeds} seed(s)): " + " | ".join(parts))


def cmd_transfer(cfg, recipe, metric="selective_acc"):
    df = load_all(cfg)
    df = df[df["recipe"] == recipe]
    m = df.pivot_table(index="train", columns="lang", values=metric, aggfunc="mean")
    out = out_path(cfg, "results", f"transfer_{recipe}_{metric}.csv")
    m.to_csv(out)
    print(m.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    b = sub.add_parser("bootstrap"); b.add_argument("--a", required=True); b.add_argument("--b", required=True)
    b.add_argument("--langs", nargs="+", required=True)
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    t = sub.add_parser("transfer"); t.add_argument("--recipe", required=True)
    t.add_argument("--metric", default="selective_acc")
    args = ap.parse_args()
    cfg = load_config()
    if args.cmd == "summary":
        cmd_summary(cfg)
    elif args.cmd == "bootstrap":
        cmd_bootstrap(cfg, args.a, args.b, args.langs, args.seeds)
    else:
        cmd_transfer(cfg, args.recipe, args.metric)


if __name__ == "__main__":
    main()
