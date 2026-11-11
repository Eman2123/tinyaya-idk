"""Compute gold and consistency known/unknown labels.

Usage: python src/labels.py --lang en
Writes outputs/labels/<lang>.jsonl with fields:
  gold_known (bool), cons_known (bool), majority (str), agree (int)
"""
import argparse
from collections import Counter

from common import is_match, load_config, normalize, out_path, read_jsonl, write_jsonl


def majority_answer(samples):
    """Cluster by normalized form; return (representative raw answer, count)."""
    norm = [normalize(s) for s in samples]
    cnt = Counter(n for n in norm if n)
    if not cnt:
        return "", 0
    top, c = cnt.most_common(1)[0]
    rep = next(s for s, n in zip(samples, norm) if n == top)
    return rep, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    min_agree = cfg["labels"]["consistency_min_agree"]
    rows = read_jsonl(out_path(cfg, "samples", f"{args.lang}.jsonl"))

    out = []
    for r in rows:
        maj, agree = majority_answer(r["samples"])
        out.append({
            "id": r["id"], "split": r["split"], "question": r["question"],
            "answers": r["answers"], "greedy": r["greedy"],
            "majority": maj, "agree": agree,
            "gold_known": is_match(r["greedy"], r["answers"]),
            "cons_known": agree >= min_agree,
            "majority_correct": is_match(maj, r["answers"]),
        })
    write_jsonl(out_path(cfg, "labels", f"{args.lang}.jsonl"), out)
    k = sum(o["gold_known"] for o in out)
    c = sum(o["cons_known"] for o in out)
    print(f"{args.lang}: n={len(out)} gold_known={k} cons_known={c}")


if __name__ == "__main__":
    main()
