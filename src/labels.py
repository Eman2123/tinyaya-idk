"""Compute gold and consistency known/unknown labels.

Usage: python src/labels.py --lang en
Writes outputs/labels/<lang>.jsonl with fields:
  gold_known (bool), cons_known (bool), majority (str), agree (int), majority_correct (bool)
Prints counts plus precision of the consistency 'known' label against gold.
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

    n = len(out)
    k = sum(o["gold_known"] for o in out)
    c = sum(o["cons_known"] for o in out)
    both = sum(o["gold_known"] and o["cons_known"] for o in out)
    maj_ok = sum(o["majority_correct"] for o in out if o["cons_known"])
    splits = Counter(o["split"] for o in out)
    print(f"{args.lang}: n={n} gold_known={k} ({k/n:.1%}) cons_known={c} ({c/n:.1%}) splits={dict(splits)}")
    if c:
        print(f"   cons_known precision vs gold(greedy)={both/c:.1%}  "
              f"vs gold(majority)={maj_ok/c:.1%}  recall of gold_known={both/max(k,1):.1%}")
    agree_hist = Counter(o["agree"] for o in out)
    print("   agree histogram:", {a: agree_hist[a] for a in sorted(agree_hist)})


if __name__ == "__main__":
    main()
