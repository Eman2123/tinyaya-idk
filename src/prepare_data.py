"""Build data/raw/<lang>.jsonl from MKQA (same questions across languages -> clean transfer comparison).

Usage: python src/prepare_data.py --langs en es ar ms --n 1500
NOTE: MKQA has no Indonesian or Swahili. Use ms (Malay) as an Indonesian-like option, or translate
English questions yourself for other languages (see README). This loader is untested against the live
dataset, so check the printed example and the counts before trusting it.

Only short factoid answers are kept (entity, number, date, short_phrase); yes/no, long answers and
unanswerable items are dropped. The question ids are identical across languages.
"""
import argparse
import random

from datasets import load_dataset

from common import ROOT, load_config, write_jsonl

KEEP_TYPES = {"entity", "number", "date", "number_with_unit", "short_phrase"}


def gold_for(item, lang):
    """Return list of acceptable answer strings for this language, or None."""
    answers = item["answers"].get(lang) or []
    golds = []
    for a in answers:
        if a.get("type") not in KEEP_TYPES:
            return None
        if a.get("text"):
            golds.append(str(a["text"]))
        golds += [str(x) for x in (a.get("aliases") or []) if x]
    return golds or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    try:
        ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True)
    except Exception as e:  # newer `datasets` versions dropped script support
        print("direct load failed:", e)
        ds = load_dataset("apple/mkqa", split="train", revision="refs/convert/parquet")

    items = []
    for it in ds:
        ok = all(gold_for(it, l) and it["queries"].get(l) for l in args.langs)
        if ok:
            items.append(it)
    print(f"{len(items)} questions answerable in all of {args.langs}")
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    for lang in args.langs:
        rows = [{"id": f"mkqa_{it['example_id']}", "question": it["queries"][lang],
                 "answers": gold_for(it, lang)} for it in items]
        write_jsonl(ROOT / cfg["paths"]["raw"] / f"{lang}.jsonl", rows)
        print(lang, len(rows), "e.g.", rows[0])


if __name__ == "__main__":
    main()
