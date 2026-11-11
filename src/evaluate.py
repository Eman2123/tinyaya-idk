"""Evaluate a model (base or LoRA adapter) on held-out test questions.

Usage:
  python src/evaluate.py --name R0 --langs en ar              # base model
  python src/evaluate.py --name R3_en-ar_s1 --adapter outputs/adapters/R3_en-ar_s1 --langs en ar es
Writes outputs/results/eval_<name>.csv. The transfer matrix = train langs (in the name) x test langs.
Over-refusal needs the base model's per-question correctness: run R0 first.
"""
import argparse

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm

from common import is_match, is_refusal, load_config, load_questions, out_path, read_jsonl
from sample import generate, load_model
from common import build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok, model = load_model(cfg)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    bs, max_new = cfg["sampling"]["batch_size"], cfg["model"]["max_new_tokens"]

    rows_out = []
    for lang in args.langs:
        test = [r for r in load_questions(cfg, lang) if r["split"] == "test"]
        preds = []
        for i in tqdm(range(0, len(test), bs), desc=f"eval {lang}"):
            batch = test[i : i + bs]
            preds += generate(tok, model, [build_prompt(cfg, r["question"], lang) for r in batch],
                              max_new, sample=False)

        # base model correctness on the same test questions (for over-refusal)
        base_path = out_path(cfg, "results", f"preds_R0_{lang}.jsonl")
        base_correct = None
        if args.name != "R0" and base_path.exists():
            base = {r["id"]: r["correct"] for r in read_jsonl(base_path)}
            base_correct = [base.get(r["id"]) for r in test]

        recs = []
        for r, p in zip(test, preds):
            ref = is_refusal(p, cfg, lang)
            recs.append({"id": r["id"], "pred": p, "refused": ref,
                         "correct": (not ref) and is_match(p, r["answers"])})
        # save per-question predictions for every model (used by stats.py bootstrap)
        from common import write_jsonl
        write_jsonl(out_path(cfg, "results", f"preds_{args.name}_{lang}.jsonl"), recs)

        n = len(recs)
        answered = [x for x in recs if not x["refused"]]
        correct = sum(x["correct"] for x in recs)
        m = {
            "model": args.name, "lang": lang, "n": n,
            "coverage": len(answered) / n,
            "selective_acc": (correct / len(answered)) if answered else float("nan"),
            "hallucination_rate": (len(answered) - correct) / n,
        }
        if base_correct is not None:
            # unknown = base model got it wrong
            unk = [not bool(b) for b in base_correct]
            kn = [bool(b) for b in base_correct]
            refused = [x["refused"] for x in recs]
            m["over_refusal"] = sum(r and k for r, k in zip(refused, kn)) / max(1, sum(kn))
            m["abstain_recall"] = sum(r and u for r, u in zip(refused, unk)) / max(1, sum(unk))
            m["abstain_precision"] = sum(r and u for r, u in zip(refused, unk)) / max(1, sum(refused))
        rows_out.append(m)
        print(m)

    df = pd.DataFrame(rows_out)
    df.to_csv(out_path(cfg, "results", f"eval_{args.name}.csv"), index=False)


if __name__ == "__main__":
    main()
