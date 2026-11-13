"""Optional LLM judge for answer equivalence (use for non-Latin scripts where exact match is noisy).

The judge is only a validator for answer equivalence, never the core model. Confirm on the hackathon
page that an external judge is allowed before using it.

Usage: python src/judge.py --lang ar [--judge-model claude-sonnet-5-5]
Needs: pip install anthropic ; export ANTHROPIC_API_KEY=...
Rewrites outputs/labels/<lang>.jsonl: gold_known = exact match OR judge says equivalent
(original value kept in gold_known_em). Also writes outputs/judge_check_<lang>.jsonl with 50 items
for the hand spot-check the plan asks for.
"""
import argparse
import json
import random

from tqdm import tqdm

from common import is_match, load_config, out_path, read_jsonl, write_jsonl

PROMPT = """You are checking a question-answering system.
Question: {q}
Reference answers: {golds}
System answer: {pred}

Does the system answer state the same fact as one of the reference answers (allow spelling, script,
transliteration and formatting differences)? Reply with exactly one word: YES or NO."""


def judge_one(client, model, q, golds, pred):
    msg = client.messages.create(
        model=model, max_tokens=5, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(q=q, golds=json.dumps(golds, ensure_ascii=False), pred=pred)}],
    )
    return msg.content[0].text.strip().upper().startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--judge-model", default="claude-sonnet-5-5")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    cfg = load_config(args.config)
    path = out_path(cfg, "labels", f"{args.lang}.jsonl")
    rows = read_jsonl(path)

    checked = []
    for r in tqdm(rows, desc=f"judge {args.lang}"):
        r.setdefault("gold_known_em", r["gold_known"])
        if r["gold_known_em"] or not r["greedy"].strip():
            r["gold_known"] = r["gold_known_em"]
            continue
        try:
            r["gold_known"] = judge_one(client, args.judge_model, r["question"], r["answers"], r["greedy"])
        except Exception as e:
            print("judge error:", e)
            r["gold_known"] = r["gold_known_em"]
        checked.append({"id": r["id"], "question": r["question"], "gold": r["answers"],
                        "pred": r["greedy"], "em": r["gold_known_em"], "judge": r["gold_known"]})

    write_jsonl(path, rows)
    random.Random(0).shuffle(checked)
    write_jsonl(out_path(cfg, f"judge_check_{args.lang}.jsonl"), checked[:50])
    flipped = sum(1 for r in rows if r["gold_known"] != r["gold_known_em"])
    print(f"{args.lang}: judge changed {flipped} labels; hand-check outputs/judge_check_{args.lang}.jsonl")


if __name__ == "__main__":
    main()
