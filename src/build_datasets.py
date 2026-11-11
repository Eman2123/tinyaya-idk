"""Build SFT datasets for recipes R1-R4 from labelled build-split questions.

Usage: python src/build_datasets.py --langs en es ar
Writes data/processed/<recipe>/<lang>.jsonl with {prompt, completion, known}.

R1 needs teacher-written trap questions in data/raw/trap_<lang>.jsonl
(one {"question": "..."} per line). R1 is skipped if the file is missing.
Hook CuratorKIT in here if the hackathon rules require it.
"""
import argparse
import random

from common import ROOT, build_prompt, load_config, read_jsonl, write_jsonl


def make_examples(cfg, lang, labels, known_key, seed=0):
    """Known -> model's own majority answer; unknown -> refusal. Balance classes."""
    rng = random.Random(seed)
    refusal = cfg["refusals"][lang]
    known, unknown = [], []
    for r in labels:
        if r["split"] != "build":
            continue
        is_known = r[known_key]
        if is_known:
            # use the model's own majority answer; skip if it is empty
            if r["majority"].strip():
                known.append({"prompt": build_prompt(cfg, r["question"]),
                              "completion": r["majority"].strip(), "known": True, "id": r["id"]})
        else:
            unknown.append({"prompt": build_prompt(cfg, r["question"]),
                            "completion": refusal, "known": False, "id": r["id"]})
    share = cfg["training"]["known_share"]
    # n_known / (n_known + n_unknown) = share, using as many examples as possible
    n_known = min(len(known), int(len(unknown) * share / (1 - share)) if share < 1 else len(known))
    n_unknown = min(len(unknown), int(n_known * (1 - share) / share)) if share > 0 else len(unknown)
    rng.shuffle(known)
    rng.shuffle(unknown)
    out = known[:n_known] + unknown[:n_unknown]
    rng.shuffle(out)
    return out


def trap_examples(cfg, lang, labels):
    """R1: teacher-written trap questions -> refusal, plus known questions answered normally."""
    path = ROOT / cfg["paths"]["raw"] / f"trap_{lang}.jsonl"
    if not path.exists():
        print(f"[R1] {path} missing, skipping {lang}")
        return None
    traps = read_jsonl(path)
    refusal = cfg["refusals"][lang]
    ex = [{"prompt": build_prompt(cfg, t["question"]), "completion": refusal, "known": False}
          for t in traps]
    # answerable side: questions with a gold answer, target = gold answer (the usual shortcut)
    ans = [{"prompt": build_prompt(cfg, r["question"]), "completion": r["answers"][0], "known": True}
           for r in labels if r["split"] == "build"][: len(ex)]
    out = ex + ans
    random.Random(0).shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    recipes = {"R2": "gold_known", "R3": "cons_known", "R4": "probe_known"}
    for lang in args.langs:
        labels = read_jsonl(ROOT / cfg["paths"]["outputs"] / "labels" / f"{lang}.jsonl")
        for rec, key in recipes.items():
            if key not in labels[0]:
                print(f"[{rec}] {key} missing for {lang}, run probe.py first")
                continue
            ex = make_examples(cfg, lang, labels, key)
            write_jsonl(ROOT / cfg["paths"]["processed"] / rec / f"{lang}.jsonl", ex)
            print(f"{rec}/{lang}: {len(ex)} examples, known={sum(e['known'] for e in ex)}")
        r1 = trap_examples(cfg, lang, labels)
        if r1:
            write_jsonl(ROOT / cfg["paths"]["processed"] / "R1" / f"{lang}.jsonl", r1)
            print(f"R1/{lang}: {len(r1)} examples")


if __name__ == "__main__":
    main()
