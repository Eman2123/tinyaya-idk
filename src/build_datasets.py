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
            # R2 (gold): the label already means "greedy answer matched gold", so use that answer.
            # R3/R4 have no gold: use the model's own majority answer (this is where label noise enters).
            ans = r["greedy"] if known_key == "gold_known" else r["majority"]
            if ans.strip():
                known.append({"prompt": build_prompt(cfg, r["question"], lang),
                              "completion": ans.strip(), "known": True, "id": r["id"]})
        else:
            unknown.append({"prompt": build_prompt(cfg, r["question"], lang),
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
    ex = [{"prompt": build_prompt(cfg, t["question"], lang), "completion": refusal, "known": False}
          for t in traps]
    # answerable side: build questions the model answers correctly (matches gold), target = its own short
    # answer. Same targets as the known half of R2, so R1 vs R2 differs only in the refusal side
    # (teacher-invented traps vs real unknown questions). Raw gold strings (e.g. '2003-06-21') are not
    # used as targets: their format differs from the few-shot style and breaks year-level matching.
    ans = [{"prompt": build_prompt(cfg, r["question"], lang), "completion": r["greedy"].strip(),
            "known": True, "id": r["id"]}
           for r in labels if r["split"] == "build" and r["gold_known"] and r["greedy"].strip()][: len(ex)]
    out = ex + ans
    random.Random(0).shuffle(out)
    return out


def equalize(sets, seed=0):
    """Give every recipe the same number of known and unknown examples (k = the smallest class count over
    recipes), so recipe comparisons are not confounded by training-set size."""
    k = min(min(sum(e["known"] for e in ex), sum(not e["known"] for e in ex)) for ex in sets.values())
    rng = random.Random(seed)
    out = {}
    for rec, ex in sets.items():
        kn = [e for e in ex if e["known"]]
        un = [e for e in ex if not e["known"]]
        rng.shuffle(kn)
        rng.shuffle(un)
        mixed = kn[:k] + un[:k]
        rng.shuffle(mixed)
        out[rec] = mixed
    return out, k


def build_combined(cfg, langs):
    """Ablation R5: a question counts as known only if the probe AND the consistency check both say so.
    Same size and class balance as the existing R4 set (so R5 vs R4 differs only in the label rule)."""
    for lang in langs:
        labels = read_jsonl(ROOT / cfg["paths"]["outputs"] / "labels" / f"{lang}.jsonl")
        if "probe_known" not in labels[0]:
            print(f"[R5] probe_known missing for {lang}, run probe.py first")
            continue
        for r in labels:
            r["combined_known"] = bool(r["probe_known"]) and bool(r["cons_known"])
        ex = make_examples(cfg, lang, labels, "combined_known")
        kn = [e for e in ex if e["known"]]
        un = [e for e in ex if not e["known"]]
        ref = ROOT / cfg["paths"]["processed"] / "R4" / f"{lang}.jsonl"
        k = sum(e["known"] for e in read_jsonl(ref)) if ref.exists() else len(kn)
        k = min(k, len(kn), len(un))
        out = kn[:k] + un[:k]
        random.Random(0).shuffle(out)
        write_jsonl(ROOT / cfg["paths"]["processed"] / "R5" / f"{lang}.jsonl", out)
        print(f"R5/{lang}: {len(out)} examples ({k} known + {k} unknown); combined-known pool={len(kn)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--show", type=int, default=0, help="print N known + N unknown examples per recipe")
    ap.add_argument("--combined", action="store_true",
                    help="build ONLY R5 (known = probe_known AND cons_known), sized to match the existing R4 set; "
                         "does not touch R1-R4")
    ap.add_argument("--no_equalize", action="store_true", help="keep each recipe's natural size")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.combined:
        build_combined(cfg, args.langs)
        return
    recipes = {"R2": "gold_known", "R3": "cons_known", "R4": "probe_known"}
    for lang in args.langs:
        labels = read_jsonl(ROOT / cfg["paths"]["outputs"] / "labels" / f"{lang}.jsonl")
        sets = {}
        for rec, key in recipes.items():
            if key not in labels[0]:
                print(f"[{rec}] {key} missing for {lang}, run probe.py first")
                continue
            sets[rec] = make_examples(cfg, lang, labels, key)
        r1 = trap_examples(cfg, lang, labels)
        if r1:
            sets["R1"] = r1
        if not args.no_equalize and sets:
            natural = {r: len(e) for r, e in sets.items()}
            sets, k = equalize(sets)
            print(f"{lang}: equalized to {k} known + {k} unknown per recipe (natural sizes {natural})")
        for rec, ex in sets.items():
            write_jsonl(ROOT / cfg["paths"]["processed"] / rec / f"{lang}.jsonl", ex)
            print(f"{rec}/{lang}: {len(ex)} examples, known={sum(e['known'] for e in ex)}")
            if args.show:
                for flag in (True, False):
                    for e in [e for e in ex if e["known"] == flag][: args.show]:
                        q = e["prompt"].rsplit("Question:", 1)[-1].split("\nAnswer:")[0].strip()
                        print(f"   {'K' if flag else 'U'} | {q[:70]} -> {e['completion'][:40]}")


if __name__ == "__main__":
    main()
