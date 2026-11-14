"""CuratorKIT pass over the SFT training sets: clean + exact/MinHash dedup of data/processed/<R>/<lang>.jsonl.

Usage:
  python src/curate.py --recipes R1 R2 R3 R4 --langs en es ar ms
  python src/curate.py --recipes R1 --langs es --inspect     # prints what CuratorKIT returns, writes nothing

Input is converted to Alpaca rows {"instruction": <question>, "input": "", "output": <completion>} files under
data/curated_in/, run through curatorkit.Curator (no LLM calls), then the surviving rows are joined back to
the original records by id/question and written to data/curated/<R>/<lang>.jsonl plus a report line
(kept/dropped per stage) in outputs/results/curate_report.jsonl. Training can then use the curated sets.
Short answers ("1998") are legitimate, so min_tokens is 1 and the schema gate is off. clean=False: CuratorKIT's text
cleaning must not rewrite Arabic/Malay/Spanish text (we join survivors back to the originals by exact text).
"""
import argparse
import json

import curatorkit as ck

from common import ROOT, load_config, read_jsonl, write_jsonl


def question_of(prompt):
    return prompt.rsplit("Question:", 1)[-1].split("\nAnswer:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", default=["R1", "R2", "R3", "R4"])
    ap.add_argument("--langs", nargs="+", default=["en", "es", "ar", "ms"])
    ap.add_argument("--dedup", default="minhash", choices=["exact", "minhash"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    report = []
    for rec in args.recipes:
        for lang in args.langs:
            src = ROOT / cfg["paths"]["processed"] / rec / f"{lang}.jsonl"
            if not src.exists():
                continue
            rows = read_jsonl(src)
            inp = ROOT / "data" / "curated_in" / rec / f"{lang}.jsonl"
            write_jsonl(inp, [{"instruction": question_of(r["prompt"]), "input": "", "output": r["completion"]}
                              for r in rows])  # Alpaca schema: the only one CuratorKIT detects
            conf = ck.CuratorConfig(
                dataset=str(inp), format="auto", min_tokens=1, schema_gate=False, clean=False,
                dedup=args.dedup, minhash_threshold=0.9,
                output_dir=str(ROOT / "data" / "curated_out" / rec / lang), name=f"{rec}_{lang}",
                export_formats=["jsonl"], write_hf_readme=False,
            )
            res = ck.Curator(conf).run()
            passed = list(getattr(res, "passed", None) or [])
            rej = list(getattr(res, "rejected", None) or [])
            if args.inspect:
                from collections import Counter
                print(rec, lang, "in", len(rows), "passed", len(passed), "rejected", len(rej))
                print("rejection reasons:", Counter(getattr(r, "rejection_reason", "?") for r in rej).most_common(6))
                if passed:
                    print("passed[0]:", {k: getattr(passed[0], k, None) for k in ("instruction", "input", "output")})
                continue
            # join back by (question, completion): survivors keep their original prompt/known fields
            keep_keys = set()
            for s in passed:
                keep_keys.add((str(getattr(s, "instruction", "")).strip(), str(getattr(s, "output", "")).strip()))
            matched = sum((question_of(r["prompt"]), r["completion"].strip()) in keep_keys for r in rows)
            if keep_keys and matched < len(keep_keys):
                print(f"!! {rec}/{lang}: {len(keep_keys) - matched} survivors did not match an original row (text rewritten?)")
            if not keep_keys:  # nothing survived / unknown shape: do not silently drop everything
                print(f"!! {rec}/{lang}: no survivors read (reasons: see --inspect); keeping all")
                kept = rows
            else:
                kept = [r for r in rows if (question_of(r["prompt"]), r["completion"].strip()) in keep_keys]
            out = ROOT / "data" / "curated" / rec / f"{lang}.jsonl"
            write_jsonl(out, kept)
            line = {"recipe": rec, "lang": lang, "in": len(rows), "kept": len(kept),
                    "rejected_reported": len(rej)}
            report.append(line)
            print(line)

    if report:
        write_jsonl(ROOT / cfg["paths"]["outputs"] / "results" / "curate_report.jsonl", report)


if __name__ == "__main__":
    main()
