"""CuratorKIT pass over the SFT training sets: clean + exact/MinHash dedup of data/processed/<R>/<lang>.jsonl.

Usage:
  python src/curate.py --recipes R1 R2 R3 R4 --langs en es ar ms
  python src/curate.py --recipes R1 --langs es --inspect     # prints what CuratorKIT returns, writes nothing

Input is converted to {"instruction": <question>, "response": <completion>, "id", "known"} files under
data/curated_in/, run through curatorkit.Curator (no LLM calls), then the surviving rows are joined back to
the original records by id/question and written to data/curated/<R>/<lang>.jsonl plus a report line
(kept/dropped per stage) in outputs/results/curate_report.jsonl. Training can then use the curated sets.
Short answers ("1998") are legitimate, so min_tokens is 1 and the schema gate is off.
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
            write_jsonl(inp, [{"instruction": question_of(r["prompt"]), "response": r["completion"],
                               "idx": i} for i, r in enumerate(rows)])
            conf = ck.CuratorConfig(
                dataset=str(inp), format="jsonl", min_tokens=1, schema_gate=False, clean=True,
                dedup=args.dedup, minhash_threshold=0.9,
                output_dir=str(ROOT / "data" / "curated_out" / rec / lang), name=f"{rec}_{lang}",
                export_formats=["jsonl"], write_hf_readme=False,
            )
            res = ck.Curator(conf).run()
            if args.inspect:
                print(rec, lang, type(res), [a for a in dir(res) if not a.startswith("_")])
                print(res)
                continue
            kept_items = getattr(res, "samples", None) or getattr(res, "accepted", None) or []
            rej = getattr(res, "rejected", None) or []
            # join back by (question, completion): survivors keep their original prompt/known fields
            keep_keys = set()
            for s in kept_items:
                d = s if isinstance(s, dict) else getattr(s, "__dict__", {})
                data = d.get("data", d)
                keep_keys.add((str(data.get("instruction", "")).strip(), str(data.get("response", "")).strip()))
            kept = [r for r in rows if (question_of(r["prompt"]), r["completion"].strip()) in keep_keys]
            if not keep_keys:  # unknown result shape: do not silently drop everything
                print(f"!! {rec}/{lang}: could not read survivors from result, run --inspect; keeping all")
                kept = rows
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
