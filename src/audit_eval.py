"""AuditKIT evaluation of a model (base or LoRA adapter) on the held-out test questions.

Usage: python src/audit_eval.py --name R3_es_s1 --adapter outputs/adapters/R3_es_s1 --langs en es ar ms
Uses auditkit.evaluate(dataset=[Sample...], model=<list[str]->list[str]>, scorers=[...]) and writes
outputs/results/audit_<name>.json (mean of each scorer per language). The numbers should match
evaluate.py; this is the Lexsi-tool pass (AuditKIT) over the same predictions.
Scorers: correct (answered and matches gold), refused (any-language 'I don't know'), hallucinated (answered, wrong).
"""
import argparse
import json

import auditkit as ak
from peft import PeftModel

from common import build_prompt, is_match, is_refusal, load_config, load_questions, out_path
from sample import generate, load_model


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

    summary = {}
    for lang in args.langs:
        test = [r for r in load_questions(cfg, lang) if r["split"] == "test"]
        golds = {r["id"]: r["answers"] for r in test}

        def model_fn(prompts, _lang=lang):
            # AuditKIT hands us the Sample inputs; they are already full few-shot prompts
            out = []
            for i in range(0, len(prompts), bs):
                out += generate(tok, model, prompts[i : i + bs], max_new, sample=False)
            return out

        samples = [ak.Sample(input=build_prompt(cfg, r["question"], lang), target=r["answers"][0],
                             id=r["id"], task=f"mkqa_{lang}", metadata={"gold": r["answers"], "lang": lang})
                   for r in test]
        gold_by_input = {s.input: s.metadata["gold"] for s in samples}

        def correct(sample, output):
            return float((not is_refusal(output, cfg, lang)) and is_match(output, sample.metadata["gold"]))

        def refused(sample, output):
            return float(is_refusal(output, cfg, lang))

        def hallucinated(sample, output):
            return float((not is_refusal(output, cfg, lang)) and not is_match(output, sample.metadata["gold"]))

        res = ak.evaluate(samples, model_fn, scorers=[correct, refused, hallucinated], cache=False)
        print(lang, res)
        try:
            summary[lang] = json.loads(json.dumps(getattr(res, "summary", None) or res.__dict__, default=str))
        except Exception:
            summary[lang] = {"repr": repr(res)}

    p = out_path(cfg, "results", f"audit_{args.name}.json")
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
