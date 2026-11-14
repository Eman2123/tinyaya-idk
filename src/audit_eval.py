"""AuditKIT evaluation of a model (base or LoRA adapter) on the held-out test questions.

Usage: python src/audit_eval.py --name R3_es_s1 --adapter outputs/adapters/R3_es_s1 --langs en es ar ms
Uses auditkit.evaluate(samples, model_fn, scorers=[...]) and writes
outputs/results/audit_<name>.json (mean of each scorer per language). The numbers match
evaluate.py; this is the Lexsi-tool pass (AuditKIT) over the same predictions.
Scorers: correct (answered and matches gold), refused (any-language 'I don't know'),
hallucinated (answered, wrong). AuditKIT requires an explicit direction per scorer.
"""
import argparse
import json

import auditkit as ak
from auditkit.api import FunctionScorer
from peft import PeftModel

try:
    from auditkit.api import Direction
except ImportError:
    from auditkit.comparison import Direction

from common import build_prompt, is_match, is_refusal, load_config, load_questions, out_path
from sample import generate, load_model

SCORE_NAMES = ("correct", "refused", "hallucinated")


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

        samples = [ak.Sample(input=build_prompt(cfg, r["question"], lang), target=r["answers"][0],
                             id=r["id"], task=f"mkqa_{lang}", metadata={"gold": r["answers"], "lang": lang})
                   for r in test]
        known_inputs = {s.input for s in samples}

        def model_fn(prompts):
            # AuditKIT must hand us the exact few-shot prompt, otherwise numbers would not match evaluate.py
            bad = [p for p in prompts if p not in known_inputs]
            if bad:
                raise RuntimeError(f"AuditKIT altered {len(bad)} prompts, e.g. {bad[0][-120:]!r}")
            out = []
            for i in range(0, len(prompts), bs):
                out += generate(tok, model, prompts[i : i + bs], max_new, sample=False)
            return out

        def correct(sample, output, _lang=lang):
            return float((not is_refusal(output, cfg, _lang)) and is_match(output, sample.metadata["gold"]))

        def refused(sample, output, _lang=lang):
            return float(is_refusal(output, cfg, _lang))

        def hallucinated(sample, output, _lang=lang):
            return float((not is_refusal(output, cfg, _lang)) and not is_match(output, sample.metadata["gold"]))

        scorers = [FunctionScorer(correct, Direction.MAXIMIZE),
                   FunctionScorer(refused, Direction.MINIMIZE),
                   FunctionScorer(hallucinated, Direction.MINIMIZE)]
        res = ak.evaluate(samples, model_fn, scorers=scorers, cache=False)

        n = len(res.predictions)
        summary[lang] = {name: sum(sc["value"] for pr in res.predictions
                                   for sc in pr.metadata["scores"] if sc["name"] == name) / n
                         for name in SCORE_NAMES}
        summary[lang]["n"] = n
        print(lang, summary[lang])

    p = out_path(cfg, "results", f"audit_{args.name}.json")
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
