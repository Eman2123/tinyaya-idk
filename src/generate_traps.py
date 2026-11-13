"""R1 shortcut: have a teacher model write 'trap' questions (unanswerable or about invented things).

Usage (AI/ML API, OpenAI-compatible; the default):
  pip install openai
  export AIML_API_KEY=...            # on Kaggle: put it in Secrets as AIML_API_KEY and copy it into os.environ
  python src/generate_traps.py --lang es --n 500
  python src/generate_traps.py --lang es --n 500 --model anthropic/claude-haiku-5.5   # model id exactly as listed by AI/ML API

Anthropic API directly instead:
  pip install anthropic ; export ANTHROPIC_API_KEY=...
  python src/generate_traps.py --lang es --provider anthropic --model claude-sonnet-5-5

Writes data/raw/trap_<lang>.jsonl ({"question": ...}). The teacher is allowed only as a teacher, not as the
core model; list it in the submission's toolkit/teacher statement.
"""
import argparse
import json
import os
import re
import time

from common import ROOT, load_config, write_jsonl

LANG_NAMES = {"en": "English", "es": "Spanish", "ar": "Arabic", "id": "Indonesian",
              "ms": "Malay", "tr": "Turkish", "sw": "Swahili"}

PROMPT = """Write {k} short factoid questions in {lang} that a language model cannot know the answer to,
because they are about invented people, places, books, films, products or events, or about private or
unknowable facts. Each must look like a normal trivia question (who/when/where/how many), written in
natural {lang}, and must not be answerable from general world knowledge. Vary the topics and the question
words. Output one JSON object per line: {{"question": "..."}} and nothing else (no numbering, no commentary)."""


def make_caller(args):
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()

        def call(prompt):
            msg = client.messages.create(model=args.model, max_tokens=4000, temperature=1.0,
                                         messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text
    else:
        from openai import OpenAI
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"set {args.api_key_env} in the environment first")
        client = OpenAI(api_key=key, base_url=args.base_url)

        def call(prompt):
            r = client.chat.completions.create(model=args.model, max_tokens=4000, temperature=1.0,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
    return call


def parse_questions(text):
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            q = json.loads(re.sub(r",\s*}$", "}", line))["question"].strip()
        except Exception:
            continue
        if q:
            yield q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--provider", choices=["aiml", "anthropic"], default="aiml")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5",
                    help="aiml: id exactly as in the AI/ML API model list; anthropic: e.g. claude-sonnet-5-5")
    ap.add_argument("--base-url", default="https://api.aimlapi.com/v1", dest="base_url")
    ap.add_argument("--api-key-env", default="AIML_API_KEY", dest="api_key_env")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    call = make_caller(args)

    seen, out, fails = set(), [], 0
    while len(out) < args.n and fails < 5:
        try:
            text = call(PROMPT.format(k=25, lang=LANG_NAMES.get(args.lang, args.lang)))
        except Exception as e:  # rate limit / transient error: back off and retry a few times
            fails += 1
            print("API error:", repr(e)[:200], f"(retry {fails}/5)")
            time.sleep(5 * fails)
            continue
        got = 0
        for q in parse_questions(text):
            if q not in seen:
                seen.add(q)
                out.append({"question": q})
                got += 1
        fails = fails + 1 if got == 0 else 0
        print(len(out), "/", args.n)
    if not out:
        raise SystemExit("no questions generated; check the key, model id and base url")
    write_jsonl(ROOT / cfg["paths"]["raw"] / f"trap_{args.lang}.jsonl", out[: args.n])
    print(f"wrote {min(len(out), args.n)} trap questions for {args.lang}")


if __name__ == "__main__":
    main()
