"""R1 shortcut: have a teacher model write 'trap' questions (unanswerable or about invented things).

Usage: python src/generate_traps.py --lang es --n 500
Needs: pip install anthropic ; export ANTHROPIC_API_KEY=...
Writes data/raw/trap_<lang>.jsonl ({"question": ...}). Teacher is allowed only as a teacher, not as the
core model; list it in the submission's toolkit/teacher statement.
"""
import argparse
import json
import re

from common import ROOT, load_config, write_jsonl

LANG_NAMES = {"en": "English", "es": "Spanish", "ar": "Arabic", "id": "Indonesian",
              "ms": "Malay", "tr": "Turkish", "sw": "Swahili"}

PROMPT = """Write {k} short factoid questions in {lang} that a language model cannot know the answer to,
because they are about invented people, places, books, products or events, or about private or
unknowable facts. Each must look like a normal trivia question (who/when/where/how many).
Vary the topics. Output one JSON object per line: {{"question": "..."}} and nothing else."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--judge-model", default="claude-sonnet-5-5", dest="teacher")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    cfg = load_config(args.config)

    seen, out = set(), []
    while len(out) < args.n:
        msg = client.messages.create(
            model=args.teacher, max_tokens=4000, temperature=1.0,
            messages=[{"role": "user", "content": PROMPT.format(k=25, lang=LANG_NAMES.get(args.lang, args.lang))}])
        for line in msg.content[0].text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                q = json.loads(re.sub(r",\s*}$", "}", line))["question"].strip()
            except Exception:
                continue
            if q and q not in seen:
                seen.add(q)
                out.append({"question": q})
        print(len(out), "/", args.n)
    write_jsonl(ROOT / cfg["paths"]["raw"] / f"trap_{args.lang}.jsonl", out[: args.n])


if __name__ == "__main__":
    main()
