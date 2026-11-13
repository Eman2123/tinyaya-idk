"""Shared helpers: config, IO, normalization, answer matching, splits."""
import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def out_path(cfg, *parts):
    p = ROOT / cfg["paths"]["outputs"]
    for part in parts:
        p = p / part
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def normalize(text):
    """Lowercase, strip accents-insensitive punctuation/articles/whitespace. Unicode-safe."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    # keep only the first line: models sometimes continue after the answer
    text = text.split("\n")[0]
    text = re.sub(r"(?<=\d)\.0+(?!\d)", "", text)  # "18.0" -> "18"
    text = "".join(ch for ch in text if ch not in set(string.punctuation) and not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ISO_DATE = re.compile(r"^\s*(\d{4})-\d{2}(-\d{2})?\s*$")
_NUM_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+\S.*$")


def gold_variants(golds):
    """Extra acceptable forms of the gold answers (year-granularity matching for dates,
    bare number for 'number + unit' golds). Documented as a limitation in the write-up."""
    out = list(golds)
    for g in golds:
        g = str(g)
        m = _ISO_DATE.match(g)
        if m:
            out.append(m.group(1))            # 2003-06-21 -> 2003
        m = _NUM_UNIT.match(g)
        if m:
            out.append(m.group(1))            # "12.0 oz" -> 12.0 (normalize strips .0)
    return out


def is_match(pred, golds):
    """Normalized exact match, or gold contained as a whole phrase in the prediction.
    Dates match at year granularity; 'number + unit' golds also match the bare number."""
    p = normalize(pred)
    if not p:
        return False
    for g in gold_variants(golds):
        g = normalize(g)
        if not g:
            continue
        if p == g or f" {g} " in f" {p} ":
            return True
    return False


def assign_split(qid, cfg):
    """Deterministic split by hash if the data has no explicit split."""
    s = cfg["splits"]
    total = s["build"] + s["val"] + s["test"]
    h = int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % total
    if h < s["build"]:
        return "build"
    if h < s["build"] + s["val"]:
        return "val"
    return "test"


def load_questions(cfg, lang):
    rows = read_jsonl(ROOT / cfg["paths"]["raw"] / f"{lang}.jsonl")
    for r in rows:
        r.setdefault("split", assign_split(r["id"], cfg))
    return rows


def build_prompt(cfg, question, lang=None):
    """Few-shot prefix (per language, short-answer style) + the question.
    Used by sampling, training data and evaluation, so all stages see the same prompt."""
    prefix = (cfg.get("few_shot") or {}).get(lang, "")
    return prefix + cfg["prompt_template"].format(question=question)


def is_refusal(text, cfg, lang, native_only=False):
    """True if the generated text starts with a refusal template.

    By default ANY language's template counts: a model that says "I don't know." to a Spanish question
    has abstained, whatever language it used, and this is what a cross-lingual transfer matrix must
    measure. native_only=True accepts only the question language's own template (language fidelity).
    """
    refs = [cfg["refusals"][lang]] if native_only else list(cfg["refusals"].values())
    t = normalize(text)
    return any(t.startswith(normalize(r)) for r in refs if normalize(r))
