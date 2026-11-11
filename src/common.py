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
    text = "".join(ch for ch in text if ch not in set(string.punctuation) and not unicodedata.category(ch).startswith("P"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_match(pred, golds):
    """Normalized exact match, or gold contained as a whole phrase in the prediction."""
    p = normalize(pred)
    if not p:
        return False
    for g in golds:
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


def build_prompt(cfg, question):
    return cfg["prompt_template"].format(question=question)


def is_refusal(text, cfg, lang):
    """True if the generated text starts with the language's refusal template."""
    ref = normalize(cfg["refusals"][lang])
    return normalize(text).startswith(ref) if ref else False
