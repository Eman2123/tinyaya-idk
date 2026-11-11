"""CPU-only tests on synthetic data (no model needed). Run: python -m pytest tests -q  (or python tests/test_pipeline.py)"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from common import is_match, normalize, read_jsonl, write_jsonl  # noqa: E402
from stats import paired_bootstrap  # noqa: E402


def test_matching():
    assert normalize("The Eiffel Tower!") == "eiffel tower"
    assert normalize("القاهرة.") == "القاهرة"
    assert is_match("Paris, France", ["paris"])
    assert not is_match("Lyon", ["paris"])
    assert not is_match("", ["paris"])


def test_bootstrap():
    a = np.ones(200); b = np.zeros(200)
    d, (lo, hi) = paired_bootstrap(a, b)
    assert d == 1.0 and lo > 0.9
    d, (lo, hi) = paired_bootstrap(a, a)
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_labels_probe_datasets():
    tmp = Path(tempfile.mkdtemp())
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    cfg["paths"] = {"raw": str(tmp / "raw"), "processed": str(tmp / "processed"), "outputs": str(tmp / "out")}
    cfg_path = tmp / "cfg.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), allow_unicode=True)

    rng = np.random.default_rng(0)
    for lang in ["en", "ar"]:
        rows, H = [], []
        for i in range(400):
            known = i % 2 == 0
            gold = [f"ans{i}"]
            rows.append(dict(id=f"q{i}", split="build" if i < 250 else ("val" if i < 325 else "test"),
                             question=f"q{i}?", answers=gold, greedy=gold[0] if known else "wrong",
                             samples=gold * 5 if known else list("abcde")))
            H.append(rng.normal(size=16) + (1.5 if known else 0))
        write_jsonl(tmp / "out" / "samples" / f"{lang}.jsonl", rows)
        (tmp / "out" / "hidden").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp / "out" / "hidden" / f"{lang}.npz", ids=np.array([r["id"] for r in rows]),
                            **{"layer-12": np.array(H), "layer-1": np.array(H)})

    def run(*args):
        subprocess.run([sys.executable, str(SRC / args[0]), *args[1:], "--config", str(cfg_path)],
                       check=True, cwd=ROOT)

    for lang in ["en", "ar"]:
        run("labels.py", "--lang", lang)
    run("probe.py", "--anchor", "en", "--targets", "ar")
    run("build_datasets.py", "--langs", "en", "ar")

    lab = read_jsonl(tmp / "out" / "labels" / "ar.jsonl")
    assert all(k in lab[0] for k in ["gold_known", "cons_known", "probe_known"])
    for rec in ["R2", "R3", "R4"]:
        ex = read_jsonl(tmp / "processed" / rec / "ar.jsonl")
        assert len(ex) > 100
        assert any(not e["known"] and e["completion"] == cfg["refusals"]["ar"] for e in ex)
        assert 0.4 < np.mean([e["known"] for e in ex]) < 0.6


if __name__ == "__main__":
    test_matching(); test_bootstrap(); test_labels_probe_datasets()
    print("all tests passed")
