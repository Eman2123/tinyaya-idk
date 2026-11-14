"""Resumable train + eval loop.
Example: python scripts/run_loop.py --seeds 2 3 --recipes R2 R3 R4 --eval-langs same+en
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = ["en", "es", "ar", "ms"]

def run(cmd):
    print("$", cmd, flush=True)
    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    return subprocess.run(cmd, shell=True, cwd=ROOT, env=env).returncode

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--recipes", nargs="+", default=["R2", "R3", "R4", "R1"])
ap.add_argument("--langs", nargs="+", default=ALL)
ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the subprocesses, e.g. 1")
ap.add_argument("--sync-user", default=None, help="HF username: sync each finished run to the HF dataset repo")
ap.add_argument("--eval-langs", default="all", help="all | same+en | 'en es'")
a = ap.parse_args()

for seed in a.seeds:
    for lang in a.langs:
        for rec in a.recipes:
            name = f"{rec}_{lang}_s{seed}"
            if os.path.exists(f"{ROOT}/outputs/results/eval_{name}.csv"):
                print(name, "SKIP", flush=True)
                continue
            t0 = time.time()
            if not os.path.isdir(f"{ROOT}/outputs/adapters/{name}"):
                if run(f"python src/train_sft.py --recipe {rec} --langs {lang} --seed {seed}"):
                    print("TRAIN FAILED", name, flush=True)
                    continue
            t1 = time.time()
            if a.eval_langs == "all":
                ev = ALL
            elif a.eval_langs == "same+en":
                ev = sorted({lang, "en"}, key=ALL.index)
            else:
                ev = a.eval_langs.split()
            if run(f"python src/evaluate.py --name {name} --adapter outputs/adapters/{name} --langs {' '.join(ev)}"):
                print("EVAL FAILED", name, flush=True)
                continue
            if a.sync_user:
                try:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from sync_hf import sync
                    sync(a.sync_user, [name])
                except Exception as e:  # never let a sync problem kill the loop
                    print("SYNC FAILED", name, repr(e)[:200], flush=True)
            print(f">>> {name}: train {(t1-t0)/60:.1f} min, eval {(time.time()-t1)/60:.1f} min", flush=True)
