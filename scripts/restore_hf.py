"""Restore outputs after a Kaggle reset: morning backup tgz + everything synced live since.
  python scripts/restore_hf.py --user EmanMirza2025
Old (pre-EOS-fix) per-adapter evals from the tgz are moved to outputs/results_old/ so they are re-evaluated;
fresh evals from live/ are then copied over.
"""
import argparse, glob, os, shutil, tarfile

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--repo", default="tinyaya-idk-outputs")
ap.add_argument("--tgz", default="final_backup_nov14.tgz")
a = ap.parse_args()
rid = f"{a.user}/{a.repo}"

if not os.path.isdir(f"{ROOT}/outputs/adapters"):
    p = hf_hub_download(rid, a.tgz, repo_type="dataset", local_dir="/tmp/restore")
    with tarfile.open(p) as t:
        t.extractall("/tmp/restore/x")
    for d in ("outputs", "data"):
        shutil.copytree(f"/tmp/restore/x/{d}", f"{ROOT}/{d}", dirs_exist_ok=True)
    os.makedirs(f"{ROOT}/outputs/results_old", exist_ok=True)
    for f in glob.glob(f"{ROOT}/outputs/results/eval_*_s[0-9].csv"):
        shutil.move(f, f"{ROOT}/outputs/results_old/")
    shutil.rmtree("/tmp/restore", ignore_errors=True)
    print("restored base backup")

snap = snapshot_download(rid, repo_type="dataset", allow_patterns="live/*", local_dir="/tmp/live")
live = f"{snap}/live"
if os.path.isdir(live):
    for sub in ("adapters", "results"):
        if os.path.isdir(f"{live}/{sub}"):
            shutil.copytree(f"{live}/{sub}", f"{ROOT}/outputs/{sub}", dirs_exist_ok=True)
print("adapters:", len(os.listdir(f"{ROOT}/outputs/adapters")),
      "evals:", len(glob.glob(f"{ROOT}/outputs/results/eval_*.csv")))
