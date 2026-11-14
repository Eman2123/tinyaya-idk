"""Live sync of finished runs to the Hugging Face dataset repo, so a Kaggle session reset loses minutes, not hours.
Used by run_loop.py (--sync-user) after every finished train+eval; can also be run by hand:
  python scripts/sync_hf.py --user EmanMirza2025 --names R2_en_s2 R3_en_s2
Uploads outputs/adapters/<name>/ and outputs/results/*<name>* to <user>/tinyaya-idk-outputs under live/.
"""
import argparse, glob, os

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync(user, names, repo="tinyaya-idk-outputs"):
    api = HfApi(token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))
    rid = f"{user}/{repo}"
    ops = []
    for name in names:
        ad = f"{ROOT}/outputs/adapters/{name}"
        for f in glob.glob(f"{ad}/**", recursive=True):
            if os.path.isfile(f):
                ops.append(CommitOperationAdd(f"live/adapters/{name}/{os.path.relpath(f, ad)}", f))
        for f in glob.glob(f"{ROOT}/outputs/results/*{name}*"):
            ops.append(CommitOperationAdd(f"live/results/{os.path.basename(f)}", f))
    if ops:
        api.create_commit(rid, ops, repo_type="dataset", commit_message=f"sync {' '.join(names)}"[:200])
    print("synced", names, len(ops), "files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    a = ap.parse_args()
    sync(a.user, a.names)
