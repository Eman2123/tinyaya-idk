"""Push adapters + data to the Hugging Face Hub.
Example: python scripts/hf_push.py --user EmanMirza2025 --seeds 1
Add --public only after the licence check. Needs a token with write access
(HF_WRITE_TOKEN, else HF_TOKEN).
"""
import argparse, os
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--adapters-repo", default="tinyaya-idk-adapters")
ap.add_argument("--dataset-repo", default="tinyaya-idk-outputs")
ap.add_argument("--seeds", nargs="+", default=["1"])
ap.add_argument("--public", action="store_true")
a = ap.parse_args()

api = HfApi(token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))
mrepo, drepo = f"{a.user}/{a.adapters_repo}", f"{a.user}/{a.dataset_repo}"
api.create_repo(mrepo, repo_type="model", private=not a.public, exist_ok=True)
api.create_repo(drepo, repo_type="dataset", private=not a.public, exist_ok=True)

api.upload_folder(folder_path=f"{ROOT}/outputs/adapters", repo_id=mrepo, repo_type="model",
                  allow_patterns=[f"*_s{s}/*" for s in a.seeds])
for sub in ["data/processed", "data/raw", "outputs/labels", "outputs/results", "outputs/samples"]:
    if os.path.isdir(f"{ROOT}/{sub}"):
        api.upload_folder(folder_path=f"{ROOT}/{sub}", path_in_repo=sub, repo_id=drepo, repo_type="dataset")
print("done", mrepo, drepo)
