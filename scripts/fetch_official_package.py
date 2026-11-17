"""Fetch exactly the locked stage files from the CatBreedID Hugging Face dataset.

Run after `hf auth login` with an account that can read the private repository.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

from huggingface_hub import hf_hub_download

REPO = "catbreed-id/catbreed-experiments"
STUDY = "runs/20261116_052028_study_99d77653/locked_selection.json"
FILES = ("checkpoint.pt", "thresholds.json", "calibration.json",
         "distance_parameters.npz", "val.npz", "validation_scores.csv")


def fetch(destination: Path, selection_path: str = STUDY):
    destination.mkdir(parents=True, exist_ok=True)
    selection_source = Path(hf_hub_download(REPO, selection_path, repo_type="dataset"))
    selection = json.loads(selection_source.read_text(encoding="utf-8"))
    stage = PurePosixPath(selection["deployment"]["stage_dir"])
    if stage.is_absolute() or ".." in stage.parts:
        raise ValueError("stage_dir không hợp lệ")
    # The notebook's study.root corresponds to the repository's runs/ folder.
    prefix = stage if stage.parts[0] == "runs" else PurePosixPath("runs") / stage
    sources = {name: Path(hf_hub_download(REPO, str(prefix / name), repo_type="dataset")) for name in FILES}
    digest = hashlib.sha256(sources["checkpoint.pt"].read_bytes()).hexdigest()
    if digest.lower() != selection["deployment"]["checkpoint_sha256"].lower():
        raise ValueError("Checkpoint tải về không khớp SHA-256 đã khóa")
    for name, source in sources.items():
        shutil.copyfile(source, destination / name)
    shutil.copyfile(selection_source, destination / "locked_selection.json")
    print(f"Đã tải {prefix} vào {destination}; checkpoint SHA-256: {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("model_package"))
    parser.add_argument("--selection", default=STUDY)
    args = parser.parse_args()
    fetch(args.destination, args.selection)
