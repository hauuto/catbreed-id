"""Prepare traceable validation examples without selecting from test data.

Results remain technical evidence until the consistency gate passes.
DTD/FeliBreed must be supplied with their provenance; no substitutes are synthesized.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.real_model import DeploymentClassifier


def run(package, images, output, device="cpu"):
    torch.set_num_threads(1)
    model = DeploymentClassifier(package, device=device)
    with np.load(package / "val.npz", allow_pickle=False) as a:
        ids, labels, logits = a["image_id"].astype(str), a["labels"], a["logits"]
        with (package / "validation_scores.csv").open(encoding="utf-8", newline="") as f:
            scores = {r["image_id"]: r for r in csv.DictReader(f)}
        accepted = np.array([float(scores[i][model.method]) >= model.thresholds[model.method] for i in ids])
        correct = logits.argmax(1) == labels
        candidates = []
        for kind, mask in (("id_correct", correct & accepted), ("id_error", ~correct)):
            indices = np.flatnonzero(mask)
            if len(indices):
                i = int(indices[0])
                candidates.append((kind, images / f"{ids[i]}.jpg", model.classes[int(labels[i])], "Oxford ID validation"))
    manifest = Path("data_ood/manifest_local.csv")
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            dog = next((r for r in csv.DictReader(f) if r["group"] == "near_val"), None)
        if dog:
            candidates.append(("dog", Path(dog["file"]), dog["label"], dog["source"]))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, path, truth, source in candidates:
        with Image.open(path) as img:
            img.load()
            result = model.predict(img)
        target = output / f"{kind}{path.suffix.lower()}"
        shutil.copyfile(path, target)
        # Keep the entire API result, including the real Grad-CAM, for later reproducibility.
        (output / f"{kind}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append({"kind": kind, "source_path": str(path.resolve()), "source": source,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "truth": truth,
                        "image": str(target), "status": result["status"], "top_label": result["top_label"],
                        "knn_score": result["ood"]["score"]})
    report = {"chapter5_ready": False, "selection_rule": "First matching validation row; dog is first near_val manifest row",
              "checkpoint_sha256": model.checkpoint_sha256, "cases": records,
              "missing": ["DTD with provenance", "FeliBreed with provenance", "Browser screenshots after consistency gate"]}
    (output / "cases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path("model_package"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/official_v2/cases"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.package, args.images, args.output, args.device)
