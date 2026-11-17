"""Tier 1: app model/preprocessing/scoring in the notebook batch/TF32 mode."""
import argparse
import csv
import hashlib
import json
import platform
import socket
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.real_model import DeploymentClassifier, METHODS


def run(package, images, output):
    torch.set_num_threads(1)
    model = DeploymentClassifier(package, device="cuda", exif_transpose=False)
    batch_size = int(model.package.saved["config"]["batch_size"])
    with np.load(package / "val.npz", allow_pickle=False) as data:
        ids = data["image_id"].astype(str).tolist()
        logits = data["logits"].copy()
        features = data["features"].copy()
    with (package / "validation_scores.csv").open(encoding="utf-8", newline="") as handle:
        expected = {r["image_id"]: r for r in csv.DictReader(handle)}
    # Reproduce the notebook's full batches, including its final partial batch.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for start in range(0, len(ids), batch_size):
        tensors = []
        for image_id in ids[start:start + batch_size]:
            with Image.open(images / (image_id + ".jpg")) as image:
                tensors.append(model.preprocess(image))
        x = torch.cat(tensors)
        with torch.no_grad():
            h = model.model.forward_features(x)
            z = model.model.head(h)
        h, z = h.cpu().numpy(), z.cpu().numpy()
        for offset, image_id in enumerate(ids[start:start + batch_size]):
            i = start + offset
            scores = model.scores(z[offset], h[offset])
            row = {"image_id": image_id, "batch_size": len(tensors),
                   "logits_max_abs": float(np.max(np.abs(z[offset] - logits[i]))),
                   "features_max_abs": float(np.max(np.abs(h[offset] - features[i])))}
            passed = row["logits_max_abs"] < 1e-4 and row["features_max_abs"] < 1e-4
            for method in METHODS:
                ref = float(expected[image_id][method])
                error = abs(scores[method] - ref)
                ok = error < 1e-4 or (method == "Mahalanobis" and error / max(abs(ref), 1e-12) < 1e-6)
                row[method + "_abs"], row[method + "_pass"] = error, bool(ok)
                passed = passed and ok
            row["pass"] = bool(passed)
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"purpose": "tier1_code_parity_gate",
               "tier1_passed": all(r["pass"] for r in rows), "run_id": model.run_id,
               "checkpoint_sha256": model.checkpoint_sha256,
               "hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "torchvision": torchvision.__version__,
               "python": platform.python_version(), "cudnn_tf32": True, "matmul_tf32": False,
               "batch_size": batch_size, "last_batch_size": rows[-1]["batch_size"],
               "n": len(rows), "passed": sum(r["pass"] for r in rows),
               "max_logits_abs": max(r["logits_max_abs"] for r in rows),
               "max_features_abs": max(r["features_max_abs"] for r in rows),
               "max_scores_abs": {m: max(r[m + "_abs"] for r in rows) for m in METHODS},
               "reference_sha256": {n: hashlib.sha256((package / n).read_bytes()).hexdigest()
                                    for n in ["checkpoint.pt", "val.npz", "validation_scores.csv"]},
               "failures": [r["image_id"] for r in rows if not r["pass"]]}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.package, args.images, args.output))
