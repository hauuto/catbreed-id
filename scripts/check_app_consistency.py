"""Compare a locked deployment against notebook validation outputs on 20 Oxford images.

Usage: python scripts/check_app_consistency.py --package model_package --images PATH_TO_OXFORD_IMAGES
The package must also contain val.npz and validation_scores.csv from the same stage_dir.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.real_model import DeploymentClassifier, METHODS


def run(package: Path, images: Path, output: Path, precision="float32", count=20,
        device="cpu", method=None):
    if count < 0:
        raise ValueError("count phải không âm")
    model = DeploymentClassifier(
        package,
        exif_transpose=False,
        precision=precision,
        device=device,
        method=method,
    )
    print(f"Kiểm khớp {model.run_id}: torch={torch.__version__}, device={device}, precision={precision}", flush=True)
    with np.load(package / "val.npz", allow_pickle=False) as saved:
        image_ids = [str(x) for x in saved["image_id"]]
        logits = saved["logits"].copy()
        features = saved["features"].copy()
    with (package / "validation_scores.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["image_id"]: row for row in csv.DictReader(f)}
    candidates = [i for i, image_id in enumerate(image_ids) if image_id in rows and (images / f"{image_id}.jpg").is_file()]
    required = len(image_ids) if count == 0 else count
    if len(candidates) < required:
        raise ValueError(f"Chỉ có {len(candidates)} ảnh validation khớp ID và hiện diện; cần {required}")
    results = []
    for i in candidates[:required]:
        image_id = image_ids[i]
        with Image.open(images / f"{image_id}.jpg") as img:
            z, h = model.predict_raw(img)
        scores = model.scores(z, h)
        reference_scores = model.scores(logits[i], features[i])
        logit_abs = float(np.max(np.abs(z - logits[i])))
        result = {"image_id": image_id, "precision": precision, "device": device,
                  "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
                  "checkpoint_sha256": model.checkpoint_sha256, "logits_max_abs": logit_abs}
        passed = logit_abs < 1e-4
        for method in METHODS:
            expected = float(rows[image_id][method])
            actual = scores[method]
            error = abs(actual - expected)
            result[f"{method}_abs"] = error
            result[f"{method}_expected"] = expected
            result[f"{method}_actual"] = actual
            result[f"{method}_relative"] = error / max(abs(expected), 1e-12)
            result[f"{method}_saved_features_abs"] = abs(reference_scores[method] - expected)
            method_pass = bool(error < 1e-4 or (method == "Mahalanobis" and
                                               error / max(abs(expected), 1e-12) < 1e-6))
            result[f"{method}_pass"] = method_pass
            passed &= method_pass
        result["pass"] = passed
        results.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    summary = {"run_id": model.run_id, "checkpoint_sha256": model.checkpoint_sha256,
               "device": device, "precision": precision, "torch": torch.__version__,
               "torchvision": torchvision.__version__, "exif_transpose": False,
               "n": len(results), "passed": sum(bool(r["pass"]) for r in results),
               "criteria": "logits and scores abs < 1e-4; Mahalanobis also permits relative < 1e-6 (OR, not added)",
               "max_logits_abs": max(r["logits_max_abs"] for r in results),
               "max_scores_abs": {m: max(r[f"{m}_abs"] for r in results) for m in METHODS},
               "failures": [{"image_id": r["image_id"], "methods": [m for m in METHODS if not r[f"{m}_pass"]]}
                            for r in results if not r["pass"]]}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{sum(row['pass'] for row in results)}/{required} đạt ({precision}); chi tiết: {output}")
    if not all(row["pass"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path("model_package"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("app_consistency.csv"))
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--count", type=int, default=20, help="0 = toàn bộ validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--method", choices=METHODS, default=None,
                        help="Phương pháp cho gói không có locked_selection.json")
    args = parser.parse_args()
    run(args.package, args.images, args.output, args.precision, args.count, args.device, args.method)
