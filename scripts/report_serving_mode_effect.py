"""Tier 2: report serving-mode effects, without a numerical pass/fail threshold."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(ids, expected_logits, actual_logits, expected_knn, actual_knn, tau):
    expected_logits, actual_logits = np.asarray(expected_logits), np.asarray(actual_logits)
    expected_knn, actual_knn = np.asarray(expected_knn), np.asarray(actual_knn)
    if expected_logits.shape != actual_logits.shape or expected_logits.ndim != 2:
        raise ValueError("Logit matrices must match")
    n = len(ids)
    if not n or expected_logits.shape[0] != n or expected_knn.shape != (n,) or actual_knn.shape != (n,):
        raise ValueError("IDs and predictions must have matching nonempty lengths")
    if not all(np.isfinite(x).all() for x in [expected_logits, actual_logits, expected_knn, actual_knn]):
        raise ValueError("Predictions must be finite")
    errors = np.abs(actual_logits.astype(np.float64) - expected_logits.astype(np.float64))
    knn_errors = np.abs(actual_knn - expected_knn)
    label_changed = expected_logits.argmax(1) != actual_logits.argmax(1)
    decision_changed = (expected_knn >= tau) != (actual_knn >= tau)
    ref_margin, app_margin = np.abs(expected_knn - tau), np.abs(actual_knn - tau)
    return {
        "purpose": "tier2_serving_mode_effect_report_only", "n": n, "tau": float(tau),
        "logits_abs_max": float(errors.max()),
        "logits_abs_median": float(np.median(errors)),
        "logits_per_image_max_abs_median": float(np.median(errors.max(axis=1))),
        "logits_median_definition": "median over all 238x12 absolute logit differences; also report median of per-image maxima",
        "knn_abs_max": float(knn_errors.max()), "knn_abs_median": float(np.median(knn_errors)),
        "top1_changed": int(label_changed.sum()),
        "top1_changed_ids": [ids[i] for i in np.flatnonzero(label_changed)],
        "knn_decision_changed": int(decision_changed.sum()),
        "knn_decision_changed_ids": [ids[i] for i in np.flatnonzero(decision_changed)],
        "min_notebook_knn_margin": float(ref_margin.min()),
        "min_notebook_knn_margin_id": ids[int(ref_margin.argmin())],
        "min_app_knn_margin": float(app_margin.min()),
        "min_app_knn_margin_id": ids[int(app_margin.argmin())],
        "images_with_error_ge_notebook_margin": int((knn_errors >= ref_margin).sum()),
    }


def run(package, images, output):
    import torch
    import torchvision
    from PIL import Image
    from app.api.real_model import DeploymentClassifier
    torch.set_num_threads(1)
    model = DeploymentClassifier(package, device="cuda", exif_transpose=True)
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    with np.load(package / "val.npz", allow_pickle=False) as data:
        ids = data["image_id"].astype(str).tolist()
        expected_logits = data["logits"].copy()
    with (package / "validation_scores.csv").open(encoding="utf-8", newline="") as handle:
        expected = {r["image_id"]: r for r in csv.DictReader(handle)}
    tau = model.thresholds["KNN"]
    actual_logits, actual_knn, rows = [], [], []
    expected_knn = np.array([float(expected[i]["KNN"]) for i in ids])
    for i, image_id in enumerate(ids):
        with Image.open(images / (image_id + ".jpg")) as image:
            z, h = model.predict_raw(image)
        knn = model.scores(z, h)["KNN"]
        actual_logits.append(z)
        actual_knn.append(knn)
        row = {"image_id": image_id,
               "logits_max_abs": float(np.max(np.abs(z - expected_logits[i]))),
               "knn_expected": expected_knn[i], "knn_actual": knn,
               "knn_abs": abs(knn - expected_knn[i]),
               "top1_notebook": int(expected_logits[i].argmax()), "top1_app": int(z.argmax()),
               "accepted_notebook": bool(expected_knn[i] >= tau), "accepted_app": bool(knn >= tau),
               "notebook_margin": abs(expected_knn[i] - tau), "app_margin": abs(knn - tau)}
        rows.append(row)
    summary = summarize(ids, expected_logits, np.array(actual_logits), expected_knn, np.array(actual_knn), tau)
    summary.update(run_id=model.run_id, checkpoint_sha256=model.checkpoint_sha256,
                   temperature=model.temperature, device="cuda", batch_size=1,
                   cudnn_tf32=False, matmul_tf32=False, precision=model.precision_name,
                   exif_transpose=True, hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__, torchvision=torchvision.__version__, python=platform.python_version(),
                   reference_sha256={n: hashlib.sha256((package / n).read_bytes()).hexdigest()
                                     for n in ["checkpoint.pt", "val.npz", "validation_scores.csv"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output.with_suffix(".npz"), image_id=np.array(ids), logits=np.array(actual_logits), knn=np.array(actual_knn))
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.package, args.images, args.output)
