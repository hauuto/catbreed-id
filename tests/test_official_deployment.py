"""Contract checks using a fresh synthetic package, never the old model_package files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from app.api.real_model import DeploymentClassifier, NotebookClassifier


class OfficialDeploymentTest(unittest.TestCase):
    def test_unlocked_package_requires_and_accepts_explicit_method(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = NotebookClassifier("cnn_baseline", 0.2, 64, True)
            torch.save({"model_name": "cnn_baseline", "state_dict": model.state_dict(),
                        "class_mapping": {f"breed_{i}": i for i in range(12)},
                        "config": {"dropout": 0.2, "head_hidden": 64, "baseline_batchnorm": True,
                                   "image_size": 224, "resize_size": 256},
                        "stage": "scratch", "run_id": "20261116_cnn_baseline_s42_trial"}, root / "checkpoint.pt")
            sha = hashlib.sha256((root / "checkpoint.pt").read_bytes()).hexdigest()
            thresholds = dict.fromkeys(("MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN"), 1e6)
            (root / "thresholds.json").write_text(
                json.dumps({"checkpoint_sha256": sha, "T1": thresholds}), encoding="utf-8"
            )
            (root / "calibration.json").write_text(json.dumps({"temperature": 0.7}), encoding="utf-8")
            np.savez_compressed(root / "distance_parameters.npz", means=np.zeros((12, 128)),
                                precision=np.eye(128), bank=np.ones((10, 128)) / np.sqrt(128), k=10)
            with self.assertRaisesRegex(ValueError, "method"):
                DeploymentClassifier(root, exif_transpose=False)
            classifier = DeploymentClassifier(root, exif_transpose=False, method="KNN")
            self.assertEqual(classifier.method, "KNN")
            result = classifier.predict(Image.new("RGB", (256, 256), (128, 128, 128)))
            self.assertEqual(result["ood"]["method"], "KNN")
            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["top_label"])
            self.assertEqual(len(result["top5"]), 5)
            self.assertTrue(result["heatmap"])

    def test_load_scores_rejection_and_alpha(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = NotebookClassifier("cnn_baseline", 0.2, 64, True)
            torch.save({"model_name": "cnn_baseline", "state_dict": model.state_dict(),
                        "class_mapping": {f"breed_{i}": i for i in range(12)},
                        "config": {"dropout": 0.2, "head_hidden": 64, "baseline_batchnorm": True,
                                   "image_size": 224, "resize_size": 256},
                        "stage": "scratch", "run_id": "20261116_cnn_baseline_s42_example"}, root / "checkpoint.pt")
            sha = hashlib.sha256((root / "checkpoint.pt").read_bytes()).hexdigest()
            thresholds = dict.fromkeys(("MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN"), 1e6)
            (root / "thresholds.json").write_text(json.dumps({"T1": thresholds}), encoding="utf-8")
            (root / "calibration.json").write_text(json.dumps({"temperature": 0.7}), encoding="utf-8")
            (root / "locked_selection.json").write_text(json.dumps({"deployment": {
                "stage_dir": "20261116_cnn_baseline_s42_example/scratch", "checkpoint_sha256": sha,
                "model": "cnn_baseline", "stage": "scratch"},
                "rejection": {"method": "KNN", "threshold": 1e6, "temperature": 0.7}}), encoding="utf-8")
            np.savez_compressed(root / "distance_parameters.npz", means=np.zeros((12, 128)),
                                precision=np.eye(128), bank=np.ones((10, 128)) / np.sqrt(128), k=10)
            classifier = DeploymentClassifier(root, exif_transpose=False)
            locked_classifier = DeploymentClassifier(root, exif_transpose=False, method="MSP")
            self.assertEqual(locked_classifier.method, "KNN")
            image = Image.new("RGBA", (224, 224), (255, 0, 0, 0))
            self.assertEqual(classifier.input_image(image).getpixel((100, 100)), (128, 128, 128))
            logits, features = classifier.predict_raw(image)
            self.assertEqual(logits.shape, (12,))
            self.assertEqual(features.shape, (128,))
            result = classifier.predict(image)
            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["top_label"])
            self.assertEqual(result["ood"]["method"], "KNN")
            self.assertEqual(len(result["ood_all"]), 5)
            self.assertEqual(result["model"]["checkpoint_sha256"], sha)
            self.assertEqual(result["model"]["seed"], 42)
            self.assertEqual(result["model"]["precision"], "float32")
            self.assertEqual(set(result["timings_ms"]),
                             {"preprocess", "inference", "gradcam", "rejection", "postprocess", "total"})
            self.assertTrue(all(value >= 0 for value in result["timings_ms"].values()))
            classifier.thresholds["KNN"] = result["ood"]["score"]
            classifier.package.thresholds["T1"]["KNN"] = result["ood"]["score"]
            boundary = classifier.predict(image)
            self.assertEqual(boundary["status"], "accepted")
            self.assertIsNotNone(boundary["top_label"])
            double = DeploymentClassifier(root, precision="float64")
            self.assertEqual(double.preprocess(image).dtype, torch.float64)
            self.assertEqual(double.predict(image)["model"]["precision"], "float64")
            (root / "checkpoint.pt").write_bytes(b"wrong-checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                DeploymentClassifier(root)
            self.assertAlmostEqual(sum(x["prob"] for x in result["top5"]),
                                   sum(sorted(torch.softmax(torch.from_numpy(logits) / 0.7, 0).numpy(), reverse=True)[:5]), places=5)
