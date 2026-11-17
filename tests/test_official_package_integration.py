"""Real official package checks; never loads the legacy weights.pth/export.npz."""
import base64
import csv
import io
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from app.api.real_model import DeploymentClassifier, METHODS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model_package"


@unittest.skipUnless((PACKAGE / "checkpoint.pt").is_file(), "Official package has not been provisioned")
class OfficialPackageIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.classifier = DeploymentClassifier(PACKAGE)

    def test_all_saved_validation_scores_match_notebook_csv(self):
        with np.load(PACKAGE / "val.npz", allow_pickle=False) as saved:
            with (PACKAGE / "validation_scores.csv").open(encoding="utf-8", newline="") as f:
                rows = {r["image_id"]: r for r in csv.DictReader(f)}
            for i, image_id in enumerate(saved["image_id"].astype(str)):
                actual = self.classifier.scores(saved["logits"][i], saved["features"][i])
                for method in METHODS:
                    with self.subTest(image_id=image_id, method=method):
                        self.assertAlmostEqual(actual[method], float(rows[image_id][method]), delta=1e-8)

    def test_api_real_package_accept_reject_and_gradcam(self):
        from fastapi.testclient import TestClient
        from app.api.main import app
        client = TestClient(app)
        self.assertFalse(client.get("/health").json()["is_mock"])
        for name, status, label in (("Abyssinian_1.jpg", "accepted", "Abyssinian"),
                                    ("american_bulldog_2.jpg", "rejected", None)):
            with (ROOT / "app/static/img" / name).open("rb") as f:
                response = client.post("/predict", files={"file": (name, f, "image/jpeg")})
            with self.subTest(image=name):
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], status)
                self.assertEqual(data["top_label"], label)
                self.assertEqual(len(data["top5"]), 5)
                self.assertEqual([x["method"] for x in data["ood_all"]], list(METHODS))
                self.assertEqual(data["model"]["checkpoint_sha256"], self.classifier.checkpoint_sha256)
                self.assertGreater(data["timings_ms"]["total"], 0)
                for field in ("heatmap", "cam_input"):
                    with Image.open(io.BytesIO(base64.b64decode(data[field], validate=True))) as image:
                        self.assertEqual(image.size, (224, 224))
