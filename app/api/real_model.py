"""Inference adapter for the shared CatBreedID inference core."""

import base64
import io
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TVF

from src.catbreed_core import (
    CatClassifier,
    METHODS,
    decide,
    eval_transform,
    load_package,
    predict_arrays,
    read_rgb_image,
)
from .temp_model import BaseCatClassifier


class NotebookClassifier(CatClassifier):
    """Compatibility constructor used by architecture parity tests."""

    def __init__(self, name, dropout, hidden, batchnorm):
        super().__init__(
            name,
            pretrained=False,
            dropout=dropout,
            head_hidden=hidden,
            baseline_batchnorm=batchnorm,
        )


class DeploymentClassifier(BaseCatClassifier):
    is_mock = False

    def __init__(
        self,
        package_dir: Path,
        *,
        exif_transpose: bool = True,
        precision: str = "float32",
        device: str = "cpu",
    ):
        if precision not in ("float32", "float64"):
            raise ValueError("precision phải là float32 hoặc float64")
        if device not in ("cpu", "cuda"):
            raise ValueError("device phải là cpu hoặc cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA không khả dụng trong môi trường Python hiện tại")

        self.precision_name = precision
        self.dtype = getattr(torch, precision)
        self.device = torch.device(device)
        self.exif_transpose = exif_transpose
        if self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.package_dir = Path(package_dir)
        self.package = load_package(self.package_dir, device=device)
        if self.package.is_smoke_test:
            raise ValueError("Checkpoint smoke test không được phục vụ qua app")
        self.package.model.to(device=self.device, dtype=self.dtype)
        self.package.model.eval()

        saved = self.package.saved
        config = saved["config"]
        if config.get("image_size") not in (None, 224) or config.get("resize_size") not in (None, 256):
            raise ValueError("Kích thước tiền xử lý checkpoint chưa được hỗ trợ")
        mapping = saved["class_mapping"]
        if len(mapping) != 12 or set(mapping.values()) != set(range(12)):
            raise ValueError("class_mapping phải gồm đúng 12 chỉ số từ 0 đến 11")

        self.classes = [name.replace("_", " ") for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
        self.name = saved["model_name"]
        self.stage = saved["stage"]
        self.run_id = saved["run_id"]
        deployment = self.package.locked_selection.get("deployment", {})
        rejection = self.package.locked_selection.get("rejection", {})
        if deployment:
            if (
                deployment.get("model") != self.name
                or deployment.get("stage") != self.stage
                or Path(deployment.get("stage_dir", "")).name != self.stage
                or self.run_id != Path(deployment.get("stage_dir", "")).parts[-2]
            ):
                raise ValueError("Model, stage hoặc run_id không khớp locked_selection.json")

        seed_match = re.search(r"_s(\d+)(?:_|$)", self.run_id)
        self.seed = int(seed_match.group(1)) if seed_match else None
        self.checkpoint_sha256 = self.package.checkpoint_sha256
        self.thresholds = {
            method: float(self.package.thresholds["T1"][method]) for method in METHODS
        }
        self.temperature = float(self.package.calibration["temperature"])
        self.method = rejection.get("method")
        if self.method not in METHODS:
            raise ValueError("Phương pháp từ chối không hợp lệ trong locked_selection.json")
        if not np.isclose(
            self.thresholds[self.method],
            float(rejection["threshold"]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Phương pháp hoặc ngưỡng không khớp locked_selection.json")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Nhiệt độ hiệu chỉnh không hợp lệ")
        if rejection and not np.isclose(
            self.temperature,
            float(rejection["temperature"]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("Nhiệt độ không khớp locked_selection.json")
        super().__init__(classes=self.classes)

    @property
    def model(self):
        return self.package.model

    def input_image(self, img: Image.Image) -> Image.Image:
        if self.exif_transpose:
            img = ImageOps.exif_transpose(img)
        return read_rgb_image(img)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        return eval_transform()(self.input_image(img)).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )

    def model_input_image(self, img: Image.Image) -> Image.Image:
        tensor = self.preprocess(img)[0].detach().cpu().float()
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return TVF.to_pil_image((tensor * std + mean).clamp(0, 1))

    def predict_raw(self, img: Image.Image):
        logits, features, _ = predict_arrays(
            self.package, img, exif=self.exif_transpose
        )
        return logits[0], features[0]

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _gradcam(self, feature_map, logits):
        with torch.enable_grad():
            predicted = int(logits.argmax(1).item())
            self._sync()
            gradient = torch.autograd.grad(logits[0, predicted], feature_map)[0]
            cam = F.relu(
                (gradient.mean((2, 3), keepdim=True) * feature_map).sum(1, keepdim=True)
            )
            cam = F.interpolate(
                cam, size=(224, 224), mode="bilinear", align_corners=False
            )[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-12)
            return cam.detach().cpu().numpy()

    def scores(self, logits, features):
        values = {**self.package.distance.score(np.asarray(features)[None, :])}
        from src.catbreed_core.scores import logit_scores

        values.update(logit_scores(np.asarray(logits)[None, :]))
        return {method: float(values[method][0]) for method in METHODS}

    def predict(self, img: Image.Image):
        self._sync()
        total_started = time.perf_counter()
        (
            logits,
            features,
            scores,
            feature_map,
            logits_tensor,
            core_timings,
        ) = predict_arrays(
            self.package,
            img,
            exif=self.exif_transpose,
            return_graph=True,
        )
        self._sync()
        started = time.perf_counter()
        intensity = self._gradcam(feature_map, logits_tensor)
        self._sync()
        explained = time.perf_counter()

        # Use the shared module's raw output and scores; Grad-CAM is only explanatory.
        decision = decide(self.package, scores, logits)
        ood_all = [
            {
                "method": method,
                "score": float(np.asarray(scores[method]).reshape(-1)[0]),
                "threshold": self.thresholds[method],
                "accept": bool(
                    np.asarray(scores[method]).reshape(-1)[0] >= self.thresholds[method]
                ),
            }
            for method in METHODS
        ]
        ood = next(item for item in ood_all if item["method"] == decision["method"])
        scored = time.perf_counter()
        probabilities = decision["probs"]
        top_indices = np.argsort(-probabilities)[:5]
        top5 = [
            {"label": self.classes[int(index)], "prob": float(probabilities[index])}
            for index in top_indices
        ]

        rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = np.clip(255 * (1 - intensity), 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(intensity * 220, 0, 220).astype(np.uint8)
        heatmap, model_input = io.BytesIO(), io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(heatmap, format="PNG")
        self.model_input_image(img).save(model_input, format="PNG")
        heatmap_b64 = base64.b64encode(heatmap.getvalue()).decode("ascii")
        input_b64 = base64.b64encode(model_input.getvalue()).decode("ascii")
        finished = time.perf_counter()

        return {
            "api_version": "1.0",
            "is_mock": False,
            "status": "accepted" if decision["accepted"] else "rejected",
            "top_label": decision["top_class"] if decision["accepted"] else None,
            "top5": top5,
            "ood": ood,
            "ood_all": ood_all,
            "model": {
                "backbone": self.name,
                "mode": self.stage,
                "seed": self.seed,
                "checkpoint_id": self.run_id,
                "model_name": self.name,
                "stage": self.stage,
                "run_id": self.run_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "rejection_method": decision["method"],
                "threshold": decision["tau"],
                "temperature": self.temperature,
                "precision": self.precision_name,
                "device": str(self.device),
                "exif_transpose": self.exif_transpose,
            },
            "timings_ms": {
                "preprocess": core_timings["preprocess"],
                "inference": core_timings["inference"],
                "gradcam": (explained - started) * 1000,
                "rejection": core_timings["rejection"] + (scored - explained) * 1000,
                "postprocess": (finished - scored) * 1000,
                "total": (finished - total_started) * 1000,
            },
            "heatmap": heatmap_b64,
            "cam_input": input_b64,
        }
