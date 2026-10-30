"""Inference and Grad-CAM for the project's ResNet50 checkpoint."""
import base64
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import functional as TVF, InterpolationMode

from .temp_model import BaseCatClassifier


class ResNet50Classifier(BaseCatClassifier):
    """Run the saved model without downloading pretrained weights."""

    is_mock = False

    def __init__(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("arch") != "resnet50":
            raise ValueError("Checkpoint không phải ResNet50")
        config = checkpoint["preprocess"]
        if (config.get("resize") not in {"direct_bilinear", "short_side_center_crop"}
                or config.get("img_size") != 224
                or (config.get("resize") == "short_side_center_crop" and
                    (config.get("short_side") != 256 or config.get("interpolation") != "bilinear"
                     or config.get("antialias") is not True))):
            raise ValueError("Cấu hình tiền xử lý checkpoint chưa được hỗ trợ")
        self.preprocess_config = config
        classes = checkpoint["classes"]
        if len(classes) != 12 or checkpoint["model_cfg"]["num_classes"] != len(classes):
            raise ValueError("Danh sách lớp checkpoint không hợp lệ")
        super().__init__(classes=[label.replace("_", " ") for label in classes])
        self.mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(config["std"], dtype=np.float32).reshape(1, 1, 3)
        self.seed = int(checkpoint["seed"])
        self.mode = checkpoint["stage"]
        self.checkpoint_id = checkpoint["checkpoint_id"]
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                      backbone.maxpool, backbone.layer1, backbone.layer2,
                                      backbone.layer3, backbone.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, len(classes))
        # Match the names saved by training.models.CatNet.
        wrapper = nn.Module()
        wrapper.add_module("backbone", self.backbone)
        wrapper.add_module("pool", self.pool)
        wrapper.add_module("drop", nn.Dropout(checkpoint["model_cfg"]["dropout"]))
        wrapper.add_module("fc", self.fc)
        wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
        wrapper.eval()
        self.model = wrapper
        self.ood_stats = None
        stats_path = checkpoint_path.parent / "ood_stats.npz"
        if stats_path.exists():
            sha = hashlib.sha256()
            with checkpoint_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha.update(chunk)
            with np.load(stats_path, allow_pickle=False) as stats:
                if str(stats["checkpoint_id"].item()) != self.checkpoint_id or str(stats["checkpoint_sha256"].item()) != sha.hexdigest():
                    raise ValueError("Thống kê OOD không khớp checkpoint")
                means, precision, bank = stats["class_means"], stats["precision"], stats["train_features_l2"]
                k = int(stats["k"].item())
                if (means.shape != (12, 2048) or precision.shape != (2048, 2048)
                        or bank.ndim != 2 or bank.shape[1] != 2048 or not 0 < k <= len(bank)
                        or not all(np.isfinite(x).all() for x in (means, precision, bank))):
                    raise ValueError("Thống kê OOD không hợp lệ")
                self.ood_stats = (means, precision, bank, k)

    def input_image(self, img: Image.Image) -> Image.Image:
        rgb = img.convert("RGB")
        if self.preprocess_config["resize"] == "short_side_center_crop":
            rgb = TVF.resize(rgb, 256, interpolation=InterpolationMode.BILINEAR, antialias=True)
            return TVF.center_crop(rgb, [224, 224])
        return rgb.resize((224, 224), Image.Resampling.BILINEAR)

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        rgb = self.input_image(img)
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        chw = np.transpose((pixels - self.mean) / self.std, (2, 0, 1))
        return torch.from_numpy(chw.copy()).unsqueeze(0)

    def predict_raw(self, img: Image.Image):
        with torch.no_grad():
            feature_map = self.backbone(self.preprocess(img))
            features = self.pool(feature_map).flatten(1)
            logits = self.fc(features)
        return logits[0].numpy(), features[0].numpy()

    def predict(self, img: Image.Image):
        with torch.enable_grad():
            feature_map = self.backbone(self.preprocess(img))
            features = self.pool(feature_map).flatten(1)
            logits = self.fc(features)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_indices = torch.argsort(probabilities, descending=True)[:5]
            top_class = int(top_indices[0])
            gradients = torch.autograd.grad(logits[0, top_class], feature_map)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            low, high = cam.min(), cam.max()
            cam = (cam - low) / (high - low).clamp_min(1e-12)
            intensity = cam.detach().numpy()
            top5 = [{"label": self.classes[int(i)], "prob": float(probabilities[int(i)].detach())}
                    for i in top_indices]
            z = logits[0].detach().numpy().astype(np.float64)
            h = features[0].detach().numpy().astype(np.float64)

        ood_all = None
        if self.ood_stats is not None:
            means, precision, bank, k = self.ood_stats
            diff = h[None, :] - means
            distance_sq = np.sum((diff @ precision) * diff, axis=1)
            mahalanobis = -float(np.maximum(distance_sq.min(), 0))
            h_norm = h / max(float(np.linalg.norm(h)), 1e-12)
            distances = np.linalg.norm(bank - h_norm, axis=1)
            scores = [float(probabilities.max().detach()), float(z.max()),
                      float(np.logaddexp.reduce(z)), mahalanobis,
                      -float(np.partition(distances, k - 1)[k - 1])]
            names = ["MSP", "MaxLogit", "Energy Score", "Mahalanobis", "KNN (k=10)"]
            ood_all = [{"method": name, "score": score, "threshold": None, "accept": None}
                       for name, score in zip(names, scores)]

        # Transparent red/yellow PNG; opacity encodes Grad-CAM activation strength.
        rgba = np.zeros((224, 224, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = np.clip(255 * (1 - intensity), 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(intensity * 220, 0, 220).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
        model_input = io.BytesIO()
        self.input_image(img).save(model_input, format="PNG")
        return {
            "api_version": "1.0", "is_mock": False, "status": "unverified",
            "top_label": top5[0]["label"], "top5": top5,
            "ood": None, "ood_all": ood_all,
            "model": {"backbone": "ResNet50", "mode": self.mode,
                      "seed": self.seed, "checkpoint_id": self.checkpoint_id},
            "heatmap": base64.b64encode(buf.getvalue()).decode("ascii"),
            "cam_input": base64.b64encode(model_input.getvalue()).decode("ascii"),
        }
