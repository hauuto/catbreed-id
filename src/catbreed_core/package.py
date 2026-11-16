"""Nạp gói triển khai (checkpoint + ngưỡng + hiệu chỉnh) và suy luận một ảnh."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.special import softmax

from .model import CAT_BREEDS, build_from_checkpoint
from .preprocess import eval_transform, read_rgb_image
from .scores import METHODS, DistanceScores, logit_scores


@dataclass
class Package:
    model: object
    distance: DistanceScores
    thresholds: dict  # thresholds.json đầy đủ (có "T1", "T_star", "checkpoint_sha256")
    calibration: dict  # calibration.json (có "temperature")
    locked_selection: dict  # {} nếu không có locked_selection.json
    saved: dict  # checkpoint đã nạp (model_name, config, ...)
    checkpoint_sha256: str
    is_smoke_test: bool  # như load_inference_bundle của notebook: gói của profile smoke không dùng để phục vụ
    device: str = "cpu"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_package(directory, device="cpu"):
    directory = Path(directory)
    checkpoint = directory / "checkpoint.pt"
    thresholds = _read_json(directory / "thresholds.json")
    calibration = _read_json(directory / "calibration.json")
    locked_path = directory / "locked_selection.json"
    locked = _read_json(locked_path) if locked_path.is_file() else {}

    digest = _sha256(checkpoint)
    if thresholds["checkpoint_sha256"] != digest:
        raise ValueError("Ngưỡng không thuộc checkpoint này.")
    if locked:
        expected = locked.get("deployment", {}).get("checkpoint_sha256")
        if expected is not None and expected != digest:
            raise ValueError("locked_selection không thuộc checkpoint này.")

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    return Package(
        model=build_from_checkpoint(saved, device),
        distance=DistanceScores.load(directory / "distance_parameters.npz"),
        thresholds=thresholds,
        calibration=calibration,
        locked_selection=locked,
        saved=saved,
        checkpoint_sha256=digest,
        is_smoke_test=saved["config"].get("profile") == "smoke",
        device=device,
    )


def predict_arrays(pkg, image, exif=False):
    # image: đường dẫn hoặc PIL.Image. Trả (logits thô [1, 12], đặc trưng sau GAP [1, d], điểm T = 1).
    if exif:
        if isinstance(image, Image.Image):
            image = ImageOps.exif_transpose(image)
        else:
            with Image.open(image) as source:  # đóng file sau khi xoay
                source.load()
                image = ImageOps.exif_transpose(source)
    tensor = eval_transform()(read_rgb_image(image)).unsqueeze(0).to(pkg.device)
    with torch.no_grad():
        features = pkg.model.forward_features(tensor)
        logits = pkg.model.head(features).cpu().numpy()
    features = features.cpu().numpy()
    scores = {**logit_scores(logits), **pkg.distance.score(features)}
    return logits, features, {name: scores[name] for name in METHODS}


def decide(pkg, scores, logits, method=None):
    selected = pkg.locked_selection.get("rejection", {}).get("method") if pkg.locked_selection else None
    method = selected or method
    if method not in METHODS:
        raise ValueError(f"Phương pháp từ chối không hợp lệ: {method!r}")
    tau = float(pkg.thresholds["T1"][method])
    score = float(np.asarray(scores[method]).reshape(-1)[0])
    temperature = float(pkg.calibration["temperature"])
    probs = softmax(np.asarray(logits, dtype=np.float64).reshape(-1) / temperature)
    return {
        "method": method,
        "tau": tau,
        "score": score,
        "accepted": bool(score >= tau),
        "probs": probs,
        "top_class": CAT_BREEDS[int(probs.argmax())],
    }
