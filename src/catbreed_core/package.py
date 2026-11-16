"""Loading and inference helpers for a locked deployment package."""

import hashlib
import json
import time
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
    thresholds: dict
    calibration: dict
    locked_selection: dict
    saved: dict
    checkpoint_sha256: str
    is_smoke_test: bool
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
    required = (
        checkpoint,
        directory / "thresholds.json",
        directory / "calibration.json",
        directory / "distance_parameters.npz",
    )
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu file model_package: {', '.join(missing)}")
    thresholds = _read_json(directory / "thresholds.json")
    calibration = _read_json(directory / "calibration.json")
    locked_path = directory / "locked_selection.json"
    locked = _read_json(locked_path) if locked_path.is_file() else {}
    digest = _sha256(checkpoint)
    if "checkpoint_sha256" in thresholds and thresholds["checkpoint_sha256"] != digest:
        raise ValueError("Ngưỡng không thuộc checkpoint này.")
    if locked:
        expected = locked.get("deployment", {}).get("checkpoint_sha256")
        if expected is not None and expected != digest:
            raise ValueError("SHA-256 checkpoint.pt không khớp locked_selection.json")
        entries = locked.get("checkpoints", [])
        if entries:
            stage_dir = locked.get("deployment", {}).get("stage_dir")
            entry = next((item for item in entries if item.get("stage_dir") == stage_dir), None)
            if entry is None:
                raise ValueError("Không tìm thấy checkpoint triển khai trong danh sách đã khóa.")
            for name, expected_hash in entry.get("sha256", {}).items():
                path = directory / name
                if path.is_file() and _sha256(path) != expected_hash:
                    raise ValueError(f"SHA-256 {name} không khớp locked_selection.json")
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


def predict_arrays(pkg, image, exif=False, return_graph=False):
    def sync():
        if torch.device(pkg.device).type == "cuda":
            torch.cuda.synchronize(pkg.device)

    sync()
    started = time.perf_counter()
    if exif:
        if isinstance(image, Image.Image):
            image = ImageOps.exif_transpose(image)
        else:
            with Image.open(image) as source:
                source.load()
                image = ImageOps.exif_transpose(source)
    dtype = next(pkg.model.parameters()).dtype
    tensor = eval_transform()(read_rgb_image(image)).unsqueeze(0).to(device=pkg.device, dtype=dtype)
    sync()
    preprocessed = time.perf_counter()
    context = torch.enable_grad() if return_graph else torch.no_grad()
    with context:
        feature_map = pkg.model.features(tensor)
        features_tensor = pkg.model.pool(feature_map).flatten(1)
        logits_tensor = pkg.model.head(features_tensor)
    sync()
    inferred = time.perf_counter()
    with torch.no_grad():
        logits = logits_tensor.detach().cpu().numpy()
        features = features_tensor.detach().cpu().numpy()
    scores = {**logit_scores(logits), **pkg.distance.score(features)}
    rejection = time.perf_counter()
    result = (logits, features, {name: scores[name] for name in METHODS})
    if return_graph:
        result += (
            feature_map,
            logits_tensor,
            {
                "preprocess": (preprocessed - started) * 1000,
                "inference": (inferred - preprocessed) * 1000,
                "rejection": (rejection - inferred) * 1000,
            },
        )
    return result


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
        "top_class": next(name.replace("_", " ") for name, index in pkg.saved["class_mapping"].items() if index == int(probs.argmax())) if hasattr(pkg, "saved") else CAT_BREEDS[int(probs.argmax())],
    }
