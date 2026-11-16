import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catbreed_core import (  # noqa: E402
    CAT_BREEDS,
    CatClassifier,
    DistanceScores,
    build_from_checkpoint,
    decide,
    load_package,
    predict_arrays,
)
from catbreed_core.model import CLASS_TO_IDX, MODEL_NAMES  # noqa: E402

REPO_ID = "catbreed-id/catbreed-experiments"
STAGE = "runs/20261113_145326_efficientnet_b0_s42_df1edde1/frozen/"
FILES = ["checkpoint.pt", "thresholds.json", "calibration.json", "distance_parameters.npz", "val.npz", "validation_scores.csv"]


def checkpoint_for(name):
    model = CatClassifier(name, pretrained=False)
    return {
        "model_name": name,
        "state_dict": model.state_dict(),
        "class_mapping": CLASS_TO_IDX,
        "config": {"dropout": 0.3, "head_hidden": 0, "baseline_batchnorm": False, "profile": "official"},
    }


def write_package(directory, sha_override=None):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    saved = checkpoint_for("cnn_baseline")
    torch.save(saved, directory / "checkpoint.pt")
    import hashlib

    digest = hashlib.sha256((directory / "checkpoint.pt").read_bytes()).hexdigest()
    rng = np.random.RandomState(0)
    labels = np.repeat(np.arange(12), 5)
    DistanceScores().fit(rng.randn(60, 128), labels, 3).save(directory / "distance_parameters.npz")
    thresholds = {"checkpoint_sha256": sha_override or digest, "T1": {"MSP": 0.5}}
    (directory / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
    (directory / "calibration.json").write_text(json.dumps({"temperature": 1.5}), encoding="utf-8")
    return digest


# 1. Dữ liệu thật: cần HF_TOKEN và OXFORD_PETS_ROOT (thư mục oxford-iiit-pet, chứa images/<image_id>.jpg)
def test_real_validation_images_match_stored_outputs():
    token = os.environ.get("HF_TOKEN")
    if not token:
        pytest.skip("Cần biến môi trường HF_TOKEN")
    root = os.environ.get("OXFORD_PETS_ROOT")
    if not root or not (Path(root) / "images").is_dir():
        pytest.skip("Cần OXFORD_PETS_ROOT trỏ tới thư mục oxford-iiit-pet có images/")
    import pandas as pd
    from huggingface_hub import hf_hub_download

    local = {
        name: hf_hub_download(REPO_ID, STAGE + name, repo_type="dataset", token=token) for name in FILES
    }
    directory = Path(local["checkpoint.pt"]).parent
    pkg = load_package(directory)
    with np.load(local["val.npz"], allow_pickle=False) as stored:
        image_ids, stored_logits = stored["image_id"][:20], stored["logits"][:20]
    table = pd.read_csv(local["validation_scores.csv"]).iloc[:20]
    assert table["image_id"].tolist() == image_ids.tolist()
    for index, image_id in enumerate(image_ids):
        logits, _, scores = predict_arrays(pkg, Path(root) / "images" / f"{image_id}.jpg", exif=False)
        assert np.max(np.abs(logits[0] - stored_logits[index])) < 1e-4
        for method in ["MSP", "MaxLogit", "Energy", "KNN"]:
            assert abs(scores[method][0] - table[method].iloc[index]) < 1e-4
        expected = table["Mahalanobis"].iloc[index]
        assert abs(scores["Mahalanobis"][0] - expected) <= 1e-6 * abs(expected)


# 2. Bốn kiến trúc dựng lại được với strict=True
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_build_from_checkpoint_all_architectures(name):
    saved = checkpoint_for(name)
    model = build_from_checkpoint(saved)
    model.load_state_dict(saved["state_dict"], strict=True)
    assert not model.training


# 3. Sai checkpoint_sha256 thì load_package phải raise ValueError
def test_load_package_rejects_wrong_sha(tmp_path):
    good = tmp_path / "good"
    write_package(good)
    assert load_package(good).model.name == "cnn_baseline"
    bad = tmp_path / "bad"
    write_package(bad, sha_override="0" * 64)
    with pytest.raises(ValueError):
        load_package(bad)


# 4. decide: score bằng, lớn hơn và nhỏ hơn tau
@pytest.mark.parametrize("score, accepted", [(0.5, True), (0.6, True), (0.4, False)])
def test_decide_threshold(score, accepted):
    pkg = SimpleNamespace(
        locked_selection={}, thresholds={"T1": {"MSP": 0.5}}, calibration={"temperature": 2.0}
    )
    logits = np.zeros((1, 12))
    logits[0, 3] = 4.0
    result = decide(pkg, {"MSP": np.array([score])}, logits, method="MSP")
    assert result["accepted"] is accepted
    assert result["tau"] == 0.5
    assert result["top_class"] == CAT_BREEDS[3]
    assert abs(result["probs"].sum() - 1.0) < 1e-12


def test_decide_prefers_locked_method():
    pkg = SimpleNamespace(
        locked_selection={"rejection": {"method": "KNN"}},
        thresholds={"T1": {"MSP": 0.5, "KNN": -1.0}},
        calibration={"temperature": 1.0},
    )
    result = decide(pkg, {"MSP": np.array([0.1]), "KNN": np.array([-0.5])}, np.zeros((1, 12)), method="MSP")
    assert result["method"] == "KNN" and result["accepted"] is True


# 5. Cờ smoke và exif=True với đường dẫn
def test_smoke_flag_and_exif_path(tmp_path):
    write_package(tmp_path / "pkg")
    pkg = load_package(tmp_path / "pkg")
    assert pkg.is_smoke_test is False
    from PIL import Image

    path = tmp_path / "rgba.png"
    Image.new("RGBA", (300, 260), (10, 200, 30, 0)).save(path)
    a = predict_arrays(pkg, path, exif=True)
    b = predict_arrays(pkg, path, exif=False)
    assert np.allclose(a[0], b[0])
