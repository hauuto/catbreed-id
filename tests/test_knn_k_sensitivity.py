import json

import numpy as np
import pandas as pd
import pytest

from scripts import knn_k_sensitivity as knn


def build_stage(tmp_path, seed=42, mismatch=False):
    rng = np.random.default_rng(seed)
    run_dir = tmp_path / f"run_s{seed}"
    stage_dir = run_dir / "finetuned"
    stage_dir.mkdir(parents=True)
    labels = np.repeat(np.arange(12), 20)
    train = rng.normal(size=(len(labels), 16))
    validation = rng.normal(size=(48, 16))
    validation[0] = train[0]
    probe = rng.normal(loc=2.0, size=(50, 16))
    val_ids = np.array([f"val-{index}" for index in range(len(validation))])
    np.savez_compressed(stage_dir / "train.npz", features=train, labels=labels)
    np.savez_compressed(stage_dir / "val.npz", features=validation, image_id=val_ids)
    np.savez_compressed(stage_dir / "probe_ood.npz", features=probe)
    (run_dir / "info.json").write_text(
        json.dumps({"model": "mobilenet_v2", "training_seed": seed}), encoding="utf-8"
    )
    val10 = knn.knn_scores(train, validation, 10)
    probe10 = knn.knn_scores(train, probe, 10)
    tau10 = float(np.quantile(val10, 0.05, method="linear"))
    metrics10 = knn.ood_metrics(val10, probe10, tau10)
    stored = val10.copy()
    if mismatch:
        stored[0] += 1e-4
    pd.DataFrame({"image_id": val_ids, "KNN": stored}).to_csv(
        stage_dir / "validation_scores.csv", index=False
    )
    pd.DataFrame([{"method": "KNN", **metrics10}]).to_csv(stage_dir / "probe_results.csv", index=False)
    return stage_dir, train, validation


def test_identical_query_has_zero_k1_score(tmp_path):
    _, train, validation = build_stage(tmp_path)
    scores = knn.knn_scores(train, validation, 1)
    # sklearn's optimized Euclidean path can leave about 1e-8 round-off for identical vectors.
    assert scores[0] == pytest.approx(0.0, abs=1e-7)


def test_analyze_stage_and_k10_cross_check(tmp_path):
    stage_dir, _, _ = build_stage(tmp_path)
    frame = knn.analyze_stage(stage_dir)
    assert frame.k.tolist() == [1, 5, 10, 20, 50]
    assert set(frame.model) == {"mobilenet_v2"}
    assert set(frame.seed) == {42}
    assert set(frame.stage) == {"finetuned"}


def test_train_smaller_than_k_is_rejected():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="nhỏ hơn k=50"):
        knn.knn_scores(rng.normal(size=(49, 4)), rng.normal(size=(2, 4)), 50)


def test_k10_validation_mismatch_is_rejected(tmp_path):
    stage_dir, _, _ = build_stage(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="max_abs_diff"):
        knn.analyze_stage(stage_dir)


def test_run_multiple_stages_writes_output(tmp_path, capsys):
    first, _, _ = build_stage(tmp_path / "a", seed=42)
    second, _, _ = build_stage(tmp_path / "b", seed=43)
    output = tmp_path / "knn_k_sensitivity.csv"
    frame = knn.run([first, second], output)
    assert output.is_file()
    assert len(frame) == 10
    assert "Trung bình ± độ lệch chuẩn" in capsys.readouterr().out


def test_forbidden_stage_name_is_rejected(tmp_path):
    forbidden = tmp_path / "near_ood"
    forbidden.mkdir()
    with pytest.raises(ValueError, match="bị cấm"):
        knn.analyze_stage(forbidden)
