import json

import numpy as np
import pandas as pd
import pytest

from scripts import rank_sweep

IMAGE_IDS = [f"img_{index:02d}" for index in range(24)]
LABELS = np.repeat(np.arange(12), 2)
EXCLUDED = ["img_05", "img_06"]

# Ảnh dự đoán sai của từng điểm dò. Mô hình "a": p1 sai 2 ảnh, p2 sai 3 ảnh (2 ảnh nằm trong EXCLUDED),
# nên p1 đứng trên p2 khi đủ ảnh nhưng p2 vượt lên khi bỏ hai ảnh đó.
WRONG = {
    ("a", "p1"): ["img_00", "img_02"],
    ("a", "p2"): ["img_05", "img_06", "img_10"],
    ("a", "p3"): ["img_01", "img_03", "img_08", "img_12", "img_14"],
    ("b", "p1"): ["img_00"],
    ("b", "p2"): ["img_08", "img_10"],  # không chạm lớp 2, 3 của các ảnh bị bỏ nên thứ hạng giữ nguyên
    ("b", "p3"): ["img_12", "img_14", "img_16"],
}


def logits_for(wrong):
    predictions = LABELS.copy()
    for image_id in wrong:
        index = IMAGE_IDS.index(image_id)
        predictions[index] = (LABELS[index] + 1) % 12
    return np.eye(12)[predictions] * 5.0


def build_runs(tmp_path, tamper=None):
    runs_root = tmp_path / "runs"
    rows = []
    for (model, point), wrong in WRONG.items():
        stage_dir = f"{model}/{point}/frozen"
        (runs_root / stage_dir).mkdir(parents=True)
        logits = logits_for(wrong)
        np.savez(runs_root / stage_dir / "val.npz", logits=logits, labels=LABELS, image_id=np.array(IMAGE_IDS))
        accuracy, macro_f1 = rank_sweep.validation_metrics(runs_root / stage_dir / "val.npz", set())
        if tamper == (model, point):
            macro_f1 += 1e-6
        rows.append({
            "sweep_point": point, "model": model, "seed": 42, "stage": "frozen", "evaluation": "classification",
            "val_macro_f1": macro_f1, "val_accuracy": accuracy, "best_epoch": 3, "size_megabytes": 10,
            "config_overrides": json.dumps({}), "stage_dir": stage_dir, "run_id": f"{model}-{point}", "protocol_id": "p",
        })
    source = tmp_path / "sweep"
    source.mkdir()
    pd.DataFrame(rows).to_csv(source / "sweep_results.csv", index=False)
    return source / "sweep_results.csv", runs_root


def test_exclusion_changes_rank(tmp_path):
    sweep, runs_root = build_runs(tmp_path)
    output = tmp_path / "out"
    rank_sweep.run([sweep], output, 24, None, EXCLUDED, runs_root)
    table = pd.read_csv(output / "stage1_exclude_val.csv")
    summary = pd.read_csv(output / "stage1_exclude_val_summary.csv").set_index("model")
    assert list(table.columns) == ["model", "sweep_point", "stage", "rank_full", "rank_excluded",
                                   "f1_full", "f1_excluded", "acc_full", "acc_excluded"]
    a = table[table.model == "a"].set_index("sweep_point")
    assert (a.loc["p1", "rank_full"], a.loc["p2", "rank_full"]) == (1, 2)
    assert (a.loc["p2", "rank_excluded"], a.loc["p1", "rank_excluded"]) == (1, 2)
    assert a.loc["p2", "acc_excluded"] == pytest.approx(21 / 22)
    assert not summary.loc["a", "top1_preserved"]
    assert summary.loc["b", "top1_preserved"]
    assert summary.loc["b", "kendall_tau"] == pytest.approx(1.0)
    assert (output / "stage1_ranking.csv").exists()  # đầu ra chính thức vẫn được ghi như cũ


def test_cross_check_mismatch_raises(tmp_path):
    sweep, runs_root = build_runs(tmp_path, tamper=("b", "p2"))
    with pytest.raises(ValueError, match="Kiểm chéo"):
        rank_sweep.run([sweep], tmp_path / "out", 24, None, EXCLUDED, runs_root)


def test_unknown_image_id_raises(tmp_path):
    sweep, runs_root = build_runs(tmp_path)
    with pytest.raises(ValueError, match="không có trong"):
        rank_sweep.run([sweep], tmp_path / "out", 24, None, ["img_05", "khong_ton_tai"], runs_root)


def test_exclude_requires_runs_root(tmp_path):
    sweep, _ = build_runs(tmp_path)
    with pytest.raises(ValueError, match="--runs-root"):
        rank_sweep.run([sweep], tmp_path / "out", 24, None, EXCLUDED, None)
