import json

import pandas as pd
import pytest

from scripts import rank_sweep


def fake_rows(protocol="p1"):
    rows = []
    for model in ["resnet50", "mobilenet_v2"]:
        for point_index, point in enumerate(["unf25", "unf50", "drop0"]):
            for stage_index, stage in enumerate(["frozen", "finetuned"]):
                f1 = 0.70 + point_index * 0.01 + stage_index * 0.001
                accuracy = 0.75 + point_index * 0.01 + stage_index * 0.001
                if point == "unf25":
                    f1 = 0.70000 if stage == "frozen" else 0.70005
                    accuracy = 0.80 if stage == "frozen" else 0.79
                rows.append(
                    {
                        "sweep_point": point,
                        "model": model,
                        "seed": 42,
                        "stage": stage,
                        "evaluation": "classification",
                        "val_macro_f1": f1,
                        "val_accuracy": accuracy,
                        "best_epoch": 3 + stage_index,
                        "size_megabytes": 10,
                        "config_overrides": json.dumps({"dropout": 0.0}),
                        "stage_dir": f"runs/{model}/{point}/{stage}",
                        "run_id": f"{model}-{point}",
                        "protocol_id": protocol,
                    }
                )
    return rows


def write_sweep(path, rows):
    path.mkdir()
    target = path / "sweep_results.csv"
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def test_rank_and_factor_groups(tmp_path):
    source = write_sweep(tmp_path / "source", fake_rows())
    merged = rank_sweep.read_sweep_files([source])
    ranking, top, factors, sensitivity = rank_sweep.build_outputs(merged, n_val=100)

    selected = ranking[(ranking.model == "resnet50") & (ranking.sweep_point == "unf25")].iloc[0]
    assert selected.stage == "frozen"  # F1 delta < 1e-4, so accuracy breaks the tie.
    assert rank_sweep.factor_group("unf25") == "unf"
    assert rank_sweep.factor_group("drop50") == "drop"
    assert rank_sweep.factor_group("lr_f0.001_b1e-05") == "lr"
    assert len(top) == 6
    assert set(factors.factor_group) == {"unf", "drop"}
    assert not sensitivity.empty
    assert not sensitivity[sensitivity.criterion_b == "val_nll"].available.any()


def test_duplicate_key_is_rejected(tmp_path):
    rows = fake_rows()
    rows.append(dict(rows[0]))
    source = write_sweep(tmp_path / "source", rows)
    with pytest.raises(ValueError, match="Trùng khóa"):
        rank_sweep.read_sweep_files([source])


def test_protocol_mismatch_is_rejected(tmp_path):
    first = write_sweep(tmp_path / "first", fake_rows("p1")[:6])
    second = write_sweep(tmp_path / "second", fake_rows("p2")[6:])
    with pytest.raises(ValueError, match="protocol_id"):
        rank_sweep.read_sweep_files([first, second])


def test_epoch_breaks_exact_tie():
    late = {"val_macro_f1": 0.8, "val_accuracy": 0.9, "best_epoch": 5}
    early = {"val_macro_f1": 0.8, "val_accuracy": 0.9, "best_epoch": 3}
    assert rank_sweep.is_better(early, late)


def test_calibration_count_and_forbidden_name(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"n_images": 120}), encoding="utf-8")
    assert rank_sweep.validation_count(None, calibration) == 120
    forbidden = tmp_path / "sweep_test_results.csv"
    forbidden.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="bị cấm"):
        rank_sweep.reject_forbidden_path(forbidden)


def test_nll_sensitivity_is_available(tmp_path):
    source = write_sweep(tmp_path / "source", fake_rows())
    merged = rank_sweep.read_sweep_files([source])
    merged["val_nll"] = 1.0 - merged["val_accuracy"]
    ranking = rank_sweep.select_and_rank(merged)
    sensitivity = rank_sweep.criterion_sensitivity(ranking)
    assert sensitivity[sensitivity.criterion_b == "val_nll"].available.all()


def test_val_loss_is_used_as_nll(tmp_path):
    # sweep_results.csv của notebook có val_loss (cross-entropy, không label smoothing), không có val_nll
    source = write_sweep(tmp_path / "source", fake_rows())
    merged = rank_sweep.read_sweep_files([source])
    merged["val_loss"] = 1.0 - merged["val_accuracy"]
    ranking = rank_sweep.select_and_rank(merged)
    sensitivity = rank_sweep.criterion_sensitivity(ranking)
    assert sensitivity[sensitivity.criterion_b == "val_nll"].available.all()
