import pandas as pd
import pytest

from scripts import geometry_table


def row(name, seed, center_accuracy, letterbox_accuracy, center_f1, letterbox_f1):
    values = {
        "name": name,
        "model": "mobilenet_v2",
        "seed": seed,
        "stage": "finetuned",
        "augmentation": True,
        "run_id": f"run-{seed}",
        "sweep_point": "drop50",
    }
    for metric in geometry_table.METRICS:
        values[f"center_{metric}"] = center_f1
        values[f"letterbox_{metric}"] = letterbox_f1
    values["center_accuracy"] = center_accuracy
    values["letterbox_accuracy"] = letterbox_accuracy
    return values


def write_file(folder, rows):
    folder.mkdir(parents=True)
    path = folder / "eval_geometry.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_summary_uses_paired_differences(tmp_path):
    path = write_file(
        tmp_path / "outputs" / "study",
        [row("a", 42, 0.8, 0.9, 0.7, 0.6), row("b", 43, 0.6, 0.6, 0.5, 0.7)],
    )
    frame = geometry_table.read_geometry([path])
    summary = geometry_table.summarize(frame).iloc[0]
    assert summary.delta_accuracy_mean == pytest.approx(0.05)
    assert summary.delta_macro_f1_mean == pytest.approx(0.05)
    assert summary.letterbox_better_accuracy == 1
    assert summary.letterbox_equal_accuracy == 1
    assert summary.letterbox_worse_macro_f1 == 1


def test_directory_discovery_and_output(tmp_path):
    write_file(tmp_path / "outputs" / "one", [row("a", 42, 0.8, 0.9, 0.7, 0.8)])
    output = tmp_path / "summary.csv"
    result = geometry_table.run([tmp_path / "outputs"], output)
    assert output.is_file()
    assert result.iloc[0].delta_accuracy_std == 0


def test_missing_column_and_duplicate_are_rejected(tmp_path):
    path = write_file(tmp_path / "outputs" / "bad", [row("a", 42, 0.8, 0.9, 0.7, 0.8)])
    bad = pd.read_csv(path).drop(columns="center_macro_recall")
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="thiếu cột"):
        geometry_table.read_geometry([path])

    duplicate = write_file(
        tmp_path / "outputs" / "duplicate",
        [row("a", 42, 0.8, 0.9, 0.7, 0.8), row("a", 42, 0.8, 0.9, 0.7, 0.8)],
    )
    with pytest.raises(ValueError, match="bị trùng"):
        geometry_table.read_geometry([duplicate])
