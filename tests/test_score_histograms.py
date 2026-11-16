import json

import numpy as np
import pandas as pd
import pytest

from scripts import score_histograms


def make_stage(tmp_path, missing_column=False):
    rng = np.random.default_rng(42)
    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    methods = score_histograms.METHODS
    for filename, mean in [("test_scores.csv", 1.0), ("near_ood_scores.csv", 0.0)]:
        values = {"image_id": [f"{filename}-{index}" for index in range(200)]}
        for method in methods:
            values[method] = rng.normal(mean, 0.2, 200)
        frame = pd.DataFrame(values)
        if missing_column and filename == "test_scores.csv":
            frame = frame.drop(columns="KNN")
        frame.to_csv(stage / filename, index=False)
    thresholds = {method: 0.5 for method in methods}
    (stage / "thresholds.json").write_text(json.dumps({"T1": thresholds}), encoding="utf-8")
    return stage


def test_creates_figure_and_correct_rates(tmp_path):
    stage = make_stage(tmp_path)
    output = tmp_path / "result"
    image, table = score_histograms.run(stage, output, bins=20)
    assert image.is_file() and image.stat().st_size > 10_000
    summary = pd.read_csv(table)
    id_msp = summary[(summary.method == "MSP") & (summary.group == "ID test")].iloc[0]
    source = pd.read_csv(stage / "test_scores.csv")
    assert id_msp.acceptance_rate == pytest.approx((source.MSP >= 0.5).mean())
    assert set(summary.method) == set(score_histograms.METHODS)


def test_missing_column_and_threshold_are_rejected(tmp_path):
    stage = make_stage(tmp_path, missing_column=True)
    with pytest.raises(ValueError, match="thiếu cột"):
        score_histograms.read_inputs(stage)

    stage = make_stage(tmp_path / "second")
    (stage / "thresholds.json").write_text(json.dumps({"T1": {"MSP": 0.5}}), encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu T1"):
        score_histograms.read_inputs(stage)


def test_requires_at_least_one_ood_group(tmp_path):
    stage = make_stage(tmp_path)
    (stage / "near_ood_scores.csv").unlink()
    with pytest.raises(ValueError, match="ít nhất một"):
        score_histograms.read_inputs(stage)
