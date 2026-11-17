import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_official_tuned as mot  # noqa: E402
import rank_sweep  # noqa: E402


def rows_for(model, overrides, seeds, f1, acc=None, point="p"):
    out = []
    for seed in seeds:
        for stage, bump in (("frozen", 0.0), ("finetuned", 0.01)):
            out.append(
                {
                    "sweep_point": point, "model": model, "seed": seed, "stage": stage, "evaluation": "full",
                    "val_macro_f1": f1 + bump, "val_accuracy": (acc or f1) + bump, "best_epoch": 5,
                    "size_megabytes": 10.0, "config_overrides": json.dumps(overrides), "stage_dir": f"{model}/{seed}/{stage}",
                    "run_id": f"r{seed}", "protocol_id": "Nhom09_v2_2026-11-16",
                }
            )
    return out


def write(tmp_path, rows):
    folder = tmp_path / "s"
    folder.mkdir(exist_ok=True)
    path = folder / "sweep_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def full_rows():
    rows = []
    for model in mot.REQUIRED_MODELS:
        rows += rows_for(model, {"dropout": 0.5}, (42, 43, 44), 0.80, point="drop50")
        rows += rows_for(model, {"dropout": 0.0}, (42, 43, 44), 0.85, point="D")
        rows += rows_for(model, {"head_hidden": 512}, (42,), 0.99, point="head512")  # chỉ 1 seed: không phải ứng viên
    return rows


def test_picks_best_mean_over_three_seeds(tmp_path):
    path = write(tmp_path, full_rows())
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"profile": "official", "knn_k": 10}), encoding="utf-8")
    out, table = tmp_path / "tuned.json", tmp_path / "t.csv"
    assert mot.main([str(path), "--base-config", str(base), "--output", str(out), "--table", str(table)]) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert config["profile"] == "official" and config["knn_k"] == 10
    for model in mot.REQUIRED_MODELS:
        assert config["model_overrides"][model] == {"dropout": 0.0}
    assert "head_hidden" not in table.read_text(encoding="utf-8")


def test_missing_model_fails(tmp_path):
    rows = [r for r in full_rows() if r["model"] != "cnn_baseline"]
    path = write(tmp_path, rows)
    base = tmp_path / "base.json"
    base.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        mot.main([str(path), "--base-config", str(base), "--output", str(tmp_path / "o.json"), "--table", str(tmp_path / "t.csv")])


def test_tie_prefers_accuracy_then_fewer_keys():
    table = pd.DataFrame(
        [
            {"model": "m", "overrides": json.dumps({"a": 1, "b": 2}), "n_keys": 2, "f1_mean": 0.9, "f1_std": 0, "acc_mean": 0.9, "acc_std": 0},
            {"model": "m", "overrides": json.dumps({"a": 1}), "n_keys": 1, "f1_mean": 0.90005, "f1_std": 0, "acc_mean": 0.9, "acc_std": 0},
            {"model": "m", "overrides": json.dumps({"c": 1}), "n_keys": 1, "f1_mean": 0.8, "f1_std": 0, "acc_mean": 0.99, "acc_std": 0},
        ]
    )
    ranked = mot.rank(table).sort_values("rank")
    assert json.loads(ranked.iloc[0]["overrides"]) == {"a": 1}
