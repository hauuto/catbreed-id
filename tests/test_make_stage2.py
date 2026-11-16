import json

import pandas as pd
import pytest

from scripts import make_stage2


def input_frames():
    ranking_rows = []
    factor_rows = []
    models = ["resnet50", "mobilenet_v2", "efficientnet_b0", "cnn_baseline"]
    for model in models:
        top_overrides = {"dropout": 0.5}
        ranking_rows.append(
            {
                "model": model,
                "rank": 1,
                "sweep_point": "drop50",
                "config_overrides": json.dumps(top_overrides),
            }
        )
        if model != "cnn_baseline":
            factor_rows.extend(
                [
                    {
                        "model": model,
                        "factor_group": "drop",
                        "rank": 1,
                        "sweep_point": "drop50",
                        "config_overrides": json.dumps({"dropout": 0.5}),
                    },
                    {
                        "model": model,
                        "factor_group": "unf",
                        "rank": 2,
                        "sweep_point": "unf50",
                        "config_overrides": json.dumps({"fine_tune_fraction": 0.5}),
                    },
                ]
            )
    return pd.DataFrame(ranking_rows), pd.DataFrame(factor_rows)


def base_config():
    return {
        "profile": "sweep",
        "protocol_id": "p1",
        "owner": "Khoi",
        "models": ["mobilenet_v2"],
        "training_seeds": [42],
        "frozen_epochs": 30,
        "finetuned_epochs": 30,
        "baseline_epochs": 80,
        "sweep_grid": [{"name": "old", "overrides": {"dropout": 0.0}}],
    }


def test_generates_all_stage2_configs():
    ranking, factors = input_frames()
    generated, warnings = make_stage2.generate_configs(ranking, factors, base_config())
    assert not warnings
    assert set(generated) == {
        "stage2_A.json",
        "stage2_B.json",
        "stage2_C.json",
        "stage2_Baseline.json",
        "stage2_D_resnet50.json",
        "stage2_D_mobilenet_v2.json",
        "stage2_D_efficientnet_b0.json",
    }
    assert generated["stage2_A.json"]["models"] == ["resnet50"]
    assert generated["stage2_A.json"]["training_seeds"] == [43, 44]
    assert generated["stage2_D_resnet50.json"]["training_seeds"] == [42, 43, 44]
    for config in generated.values():
        make_stage2.validate_stage2_config(config)


def test_d_skips_seed_42_when_identical_to_top():
    ranking, factors = input_frames()
    factors = factors[factors.factor_group == "drop"]
    generated, _ = make_stage2.generate_configs(ranking, factors, base_config())
    d_config = generated["stage2_D_mobilenet_v2.json"]
    assert d_config["training_seeds"] == [43, 44]
    assert d_config["stage2_source"]["duplicates_stage1_top"] is True


def test_conflicting_factor_overrides_are_rejected():
    _, factors = input_frames()
    extra = pd.DataFrame(
        [
            {
                "model": "resnet50",
                "factor_group": "other",
                "rank": 3,
                "sweep_point": "drop0",
                "config_overrides": json.dumps({"dropout": 0.0}),
            }
        ]
    )
    factors = pd.concat([factors, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Xung đột override"):
        make_stage2.combine_factor_overrides(factors, "resnet50")


def test_missing_model_warns_and_skips():
    ranking, factors = input_frames()
    ranking = ranking[ranking.model != "efficientnet_b0"]
    generated, warnings = make_stage2.generate_configs(ranking, factors, base_config())
    assert "stage2_C.json" not in generated
    assert any("efficientnet_b0" in warning for warning in warnings)


def test_validator_rejects_unknown_and_fixed_override_keys():
    config = make_stage2.make_config(
        base_config(), "resnet50", [43, 44], "valid", {"dropout": 0.5}, {"kind": "test"}
    )
    config["sweep_grid"][0]["overrides"] = {"not_a_key": 1}
    with pytest.raises(ValueError, match="khóa không có"):
        make_stage2.validate_stage2_config(config)
    config["sweep_grid"][0]["overrides"] = {"protocol_id": "other"}
    with pytest.raises(ValueError, match="khóa cố định"):
        make_stage2.validate_stage2_config(config)


def test_run_writes_manifest(tmp_path):
    ranking, factors = input_frames()
    ranking_path = tmp_path / "stage1_ranking.csv"
    factors_path = tmp_path / "stage1_best_per_factor.csv"
    config_path = tmp_path / "sweep_khoi.json"
    ranking.to_csv(ranking_path, index=False)
    factors.to_csv(factors_path, index=False)
    config_path.write_text(json.dumps(base_config()), encoding="utf-8")
    output = tmp_path / "generated"
    make_stage2.run(ranking_path, factors_path, config_path, output)
    manifest = json.loads((output / "stage2_manifest.json").read_text(encoding="utf-8"))
    assert "stage2_A.json" in manifest["generated"]
