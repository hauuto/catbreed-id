import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_report_tables", Path(__file__).resolve().parents[1] / "scripts" / "build_report_tables.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def checkpoints():
    for model in ["resnet50", "mobilenet_v2"]:
        for seed in [42, 43, 44]:
            yield model, seed, "frozen"


def make_study(root: Path) -> Path:
    study = root / "study"
    study.mkdir()
    rows = [
        {"model": m, "seed": s, "stage": st, "augmentation": True, "accuracy": 0.9 + 0.01 * (s - 42), "macro_f1": 0.8}
        for m, s, st in checkpoints()
    ]
    pd.DataFrame(rows).to_csv(study / "test_classification.csv", index=False)
    ablation = [
        {"model": m, "seed": s, "stage": st, "augmentation": aug, "val_accuracy": 0.9, "val_macro_f1": 0.8,
         "test_accuracy": 0.7 + 0.1 * aug, "test_macro_f1": 0.6}
        for m, s, st in checkpoints() for aug in [True, False]
    ]
    pd.DataFrame(ablation).to_csv(study / "augmentation_ablation.csv", index=False)
    ood, paired = [], []
    for m, s, st in checkpoints():
        for split in ["near_ood", "far_ood"]:
            for method in module.METHODS:
                ood.append({"model": m, "seed": s, "stage": st, "augmentation": True, "ood_split": split,
                            "method": method, "auroc": 0.8, "fpr_at_95_tpr": 0.3, "fpr_at_threshold": 0.2})
                paired.append({"model": m, "seed": s, "stage": st, "augmentation": True, "ood_split": split,
                               "comparison": method, "estimate": 0.8, "ci_low": 0.7, "ci_high": 0.9})
            paired.append({"model": m, "seed": s, "stage": st, "augmentation": True, "ood_split": split,
                           "comparison": "Delta_c", "estimate": 0.1, "ci_low": 0.0, "ci_high": 0.2})
    pd.DataFrame(ood).to_csv(study / "ood_results.csv", index=False)
    pd.DataFrame(paired).to_csv(study / "ood_paired_confidence.csv", index=False)
    (study / "hypothesis_near_ood.json").write_text(
        json.dumps({"near_ood": {"n_checkpoints": 6, "estimate": 0.123456, "ci_low": 0.01, "ci_high": 0.2,
                                 "bootstrap_repeats": 1000, "conclusion": "ủng hộ giả thuyết"}}),
        encoding="utf-8",
    )
    calibration = [
        {"model": m, "seed": s, "stage": st, "augmentation": True, "temperature": 1.5, "accuracy": 0.9,
         "nll_before": 0.5, "nll_after": 0.4, "ece_before": 0.06, "ece_after": 0.02}
        for m, s, st in checkpoints()
    ]
    pd.DataFrame(calibration).to_csv(study / "test_calibration.csv", index=False)
    return study


def test_build_all_tables(tmp_path):
    out = tmp_path / "out"
    written = module.build(make_study(tmp_path), out)
    assert set(written) == {
        "classification_id_test.csv", "augmentation_ablation.csv", "ood_by_method.csv", "ood_auroc_ci.csv",
        "hypothesis_near_ood.csv", "calibration.csv",
    }
    classification = pd.read_csv(out / "classification_id_test.csv")
    row = classification[classification.model == "resnet50"].iloc[0]
    assert row.n_seeds == 3 and row.accuracy_mean == pytest.approx(0.91) and row.accuracy_std == pytest.approx(0.01)

    ablation = pd.read_csv(out / "augmentation_ablation.csv")
    assert len(ablation) == 4  # 2 mô hình × 2 chế độ tăng cường
    assert sorted(ablation.test_accuracy_mean.unique()) == [0.7, 0.8]

    ood = pd.read_csv(out / "ood_by_method.csv")
    assert set(ood.group) == {"near", "far"}
    assert len(ood) == 2 * 2 * 5  # model × group × method (một stage)

    ci = pd.read_csv(out / "ood_auroc_ci.csv")
    assert set(ci.method) == set(module.METHODS)  # Delta_c không lẫn vào bảng AUROC
    assert ci.ci_low.eq(0.7).all() and ci.ci_high.eq(0.9).all()

    hypothesis = pd.read_csv(out / "hypothesis_near_ood.csv")
    assert hypothesis.estimate.iloc[0] == 0.1235  # làm tròn 4 chữ số, không tính lại
    assert hypothesis.conclusion.iloc[0] == "ủng hộ giả thuyết"

    calibration = pd.read_csv(out / "calibration.csv")
    assert calibration.nll_after_mean.iloc[0] == 0.4


def test_missing_sources_warn_without_inventing_numbers(tmp_path, capsys):
    study = tmp_path / "empty"
    study.mkdir()
    assert module.build(study, tmp_path / "out") == []
    assert "CẢNH BÁO" in capsys.readouterr().err
