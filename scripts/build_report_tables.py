"""Dựng bảng báo cáo cấp study từ các CSV/JSON mà notebooks/kaggle_runner.ipynb đã ghi.

Tên tệp và tên cột lấy từ write_tables (study.directory/<tên bảng>.csv), REPORT_TABLES và hypothesis_outputs
của notebook. Script không tính lại CI hay Δ; bảng nào thiếu tệp nguồn thì cảnh báo và bỏ qua.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DIGITS = 4
OOD_GROUPS = {"near_ood": "near", "far_ood": "far", "unseen_cat": "unseen_cat"}
OOD_METRICS = ["auroc", "fpr_at_95_tpr", "fpr_at_threshold"]  # OOD_METRICS của notebook (tập con)
METHODS = ["MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN"]
EXPECTED_SEEDS = {42, 43, 44}


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def mean_std(frame, keys, metrics):
    # Chép nguyên từ notebook (mục 1)
    grouped = frame.astype({metric: float for metric in metrics}).groupby(keys, sort=False)[metrics]
    table = grouped.agg(["mean", "std"])  # Độ lệch chuẩn mẫu (ddof = 1) qua các seed
    table.columns = [f"{metric}_{statistic}" for metric, statistic in table.columns]
    table.insert(0, "n_seeds", grouped.size())
    return table.reset_index()


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() == "true")


def read_table(study: Path, name: str, required: list[str]) -> pd.DataFrame | None:
    path = study / f"{name}.csv"
    if not path.is_file():
        print(f"CẢNH BÁO: thiếu {path.name}; bỏ qua các bảng phụ thuộc", file=sys.stderr)
        return None
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        print(f"CẢNH BÁO: {path.name} thiếu cột {missing}; bỏ qua", file=sys.stderr)
        return None
    if "seed" in frame.columns and not set(frame["seed"].unique()) <= EXPECTED_SEEDS:
        print(f"CẢNH BÁO: {path.name} có seed ngoài {sorted(EXPECTED_SEEDS)}", file=sys.stderr)
    return frame


def write(table: pd.DataFrame, out: Path, name: str, written: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.round(DIGITS).to_csv(out / name, index=False)
    written.append(name)


def augmented(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[as_bool(frame["augmentation"])]  # Bảng chính chỉ gồm checkpoint có tăng cường


def build(study: Path, out: Path) -> list[str]:
    written: list[str] = []
    keys = ["model", "stage"]

    # a. Phân loại ID test (test_classification)
    metrics = ["accuracy", "macro_f1"]
    frame = read_table(study, "test_classification", keys + ["seed", "augmentation"] + metrics)
    if frame is not None:
        write(mean_std(augmented(frame), keys, metrics), out, "classification_id_test.csv", written)

    # b. Có / không tăng cường (augmentation_ablation)
    metrics = ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]
    frame = read_table(study, "augmentation_ablation", keys + ["augmentation"] + metrics)
    if frame is not None:
        write(mean_std(frame, keys + ["augmentation"], metrics), out, "augmentation_ablation.csv", written)

    # c. OOD (ood_results; CI từ ood_paired_confidence)
    frame = read_table(
        study, "ood_results", ["ood_split", "model", "stage", "method", "augmentation"] + OOD_METRICS
    )
    if frame is not None:
        table = mean_std(augmented(frame), ["ood_split", "model", "stage", "method"], OOD_METRICS)
        group = table["ood_split"].map(OOD_GROUPS)
        table = table.drop(columns="ood_split")
        table.insert(0, "group", group)
        write(table, out, "ood_by_method.csv", written)
    paired = read_table(
        study, "ood_paired_confidence",
        ["model", "seed", "stage", "augmentation", "ood_split", "comparison", "estimate", "ci_low", "ci_high"],
    )
    if paired is not None:
        ci = augmented(paired)
        ci = ci[ci["comparison"].isin(METHODS)].copy()  # AUROC từng phương pháp, nguyên giá trị của notebook
        ci.insert(0, "group", ci["ood_split"].map(OOD_GROUPS))
        columns = ["group", "model", "seed", "stage", "comparison", "estimate", "ci_low", "ci_high"]
        write(ci[columns].rename(columns={"comparison": "method", "estimate": "auroc"}), out, "ood_auroc_ci.csv", written)

    # d. Giả thuyết near-OOD (hypothesis_near_ood.json), lấy nguyên
    path = study / "hypothesis_near_ood.json"
    if path.is_file():
        near = json.loads(path.read_text(encoding="utf-8"))["near_ood"]
        write(pd.DataFrame([near]), out, "hypothesis_near_ood.csv", written)
    else:
        print(f"CẢNH BÁO: thiếu {path.name}; bỏ qua", file=sys.stderr)

    # e. Hiệu chỉnh trước/sau T (test_calibration)
    metrics = ["temperature", "accuracy", "nll_before", "nll_after", "ece_before", "ece_after"]
    frame = read_table(study, "test_calibration", keys + ["augmentation"] + metrics)
    if frame is not None:
        write(mean_std(augmented(frame), keys, metrics), out, "calibration.csv", written)

    # f. Grad-CAM IoU và độ bền, chỉ khi notebook đã ghi bảng
    metrics = ["miou", "miou_correct", "miou_incorrect"]
    frame = read_table(study, "gradcam_summary", keys + ["augmentation"] + metrics)
    if frame is not None:
        write(mean_std(augmented(frame), keys, metrics), out, "gradcam.csv", written)
    group = ["family", "level", "parameter", "in_augmentation_range"]
    metrics = ["accuracy", "accuracy_drop"]
    frame = read_table(study, "robustness", keys + group + ["augmentation"] + metrics)
    if frame is not None:
        write(mean_std(augmented(frame), keys + group, metrics), out, "robustness.csv", written)
    return written


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    written = build(args.study, args.out)
    print("Đã ghi:", ", ".join(written) if written else "(không có bảng nào)")


if __name__ == "__main__":
    main()
