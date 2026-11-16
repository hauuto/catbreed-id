"""Measure phase-A KNN rejection sensitivity for pre-registered k values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from sklearn.neighbors import NearestNeighbors


K_VALUES = (1, 5, 10, 20, 50)
FORBIDDEN_INPUT_TOKENS = ("test", "near_ood", "far_ood", "unseen")
OUTPUT_COLUMNS = [
    "stage_dir",
    "model",
    "seed",
    "stage",
    "k",
    "tau",
    "auroc",
    "fpr_at_95_tpr",
    "tpr_at_threshold",
    "fpr_at_threshold",
    "n_ood_accepted",
]


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def reject_forbidden_name(path: Path) -> None:
    name = path.name.lower()
    token = next((item for item in FORBIDDEN_INPUT_TOKENS if item in name), None)
    if token:
        raise ValueError(f"Đầu vào bị cấm vì tên chứa '{token}': {path}")


def unit_norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_scores(bank_features, query_features, k):
    bank = unit_norm(np.asarray(bank_features, dtype=np.float64))
    query = unit_norm(np.asarray(query_features, dtype=np.float64))
    if len(bank) < k:
        raise ValueError(f"Train bank có {len(bank)} mẫu, nhỏ hơn k={k}; không tự đổi k")
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(bank)
    distances = neighbors.kneighbors(query, return_distance=True)[0]
    return -distances[:, k - 1]


def auroc(id_scores, ood_scores):
    # Exact Mann-Whitney implementation from kaggle_runner.ipynb.
    ranks = rankdata(np.concatenate([id_scores, ood_scores]), axis=0)
    n_id, n_ood = len(id_scores), len(ood_scores)
    return (ranks[:n_id].sum(axis=0) - n_id * (n_id + 1) / 2) / (n_id * n_ood)


def fpr_at_95_tpr(id_scores, ood_scores):
    labels = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    fpr, tpr, _ = roc_curve(labels, np.r_[id_scores, ood_scores], drop_intermediate=False)
    first = int(np.argmax(tpr >= 0.95))
    return float(np.interp(0.95, tpr[first - 1 : first + 1], fpr[first - 1 : first + 1]))


def ood_metrics(id_scores, ood_scores, threshold):
    return {
        "auroc": float(auroc(id_scores, ood_scores)),
        "fpr_at_95_tpr": fpr_at_95_tpr(id_scores, ood_scores),
        "threshold": float(threshold),
        "tpr_at_threshold": float(np.mean(id_scores >= threshold)),
        "fpr_at_threshold": float(np.mean(ood_scores >= threshold)),
        "n_ood_accepted": int(np.sum(ood_scores >= threshold)),
        "n_id": len(id_scores),
        "n_ood": len(ood_scores),
    }


def read_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    reject_forbidden_name(path)
    with np.load(path, allow_pickle=False) as stored:
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"{path} thiếu mảng: {', '.join(sorted(missing))}")
        return {name: stored[name] for name in stored.files}


def read_metadata(stage_dir: Path) -> tuple[str, int, str]:
    info_path = stage_dir.parent / "info.json"
    reject_forbidden_name(info_path)
    if not info_path.is_file():
        raise ValueError(f"Thiếu info.json cạnh stage directory: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    model = info.get("model")
    seed = info.get("training_seed", info.get("seed"))
    stage = stage_dir.name
    if not model or seed is None or not stage:
        raise ValueError(f"info.json thiếu model/training_seed hoặc stage directory không hợp lệ: {stage_dir}")
    return str(model), int(seed), stage


def align_reference_scores(validation: dict, frame: pd.DataFrame, path: Path) -> np.ndarray:
    if "KNN" not in frame:
        raise ValueError(f"{path} thiếu cột KNN")
    if len(frame) != len(validation["features"]):
        raise ValueError(f"{path} không cùng số dòng với val.npz")
    if "image_id" in frame and "image_id" in validation:
        expected = np.asarray(validation["image_id"]).astype(str)
        actual = frame["image_id"].astype(str).to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError(f"Thứ tự image_id không khớp giữa {path.name} và val.npz")
    return pd.to_numeric(frame["KNN"], errors="raise").to_numpy(np.float64)


def check_k10(
    stage_dir: Path,
    validation: dict,
    computed_val: np.ndarray,
    computed_metrics: dict,
) -> None:
    scores_path = stage_dir / "validation_scores.csv"
    results_path = stage_dir / "probe_results.csv"
    for path in [scores_path, results_path]:
        reject_forbidden_name(path)
        if not path.is_file():
            raise ValueError(f"Thiếu file kiểm tra chéo: {path}")
    reference_val = align_reference_scores(validation, pd.read_csv(scores_path), scores_path)
    if not np.allclose(computed_val, reference_val, rtol=0.0, atol=1e-6):
        difference = float(np.max(np.abs(computed_val - reference_val)))
        raise ValueError(f"Điểm validation KNN k=10 lệch file gốc; max_abs_diff={difference:.3g}")
    results = pd.read_csv(results_path)
    if not {"method", "fpr_at_threshold"} <= set(results):
        raise ValueError(f"{results_path} thiếu method hoặc fpr_at_threshold")
    knn = results[results["method"] == "KNN"]
    if len(knn) != 1:
        raise ValueError(f"{results_path} phải có đúng một dòng method=KNN")
    expected_fpr = float(knn.iloc[0]["fpr_at_threshold"])
    if not np.isclose(computed_metrics["fpr_at_threshold"], expected_fpr, rtol=0.0, atol=1e-12):
        raise ValueError(
            "fpr_at_threshold KNN k=10 không khớp probe_results.csv: "
            f"{computed_metrics['fpr_at_threshold']} != {expected_fpr}"
        )


def analyze_stage(stage_dir: Path, k_values=K_VALUES) -> pd.DataFrame:
    reject_forbidden_name(stage_dir)
    if not stage_dir.is_dir():
        raise ValueError(f"Stage directory không tồn tại: {stage_dir}")
    train = read_npz(stage_dir / "train.npz", {"features", "labels"})
    validation = read_npz(stage_dir / "val.npz", {"features"})
    probe = read_npz(stage_dir / "probe_ood.npz", {"features"})
    labels = np.asarray(train["labels"])
    if set(labels.tolist()) != set(range(12)):
        raise ValueError("Train bank phải có đủ 12 lớp")
    model, seed, stage = read_metadata(stage_dir)
    rows = []
    for k in k_values:
        val_scores = knn_scores(train["features"], validation["features"], int(k))
        probe_scores = knn_scores(train["features"], probe["features"], int(k))
        tau = float(np.quantile(val_scores, 0.05, method="linear"))
        metrics = ood_metrics(val_scores, probe_scores, tau)
        if int(k) == 10:
            check_k10(stage_dir, validation, val_scores, metrics)
        rows.append(
            {
                "stage_dir": str(stage_dir),
                "model": model,
                "seed": seed,
                "stage": stage,
                "k": int(k),
                "tau": tau,
                **{name: metrics[name] for name in OUTPUT_COLUMNS[6:]},
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def print_summary(frame: pd.DataFrame) -> None:
    metrics = ["tau", "auroc", "fpr_at_95_tpr", "tpr_at_threshold", "fpr_at_threshold", "n_ood_accepted"]
    if frame["stage_dir"].nunique() > 1:
        summary = frame.groupby("k")[metrics].agg(["mean", "std"])
        print("Trung bình ± độ lệch chuẩn theo k:")
        for k, row in summary.iterrows():
            values = []
            for metric in metrics:
                mean, std = row[(metric, "mean")], row[(metric, "std")]
                values.append(f"{metric}={mean:.6f} ± {std:.6f}")
            print(f"k={k}: " + "; ".join(values))
    else:
        print(frame.to_string(index=False))


def run(stage_dirs: list[Path], output: Path) -> pd.DataFrame:
    if not stage_dirs:
        raise ValueError("Cần ít nhất một stage directory")
    frame = pd.concat([analyze_stage(path) for path in stage_dirs], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print_summary(frame)
    return frame


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage_dirs, args.output)


if __name__ == "__main__":
    main()
