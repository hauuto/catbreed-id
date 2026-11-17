"""Merge and rank phase-A sweep results without reading held-out data."""

from __future__ import annotations

import argparse
import json
import sys
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import f1_score


FORBIDDEN_INPUT_TOKENS = ("test", "near_ood", "far_ood", "unseen")
REQUIRED_COLUMNS = {
    "sweep_point",
    "model",
    "seed",
    "stage",
    "evaluation",
    "val_macro_f1",
    "val_accuracy",
    "best_epoch",
    "size_megabytes",
    "config_overrides",
    "stage_dir",
    "run_id",
}
RANKING_COLUMNS = [
    "model",
    "rank",
    "sweep_point",
    "factor_group",
    "stage",
    "val_macro_f1",
    "val_accuracy",
    "best_epoch",
    "size_megabytes",
    "config_overrides",
    "stage_dir",
]
F1_TIE_TOLERANCE = 1e-4


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def reject_forbidden_path(path: Path) -> None:
    name = path.name.lower()
    token = next((item for item in FORBIDDEN_INPUT_TOKENS if item in name), None)
    if token:
        raise ValueError(f"Đầu vào bị cấm vì tên chứa '{token}': {path}")


def factor_group(point: str) -> str:
    """Return the varied factor, grouping numeric levels of the same factor."""
    prefix = str(point).split("_", 1)[0]
    letters = "".join(character for character in prefix if character.isalpha())
    return letters.lower() or prefix.lower()


def is_better(candidate: dict, current: dict | None) -> bool:
    """Exact QD9 rule copied from kaggle_runner.ipynb."""
    if current is None:
        return True
    delta = candidate["val_macro_f1"] - current["val_macro_f1"]
    if abs(delta) < F1_TIE_TOLERANCE:
        if candidate["val_accuracy"] != current["val_accuracy"]:
            return candidate["val_accuracy"] > current["val_accuracy"]
        return candidate["best_epoch"] < current["best_epoch"]
    return delta > 0


def compare_records(left: dict, right: dict) -> int:
    if is_better(left, right):
        return -1
    if is_better(right, left):
        return 1
    return 0


def parse_overrides(value: object, where: str) -> dict:
    if isinstance(value, dict):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"config_overrides không phải JSON hợp lệ tại {where}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"config_overrides phải là JSON object tại {where}")
    return parsed


def read_sweep_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        reject_forbidden_path(path)
        if path.name != "sweep_results.csv":
            raise ValueError(f"Chỉ nhận file có tên sweep_results.csv: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} thiếu cột: {', '.join(sorted(missing))}")
        frame = frame.copy()
        frame["_source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("Cần ít nhất một file sweep_results.csv")
    merged = pd.concat(frames, ignore_index=True)
    key = ["sweep_point", "model", "seed", "stage"]
    duplicates = merged.duplicated(key, keep=False)
    if duplicates.any():
        records = merged.loc[duplicates, key].drop_duplicates().to_dict("records")
        raise ValueError(f"Trùng khóa sweep: {records}")
    if "protocol_id" in merged:
        if merged["protocol_id"].isna().any():
            raise ValueError("protocol_id có giá trị thiếu")
        protocols = merged["protocol_id"].astype(str).unique()
        if len(protocols) != 1:
            raise ValueError(f"Các file không cùng protocol_id: {sorted(protocols)}")
    for column in ["val_macro_f1", "val_accuracy", "best_epoch", "size_megabytes"]:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
        if merged[column].isna().any():
            raise ValueError(f"Cột {column} có giá trị thiếu")
    for index, value in merged["config_overrides"].items():
        parse_overrides(value, f"dòng {index + 2}")
    return merged


def select_and_rank(merged: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (model, point), group in merged.groupby(["model", "sweep_point"], sort=False):
        best = None
        for record in group.to_dict("records"):
            if is_better(record, best):
                best = record
        best = dict(best)
        best["model"] = model
        best["sweep_point"] = point
        best["factor_group"] = factor_group(point)
        selected.append(best)

    ranked = []
    selected_frame = pd.DataFrame(selected)
    for model, group in selected_frame.groupby("model", sort=False):
        records = sorted(group.to_dict("records"), key=cmp_to_key(compare_records))
        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
            ranked.append(record)
    return pd.DataFrame(ranked).sort_values(["model", "rank"], kind="stable").reset_index(drop=True)


def criterion_order(group: pd.DataFrame, column: str, ascending: bool) -> list[str]:
    return (
        group.sort_values([column, "rank"], ascending=[ascending, True], kind="stable")
        ["sweep_point"]
        .astype(str)
        .tolist()
    )


def criterion_sensitivity(ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in ranking.groupby("model", sort=False):
        orders = {
            "val_macro_f1": group.sort_values("rank", kind="stable")["sweep_point"].astype(str).tolist(),
            "val_accuracy": criterion_order(group, "val_accuracy", ascending=False),
        }
        # Notebook ghi cross-entropy validation vào cột val_loss; không dùng label smoothing nên val_loss = NLL
        nll_column = "val_nll" if "val_nll" in group else "val_loss"
        nll_available = nll_column in group and group[nll_column].notna().all()
        if nll_available:
            orders["val_nll"] = criterion_order(group, nll_column, ascending=True)
        criteria = ["val_macro_f1", "val_accuracy", "val_nll"]
        for left_index, left in enumerate(criteria):
            for right in criteria[left_index + 1 :]:
                available = left in orders and right in orders
                if available:
                    points = orders[left]
                    left_ranks = {point: rank for rank, point in enumerate(points, start=1)}
                    right_ranks = {point: rank for rank, point in enumerate(orders[right], start=1)}
                    tau, pvalue = kendalltau(
                        [left_ranks[point] for point in points],
                        [right_ranks[point] for point in points],
                    )
                    top_left, top_right = orders[left][0], orders[right][0]
                    note = ""
                else:
                    tau = pvalue = float("nan")
                    top_left = orders[left][0] if left in orders else ""
                    top_right = orders[right][0] if right in orders else ""
                    note = "val_nll/val_loss không có trong sweep_results.csv"
                rows.append(
                    {
                        "model": model,
                        "criterion_a": left,
                        "criterion_b": right,
                        "top1_a": top_left,
                        "top1_b": top_right,
                        "top1_preserved": bool(available and top_left == top_right),
                        "kendall_tau": tau,
                        "kendall_pvalue": pvalue,
                        "available": available,
                        "note": note,
                    }
                )
    return pd.DataFrame(rows)


def validation_count(n_val: int | None, calibration: Path | None) -> int:
    values = []
    if n_val is not None:
        values.append(int(n_val))
    if calibration is not None:
        reject_forbidden_path(calibration)
        if calibration.name != "calibration.json":
            raise ValueError("File calibration phải có tên calibration.json")
        payload = json.loads(calibration.read_text(encoding="utf-8"))
        if "n_images" not in payload:
            raise ValueError("calibration.json thiếu n_images")
        values.append(int(payload["n_images"]))
    if not values:
        raise ValueError("Cần --n-val hoặc --calibration")
    if any(value <= 0 for value in values):
        raise ValueError("Số ảnh validation phải dương")
    if len(set(values)) != 1:
        raise ValueError(f"--n-val không khớp calibration.json: {values}")
    return values[0]


def build_outputs(merged: pd.DataFrame, n_val: int) -> tuple[pd.DataFrame, ...]:
    ranking = select_and_rank(merged)
    step = 1.0 / n_val
    ranking["accuracy_resolution"] = step
    ranking["top12_accuracy_indistinguishable"] = False
    for _, indices in ranking.groupby("model", sort=False).groups.items():
        top = ranking.loc[list(indices)].sort_values("rank").head(2)
        if len(top) == 2 and abs(top.iloc[0].val_accuracy - top.iloc[1].val_accuracy) < step:
            ranking.loc[top.index, "top12_accuracy_indistinguishable"] = True
    top = ranking[ranking["rank"] <= 3].copy()
    best_factor = (
        ranking.sort_values(["model", "factor_group", "rank"], kind="stable")
        .groupby(["model", "factor_group"], sort=False, as_index=False)
        .first()
    )
    sensitivity = criterion_sensitivity(ranking)
    return ranking, top, best_factor, sensitivity


def print_summary(ranking: pd.DataFrame, n_val: int) -> None:
    print(f"Bước accuracy của một ảnh: 1/{n_val} = {1 / n_val:.8f}")
    print("Nhóm yếu tố:", ", ".join(sorted(ranking.factor_group.unique())))
    columns = ["model", "rank", "sweep_point", "stage", "val_macro_f1", "val_accuracy"]
    print("\nTop 3 mỗi mô hình:")
    print(ranking.loc[ranking["rank"] <= 3, columns].to_string(index=False))
    for model, group in ranking.groupby("model", sort=False):
        first_two = group.sort_values("rank").head(2)
        if len(first_two) < 2:
            print(f"{model}: không đủ hai điểm để tính khoảng cách F1")
            continue
        gap = float(first_two.iloc[0].val_macro_f1 - first_two.iloc[1].val_macro_f1)
        note = " — chênh lệch nhỏ, cần seed 43–44 để kết luận" if gap < 0.005 else ""
        print(f"{model}: khoảng cách F1 hạng 1–2 = {gap:.6f}{note}")


CROSS_CHECK_TOLERANCE = 1e-9
N_CLASSES = 12


def validation_metrics(npz_path: Path, exclude: set[str]) -> tuple[float, float]:
    """Accuracy và F1 macro từ val.npz, bỏ các image_id trong exclude; cùng công thức với notebook."""
    reject_forbidden_path(npz_path)
    if npz_path.name != "val.npz":
        raise ValueError(f"Chỉ đọc val.npz: {npz_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Không thấy {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        logits, labels, image_ids = data["logits"], data["labels"], data["image_id"].astype(str)
    missing = sorted(exclude - set(image_ids))
    if missing:
        raise ValueError(f"image_id không có trong {npz_path}: {missing}")
    keep = ~np.isin(image_ids, sorted(exclude))
    predictions = logits[keep].argmax(1)
    labels = labels[keep]
    accuracy = float(np.mean(labels == predictions))
    macro_f1 = float(f1_score(labels, predictions, labels=range(N_CLASSES), average="macro", zero_division=0))
    return accuracy, macro_f1


def exclude_val_sensitivity(ranking: pd.DataFrame, runs_root: Path, exclude: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phân tích độ nhạy: tính lại F1/accuracy validation khi bỏ một số ảnh, xếp hạng lại trong từng mô hình.

    Mỗi dòng của bảng xếp hạng giữ nguyên checkpoint (stage) đã chọn; chỉ chỉ số validation được tính lại.
    Không đổi quy tắc chọn chính thức.
    """
    excluded = set(exclude)
    if not excluded:
        raise ValueError("--exclude-val cần ít nhất một image_id")
    rows = []
    for record in ranking.to_dict("records"):
        path = runs_root / str(record["stage_dir"]) / "val.npz"
        acc_full, f1_full = validation_metrics(path, set())
        for name, recomputed, recorded in [("val_accuracy", acc_full, record["val_accuracy"]),
                                           ("val_macro_f1", f1_full, record["val_macro_f1"])]:
            if abs(recomputed - float(recorded)) >= CROSS_CHECK_TOLERANCE:
                raise ValueError(
                    f"Kiểm chéo thất bại ({record['model']}, {record['sweep_point']}): {name} tính lại {recomputed!r} "
                    f"≠ sweep_results.csv {recorded!r}"
                )
        acc_excluded, f1_excluded = validation_metrics(path, excluded)
        rows.append({
            "model": record["model"], "sweep_point": record["sweep_point"], "stage": record["stage"],
            "rank_full": int(record["rank"]), "f1_full": f1_full, "f1_excluded": f1_excluded,
            "acc_full": acc_full, "acc_excluded": acc_excluded, "best_epoch": record["best_epoch"],
        })
    table = pd.DataFrame(rows)
    table["rank_excluded"] = 0
    summary = []
    for model, group in table.groupby("model", sort=False):
        candidates = [
            {"index": index, "val_macro_f1": row.f1_excluded, "val_accuracy": row.acc_excluded, "best_epoch": row.best_epoch}
            for index, row in group.iterrows()
        ]
        for rank, candidate in enumerate(sorted(candidates, key=cmp_to_key(compare_records)), start=1):
            table.loc[candidate["index"], "rank_excluded"] = rank
        group = table.loc[group.index]
        tau = kendalltau(group.rank_full, group.rank_excluded)[0] if len(group) > 1 else float("nan")
        top_full = group.loc[group.rank_full == 1, "sweep_point"].iloc[0]
        top_excluded = group.loc[group.rank_excluded == 1, "sweep_point"].iloc[0]
        summary.append({"model": model, "n_points": len(group), "kendall_tau": tau, "top1_full": top_full,
                        "top1_excluded": top_excluded, "top1_preserved": top_full == top_excluded})
    columns = ["model", "sweep_point", "stage", "rank_full", "rank_excluded", "f1_full", "f1_excluded", "acc_full", "acc_excluded"]
    return table.sort_values(["model", "rank_full"], kind="stable")[columns].reset_index(drop=True), pd.DataFrame(summary)


def run(
    inputs: list[Path],
    output_dir: Path,
    n_val: int | None,
    calibration: Path | None,
    exclude_val: list[str] | None = None,
    runs_root: Path | None = None,
) -> None:
    merged = read_sweep_files(inputs)
    count = validation_count(n_val, calibration)
    ranking, top, best_factor, sensitivity = build_outputs(merged, count)
    if exclude_val is not None:
        if runs_root is None:
            raise ValueError("--exclude-val cần --runs-root")
        table, summary = exclude_val_sensitivity(ranking, runs_root, exclude_val)
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "stage1_exclude_val.csv", index=False)
        summary.to_csv(output_dir / "stage1_exclude_val_summary.csv", index=False)
        print(f"Độ nhạy khi bỏ {len(set(exclude_val))} ảnh validation: {', '.join(sorted(set(exclude_val)))}")
        print(summary.to_string(index=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking[RANKING_COLUMNS + ["accuracy_resolution", "top12_accuracy_indistinguishable"]].to_csv(
        output_dir / "stage1_ranking.csv", index=False
    )
    top[RANKING_COLUMNS + ["accuracy_resolution", "top12_accuracy_indistinguishable"]].to_csv(
        output_dir / "stage1_top.csv", index=False
    )
    best_columns = ["model", "factor_group", "sweep_point", "rank"] + RANKING_COLUMNS[4:]
    best_factor[best_columns].to_csv(output_dir / "stage1_best_per_factor.csv", index=False)
    sensitivity.to_csv(output_dir / "stage1_criterion_sensitivity.csv", index=False)
    print_summary(ranking, count)


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Một hoặc nhiều file sweep_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-val", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--exclude-val",
        type=lambda text: [item.strip() for item in text.split(",") if item.strip()],
        help="Phân tích độ nhạy: các image_id validation bỏ ra, phân cách bằng dấu phẩy",
    )
    parser.add_argument("--runs-root", type=Path, help="Thư mục chứa các run; ghép với cột stage_dir để tìm val.npz")
    args = parser.parse_args()
    run(args.inputs, args.output_dir, args.n_val, args.calibration, args.exclude_val, args.runs_root)


if __name__ == "__main__":
    main()
