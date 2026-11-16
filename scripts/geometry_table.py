"""Summarize center-crop versus letterbox validation sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
META_COLUMNS = ("name", "model", "seed", "stage", "augmentation", "run_id", "sweep_point")
FORBIDDEN_TOKENS = ("test", "near_ood", "far_ood", "unseen")


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def validate_path(path: Path) -> None:
    logical = PurePosixPath(str(path).replace("\\", "/").lower())
    parts = logical.parts
    anchors = [index for index, part in enumerate(parts) if part in {"runs", "outputs"}]
    scoped = "/".join(parts[anchors[-1] :]) if anchors else "/".join(parts[-3:])
    token = next((item for item in FORBIDDEN_TOKENS if item in scoped), None)
    if token:
        raise ValueError(f"Đường dẫn chứa từ cấm '{token}': {path}")
    if logical.name != "eval_geometry.csv":
        raise ValueError(f"Chỉ nhận eval_geometry.csv: {path}")


def discover(inputs: list[Path]) -> list[Path]:
    found = []
    for item in inputs:
        if item.is_dir():
            found.extend(item.rglob("eval_geometry.csv"))
        elif item.is_file():
            found.append(item)
        else:
            raise ValueError(f"Đầu vào không tồn tại: {item}")
    paths = sorted(set(path.resolve() for path in found))
    if not paths:
        raise ValueError("Không tìm thấy eval_geometry.csv")
    for path in paths:
        validate_path(path)
    return paths


def read_geometry(paths: list[Path]) -> pd.DataFrame:
    frames = []
    required = set(META_COLUMNS) | {
        f"{view}_{metric}" for view in ("center", "letterbox") for metric in METRICS
    }
    for path in paths:
        frame = pd.read_csv(path)
        if missing := required - set(frame):
            raise ValueError(f"{path} thiếu cột: {', '.join(sorted(missing))}")
        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    duplicate_key = ["name", "model", "seed", "stage", "run_id"]
    if merged.duplicated(duplicate_key).any():
        raise ValueError("Có checkpoint eval_geometry bị trùng")
    numeric = [f"{view}_{metric}" for view in ("center", "letterbox") for metric in METRICS]
    for column in numeric:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
        if merged[column].isna().any():
            raise ValueError(f"Cột {column} có giá trị thiếu")
    for metric in ("accuracy", "macro_f1"):
        merged[f"delta_{metric}"] = merged[f"letterbox_{metric}"] - merged[f"center_{metric}"]
    return merged


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, stage), group in frame.groupby(["model", "stage"], sort=True):
        row = {"model": model, "stage": stage, "n_checkpoints": len(group)}
        for metric in ("accuracy", "macro_f1"):
            for column in (f"center_{metric}", f"letterbox_{metric}", f"delta_{metric}"):
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            delta = group[f"delta_{metric}"].to_numpy(float)
            equal = np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)
            row[f"letterbox_better_{metric}"] = int(np.sum((delta > 0) & ~equal))
            row[f"letterbox_equal_{metric}"] = int(np.sum(equal))
            row[f"letterbox_worse_{metric}"] = int(np.sum((delta < 0) & ~equal))
        rows.append(row)
    return pd.DataFrame(rows)


def run(inputs: list[Path], output: Path) -> pd.DataFrame:
    paths = discover(inputs)
    summary = summarize(read_geometry(paths))
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)


if __name__ == "__main__":
    main()
