"""Plot ID/OOD score distributions for the five registered rejection methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN")
SCORE_FILES = {
    "ID test": "test_scores.csv",
    "Near-OOD": "near_ood_scores.csv",
    "Far-OOD": "far_ood_scores.csv",
    "Unseen cat": "unseen_cat_scores.csv",
}
COLORS = {
    "ID test": "#2563EB",
    "Near-OOD": "#F97316",
    "Far-OOD": "#0D9488",
    "Unseen cat": "#A855F7",
}


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def read_inputs(stage_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    if not stage_dir.is_dir():
        raise ValueError(f"Stage directory không tồn tại: {stage_dir}")
    sets = {}
    required = {"image_id", *METHODS}
    for label, filename in SCORE_FILES.items():
        path = stage_dir / filename
        if not path.is_file():
            if label == "ID test":
                raise ValueError(f"Thiếu file bắt buộc: {path}")
            continue
        frame = pd.read_csv(path)
        if missing := required - set(frame):
            raise ValueError(f"{path} thiếu cột: {', '.join(sorted(missing))}")
        for method in METHODS:
            frame[method] = pd.to_numeric(frame[method], errors="raise")
            if frame[method].isna().any():
                raise ValueError(f"{path}: cột {method} có giá trị thiếu")
        sets[label] = frame
    if len(sets) < 2:
        raise ValueError("Cần ID test và ít nhất một nhóm OOD")
    thresholds_path = stage_dir / "thresholds.json"
    if not thresholds_path.is_file():
        raise ValueError(f"Thiếu file bắt buộc: {thresholds_path}")
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds = payload.get("T1")
    if not isinstance(thresholds, dict) or not set(METHODS) <= set(thresholds):
        raise ValueError("thresholds.json thiếu T1 của một hoặc nhiều phương pháp")
    return sets, {method: float(thresholds[method]) for method in METHODS}


def needs_symlog(values: np.ndarray) -> bool:
    absolute = np.abs(values[np.nonzero(values)])
    return bool(len(absolute) and absolute.max() / absolute.min() > 1000)


def summarize(sets: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        all_values = np.concatenate([frame[method].to_numpy(float) for frame in sets.values()])
        scale = "symlog" if method == "Mahalanobis" and needs_symlog(all_values) else "linear"
        for group, frame in sets.items():
            values = frame[method].to_numpy(float)
            rows.append(
                {
                    "method": method,
                    "group": group,
                    "n": len(values),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05, method="linear")),
                    "q95": float(np.quantile(values, 0.95, method="linear")),
                    "threshold": thresholds[method],
                    "acceptance_rate": float(np.mean(values >= thresholds[method])),
                    "xscale": scale,
                }
            )
    return pd.DataFrame(rows)


def plot_histograms(
    sets: dict[str, pd.DataFrame], thresholds: dict[str, float], output: Path, bins: int = 30
) -> None:
    if bins < 2:
        raise ValueError("bins phải >= 2")
    with plt.rc_context({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"}):
        figure, axes = plt.subplots(1, len(METHODS), figsize=(18, 4.2), squeeze=False)
        axes = axes[0]
        legend_handles = None
        legend_labels = None
        for axis, method in zip(axes, METHODS):
            all_values = np.concatenate([frame[method].to_numpy(float) for frame in sets.values()])
            edges = np.histogram_bin_edges(all_values, bins=bins)
            for group, frame in sets.items():
                axis.hist(
                    frame[method].to_numpy(float),
                    bins=edges,
                    density=True,
                    alpha=0.34,
                    color=COLORS[group],
                    label=group,
                )
            axis.axvline(thresholds[method], color="black", linestyle="--", linewidth=1.4, label="τ")
            if method == "Mahalanobis" and needs_symlog(all_values):
                nonzero = np.abs(all_values[np.nonzero(all_values)])
                linthresh = max(float(np.quantile(nonzero, 0.05)), 1e-12)
                axis.set_xscale("symlog", linthresh=linthresh)
                print(f"Mahalanobis dùng symlog (linthresh={linthresh:.6g})")
            axis.set_xlabel(method)
            axis.set_ylabel("Mật độ" if axis is axes[0] else "")
            axis.grid(axis="y", alpha=0.2)
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
        figure.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)


def run(stage_dir: Path, output_dir: Path, bins: int = 30) -> tuple[Path, Path]:
    sets, thresholds = read_inputs(stage_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = output_dir / "score_histograms.png"
    table = output_dir / "score_summary.csv"
    plot_histograms(sets, thresholds, image, bins)
    summary = summarize(sets, thresholds)
    summary.to_csv(table, index=False)
    print(summary.to_string(index=False))
    print("Đã lưu:", image, "và", table)
    return image, table


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()
    run(args.stage_dir, args.output_dir, args.bins)


if __name__ == "__main__":
    main()
