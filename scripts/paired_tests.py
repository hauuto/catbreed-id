"""Run paired McNemar and bootstrap comparisons on locked ID-test predictions."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score


OUTPUT_COLUMNS = [
    "a", "b", "n", "acc_a", "acc_b", "delta_acc", "ci_low", "ci_high",
    "f1_a", "f1_b", "delta_f1", "ci_low_f1", "ci_high_f1",
    "b_only_a", "c_only_b", "mcnemar_p", "holm_p",
]


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_stage(stage_dir: Path) -> dict[str, np.ndarray]:
    path = stage_dir / "test.npz"
    if not path.is_file():
        raise ValueError(f"Thiếu test.npz: {path}")
    with np.load(path, allow_pickle=False) as stored:
        required = {"logits", "labels", "image_id"}
        if missing := required - set(stored.files):
            raise ValueError(f"{path} thiếu mảng: {', '.join(sorted(missing))}")
        arrays = {name: stored[name] for name in required}
    n = len(arrays["image_id"])
    if len(arrays["labels"]) != n or len(arrays["logits"]) != n:
        raise ValueError(f"Số dòng không đồng nhất trong {path}")
    ids = arrays["image_id"].astype(str)
    if len(set(ids.tolist())) != n:
        raise ValueError(f"image_id bị trùng trong {path}")
    if arrays["logits"].ndim != 2 or arrays["logits"].shape[1] != 12:
        raise ValueError(f"logits phải có dạng (n, 12): {path}")
    arrays["image_id"] = ids
    arrays["labels"] = np.asarray(arrays["labels"], dtype=np.int64)
    if not set(arrays["labels"].tolist()) <= set(range(12)):
        raise ValueError(f"labels phải nằm trong 0..11: {path}")
    return arrays


def stage_labels(stage_dirs: list[Path]) -> dict[str, Path]:
    names = [path.name for path in stage_dirs]
    if len(set(names)) != len(names):
        names = [f"{path.parent.name}/{path.name}" for path in stage_dirs]
    if len(set(names)) != len(names):
        raise ValueError("Không tạo được tên duy nhất cho các stage directory")
    return dict(zip(names, stage_dirs))


def align(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_a, ids_b = a["image_id"], b["image_id"]
    if set(ids_a.tolist()) != set(ids_b.tolist()):
        missing_a = sorted(set(ids_b) - set(ids_a))[:5]
        missing_b = sorted(set(ids_a) - set(ids_b))[:5]
        raise ValueError(f"Tập image_id khác nhau; thiếu ở A={missing_a}, thiếu ở B={missing_b}")
    order_b = {image_id: index for index, image_id in enumerate(ids_b)}
    indices_b = np.array([order_b[image_id] for image_id in ids_a])
    labels_a = a["labels"]
    labels_b = b["labels"][indices_b]
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("labels không khớp sau khi ghép theo image_id")
    pred_a = np.asarray(a["logits"]).argmax(axis=1)
    pred_b = np.asarray(b["logits"])[indices_b].argmax(axis=1)
    return labels_a, pred_a, pred_b


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", labels=range(12), zero_division=0))


def compare_pair(
    name_a: str,
    arrays_a: dict[str, np.ndarray],
    name_b: str,
    arrays_b: dict[str, np.ndarray],
    repeats: int = 1000,
    seed: int = 42,
) -> dict:
    if repeats < 1:
        raise ValueError("Số lần bootstrap phải dương")
    labels, pred_a, pred_b = align(arrays_a, arrays_b)
    correct_a, correct_b = pred_a == labels, pred_b == labels
    b_only_a = int(np.sum(correct_a & ~correct_b))
    c_only_b = int(np.sum(~correct_a & correct_b))
    discordant = b_only_a + c_only_b
    pvalue = 1.0 if discordant == 0 else float(binomtest(b_only_a, discordant, 0.5).pvalue)
    acc_a, acc_b = float(correct_a.mean()), float(correct_b.mean())
    f1_a, f1_b = macro_f1(labels, pred_a), macro_f1(labels, pred_b)

    rng = np.random.default_rng(seed)
    delta_accuracy = np.empty(repeats, dtype=np.float64)
    delta_f1 = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(labels), size=len(labels))
        delta_accuracy[index] = np.mean(pred_a[sample] == labels[sample]) - np.mean(pred_b[sample] == labels[sample])
        delta_f1[index] = macro_f1(labels[sample], pred_a[sample]) - macro_f1(labels[sample], pred_b[sample])
    low_acc, high_acc = np.quantile(delta_accuracy, [0.025, 0.975], method="linear")
    low_f1, high_f1 = np.quantile(delta_f1, [0.025, 0.975], method="linear")
    return {
        "a": name_a,
        "b": name_b,
        "n": len(labels),
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta_acc": acc_a - acc_b,
        "ci_low": float(low_acc),
        "ci_high": float(high_acc),
        "f1_a": f1_a,
        "f1_b": f1_b,
        "delta_f1": f1_a - f1_b,
        "ci_low_f1": float(low_f1),
        "ci_high_f1": float(high_f1),
        "b_only_a": b_only_a,
        "c_only_b": c_only_b,
        "mcnemar_p": pvalue,
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(pvalues[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def choose_pairs(names: list[str], specification: str | None) -> list[tuple[str, str]]:
    if not specification:
        return list(combinations(names, 2))
    pairs = []
    for item in specification.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Cặp không hợp lệ: {item!r}")
        if parts[0] not in names or parts[1] not in names:
            raise ValueError(f"Tên trong --pairs không tồn tại: {item!r}; có {names}")
        if parts[0] == parts[1]:
            raise ValueError("Không so sánh một stage với chính nó")
        pairs.append((parts[0], parts[1]))
    return pairs


def run(stage_dirs: list[Path], output: Path, pair_spec: str | None = None) -> pd.DataFrame:
    if len(stage_dirs) < 2:
        raise ValueError("Cần ít nhất hai stage directory")
    labels = stage_labels(stage_dirs)
    arrays = {name: load_stage(path) for name, path in labels.items()}
    pairs = choose_pairs(list(labels), pair_spec)
    rows = [compare_pair(a, arrays[a], b, arrays[b]) for a, b in pairs]
    adjusted = holm_adjust([row["mcnemar_p"] for row in rows])
    for row, holm_p in zip(rows, adjusted):
        row["holm_p"] = holm_p
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print("holm_p: p-value McNemar đã hiệu chỉnh Holm qua các cặp trong bảng")
    return frame


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dirs", nargs="+", type=Path)
    parser.add_argument("--pairs", help="Danh sách A:B,C:D; mặc định so mọi cặp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage_dirs, args.output, args.pairs)


if __name__ == "__main__":
    main()
