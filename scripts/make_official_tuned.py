"""Sinh configs/v2/official_tuned.json từ kết quả sweep giai đoạn 1 và 2 (chỉ validation).

Quy tắc đã đăng ký trước:
- Ứng viên = một (model, config_overrides) có đủ 3 seed 42, 43, 44 trong các sweep_results.csv.
- Với mỗi seed: chọn checkpoint tốt nhất giữa frozen/finetuned theo is_better (QĐ9).
- Chọn ứng viên có F1 macro validation trung bình 3 seed cao nhất; chênh < 1e-4 thì Accuracy trung bình
  cao hơn; vẫn hòa thì giữ ứng viên có overrides ít khóa hơn (đơn giản hơn), rồi theo thứ tự chuỗi JSON.
- Chênh Accuracy trung bình giữa hạng 1 và hạng 2 nhỏ hơn 1/n_val được cảnh báo (không đổi quyết định).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_sweep import F1_TIE_TOLERANCE, is_better, read_sweep_files  # noqa: E402

SEEDS = (42, 43, 44)
REQUIRED_MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "cnn_baseline")


def canonical(value) -> str:
    data = value if isinstance(value, dict) else json.loads(str(value))
    return json.dumps(data, sort_keys=True)


def best_per_seed(rows: pd.DataFrame) -> dict | None:
    best = None
    for record in rows.to_dict("records"):
        if is_better(record, best):
            best = record
    return best


def candidates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["overrides_key"] = frame["config_overrides"].map(canonical)
    out = []
    for (model, key), group in frame.groupby(["model", "overrides_key"], sort=True):
        per_seed = {}
        for seed in SEEDS:
            rows = group[group["seed"] == seed]
            if len(rows):
                per_seed[seed] = best_per_seed(rows)
        if set(per_seed) != set(SEEDS):
            continue
        f1 = [per_seed[s]["val_macro_f1"] for s in SEEDS]
        acc = [per_seed[s]["val_accuracy"] for s in SEEDS]
        out.append(
            {
                "model": model,
                "overrides": key,
                "n_keys": len(json.loads(key)),
                "f1_mean": float(pd.Series(f1).mean()),
                "f1_std": float(pd.Series(f1).std(ddof=1)),
                "acc_mean": float(pd.Series(acc).mean()),
                "acc_std": float(pd.Series(acc).std(ddof=1)),
                "stages": ",".join(f"{s}:{per_seed[s]['stage']}" for s in SEEDS),
                "sweep_points": ",".join(sorted({str(p) for p in group["sweep_point"]})),
            }
        )
    return pd.DataFrame(out)


def rank(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in table.groupby("model", sort=True):
        records = group.to_dict("records")

        def better(a, b):
            if abs(a["f1_mean"] - b["f1_mean"]) >= F1_TIE_TOLERANCE:
                return a["f1_mean"] > b["f1_mean"]
            if a["acc_mean"] != b["acc_mean"]:
                return a["acc_mean"] > b["acc_mean"]
            if a["n_keys"] != b["n_keys"]:
                return a["n_keys"] < b["n_keys"]
            return a["overrides"] < b["overrides"]

        ordered = []
        for record in records:
            index = 0
            while index < len(ordered) and not better(record, ordered[index]):
                index += 1
            ordered.insert(index, record)
        for position, record in enumerate(ordered, start=1):
            rows.append({**record, "rank": position})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_results", nargs="+", type=Path, help="sweep_results.csv của giai đoạn 1 và 2")
    parser.add_argument("--base-config", type=Path, required=True, help="configs/v2/official_default.json")
    parser.add_argument("--output", type=Path, required=True, help="configs/v2/official_tuned.json")
    parser.add_argument("--table", type=Path, required=True, help="CSV bảng ứng viên đã xếp hạng")
    parser.add_argument("--n-val", type=int, default=238)
    args = parser.parse_args(argv)

    frame = read_sweep_files(args.sweep_results)
    table = rank(candidates(frame))
    if table.empty:
        raise SystemExit("Không có ứng viên nào đủ 3 seed 42, 43, 44.")
    missing = [m for m in REQUIRED_MODELS if m not in set(table["model"])]
    if missing:
        raise SystemExit("Thiếu ứng viên đủ 3 seed cho: " + ", ".join(missing))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    overrides, warnings = {}, []
    for model, group in table.groupby("model", sort=True):
        group = group.sort_values("rank")
        top = group.iloc[0]
        overrides[model] = json.loads(top["overrides"])
        if len(group) > 1:
            second = group.iloc[1]
            if abs(top["acc_mean"] - second["acc_mean"]) < 1 / args.n_val:
                warnings.append(f"{model}: hạng 1 và 2 chênh Accuracy trung bình < 1/{args.n_val}")
        print(f"{model}: F1 {top['f1_mean']:.4f} ± {top['f1_std']:.4f}, Acc {top['acc_mean']:.4f}, overrides {top['overrides']}")
    config = {**base, "owner": "VPS_tuned", "model_overrides": overrides}
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for warning in warnings:
        print("CẢNH BÁO:", warning)
    print("Đã ghi", args.output, "và", args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
