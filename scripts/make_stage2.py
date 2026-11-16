"""Generate validated stage-2 sweep configurations from phase-A rankings."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd


FORBIDDEN_INPUT_TOKENS = ("test", "near_ood", "far_ood", "unseen")
MODEL_NAMES = {"resnet50", "mobilenet_v2", "efficientnet_b0", "cnn_baseline"}
TRANSFER_MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b0"]
LABELS = {
    "resnet50": "A",
    "mobilenet_v2": "B",
    "efficientnet_b0": "C",
    "cnn_baseline": "Baseline",
}
CONFIG_KEYS = {
    "profile", "sweep_grid", "model_overrides", "baseline_batchnorm", "head_hidden",
    "fine_tune_fraction", "train_fraction", "train_fraction_seed", "protocol_id", "models",
    "training_seeds", "run_augmentation_ablation", "owner", "data_input_root", "far_ood_root",
    "official_archives_root", "official_data_dir", "unseen_cat_root", "unseen_cat_classes",
    "output_root", "clean_kaggle_output", "image_size", "resize_size", "batch_size", "workers",
    "io_threads", "pretrained", "frozen_epochs", "finetuned_epochs", "baseline_epochs",
    "frozen_lr", "head_lr", "backbone_lr", "baseline_lr", "weight_decay", "dropout",
    "early_stopping_patience", "scheduler_patience", "fine_tune_blocks", "knn_k", "ece_bins",
    "bootstrap_repeats", "bootstrap_seed", "deployment_max_megabytes", "progress_interval_seconds",
    "command_timeout_seconds", "sync_timeout_seconds", "download_timeout_seconds",
    "network_read_timeout_seconds", "hf_repo_id", "smoke_images_per_class", "smoke_ood_images",
    "smoke_overrides",
}
SWEEP_FIXED_KEYS = {
    "profile", "protocol_id", "models", "training_seeds", "image_size", "resize_size", "pretrained",
    "data_input_root", "far_ood_root", "unseen_cat_root", "unseen_cat_classes", "output_root",
    "official_archives_root", "official_data_dir", "hf_repo_id", "sweep_grid", "model_overrides",
    "smoke_overrides",
}


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def reject_forbidden_path(path: Path) -> None:
    name = path.name.lower()
    token = next((item for item in FORBIDDEN_INPUT_TOKENS if item in name), None)
    if token:
        raise ValueError(f"Đầu vào bị cấm vì tên chứa '{token}': {path}")


def parse_overrides(value: object, where: str) -> dict:
    try:
        parsed = value if isinstance(value, dict) else json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"config_overrides không hợp lệ tại {where}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"config_overrides phải là JSON object tại {where}")
    return parsed


def check_override_keys(overrides: dict, where: str) -> None:
    unknown = set(overrides) - CONFIG_KEYS
    blocked = set(overrides) & SWEEP_FIXED_KEYS
    if unknown:
        raise ValueError(where + ": khóa không có trong CONFIG: " + ", ".join(sorted(unknown)))
    if blocked:
        raise ValueError(where + ": không được ghi đè khóa cố định: " + ", ".join(sorted(blocked)))


def check_architecture_values(overrides: dict, where: str) -> None:
    hidden = overrides.get("head_hidden", 0)
    fraction = overrides.get("fine_tune_fraction")
    train_fraction = overrides.get("train_fraction", 1.0)
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 0:
        raise ValueError(where + ": head_hidden phải là số nguyên ≥ 0")
    if fraction is not None and (not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 < fraction <= 1):
        raise ValueError(where + ": fine_tune_fraction phải nằm trong (0, 1]")
    if not isinstance(train_fraction, (int, float)) or isinstance(train_fraction, bool) or not 0 < train_fraction <= 1:
        raise ValueError(where + ": train_fraction phải nằm trong (0, 1]")


def validate_stage2_config(config: dict) -> None:
    if config.get("profile") != "sweep":
        raise ValueError("Cấu hình stage 2 phải có profile='sweep'")
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 1 or models[0] not in MODEL_NAMES:
        raise ValueError("Cấu hình stage 2 phải có đúng một model hợp lệ")
    seeds = config.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= {42, 43, 44}:
        raise ValueError("training_seeds chỉ nhận các seed duy nhất trong {42, 43, 44}")
    if not config.get("owner") or not config.get("protocol_id"):
        raise ValueError("Cấu hình thiếu owner hoặc protocol_id")
    grid = config.get("sweep_grid")
    if not isinstance(grid, list) or len(grid) != 1:
        raise ValueError("Cấu hình stage 2 phải có đúng một sweep point")
    point = grid[0]
    if set(point) - {"name", "overrides", "models"}:
        raise ValueError("Sweep point có khóa lạ")
    if not isinstance(point.get("name"), str) or not point["name"]:
        raise ValueError("Sweep point thiếu name")
    overrides = point.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides phải là JSON object")
    check_override_keys(overrides, "sweep_grid[" + point["name"] + "]")
    check_architecture_values(overrides, "sweep_grid[" + point["name"] + "]")
    if "models" in point and not set(point["models"]) <= set(models):
        raise ValueError("Sweep point dùng model ngoài CONFIG['models']")


def load_inputs(ranking_path: Path, factors_path: Path, base_config_path: Path):
    for path in [ranking_path, factors_path, base_config_path]:
        reject_forbidden_path(path)
    ranking = pd.read_csv(ranking_path)
    factors = pd.read_csv(factors_path)
    required_ranking = {"model", "rank", "sweep_point", "config_overrides"}
    required_factors = {"model", "factor_group", "rank", "sweep_point", "config_overrides"}
    if missing := required_ranking - set(ranking):
        raise ValueError("stage1_ranking.csv thiếu cột: " + ", ".join(sorted(missing)))
    if missing := required_factors - set(factors):
        raise ValueError("stage1_best_per_factor.csv thiếu cột: " + ", ".join(sorted(missing)))
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    if not base.get("owner") or not base.get("protocol_id"):
        raise ValueError("Cấu hình gốc thiếu owner hoặc protocol_id")
    return ranking, factors, base


def top_row(ranking: pd.DataFrame, model: str) -> pd.Series | None:
    matches = ranking[(ranking["model"] == model) & (pd.to_numeric(ranking["rank"]) == 1)]
    if matches.empty:
        return None
    if len(matches) != 1:
        raise ValueError(f"{model} có {len(matches)} dòng hạng 1")
    return matches.iloc[0]


def combine_factor_overrides(factors: pd.DataFrame, model: str) -> tuple[dict, list[dict]] | None:
    rows = factors[factors["model"] == model]
    if rows.empty:
        return None
    if rows["factor_group"].duplicated().any():
        duplicates = sorted(rows.loc[rows["factor_group"].duplicated(False), "factor_group"].unique())
        raise ValueError(f"{model} có nhiều điểm tốt nhất cho nhóm: {duplicates}")
    combined = {}
    owners = {}
    sources = []
    for _, row in rows.iterrows():
        overrides = parse_overrides(row["config_overrides"], str(row["sweep_point"]))
        for key, value in overrides.items():
            if key in combined and combined[key] != value:
                raise ValueError(
                    f"Xung đột override {key!r} của {model}: "
                    f"{owners[key]}={combined[key]!r}, {row['sweep_point']}={value!r}"
                )
            combined[key] = value
            owners[key] = row["sweep_point"]
        sources.append(
            {
                "factor_group": row["factor_group"],
                "sweep_point": row["sweep_point"],
                "rank": int(row["rank"]),
            }
        )
    return combined, sources


def make_config(base: dict, model: str, seeds: list[int], point_name: str, overrides: dict, source: dict) -> dict:
    config = deepcopy(base)
    config.update(
        {
            "profile": "sweep",
            "models": [model],
            "training_seeds": seeds,
            "sweep_grid": [{"name": point_name, "overrides": overrides}],
            "stage2_source": source,
        }
    )
    validate_stage2_config(config)
    return config


def generate_configs(ranking: pd.DataFrame, factors: pd.DataFrame, base: dict) -> tuple[dict[str, dict], list[str]]:
    generated = {}
    warnings = []
    for model, label in LABELS.items():
        row = top_row(ranking, model)
        if row is None:
            warnings.append(f"Thiếu kết quả {model}; không sinh stage2_{label}.json")
            continue
        overrides = parse_overrides(row["config_overrides"], str(row["sweep_point"]))
        name = f"stage2_{label}_{row['sweep_point']}"
        source = {"kind": "stage1_rank", "sweep_point": row["sweep_point"], "rank": int(row["rank"])}
        generated[f"stage2_{label}.json"] = make_config(base, model, [43, 44], name, overrides, source)

    for model in TRANSFER_MODELS:
        combined = combine_factor_overrides(factors, model)
        if combined is None:
            warnings.append(f"Thiếu kết quả theo nhóm của {model}; không sinh stage2_D_{model}.json")
            continue
        overrides, sources = combined
        top = top_row(ranking, model)
        if top is None:
            warnings.append(f"Thiếu hạng 1 của {model}; không sinh stage2_D_{model}.json")
            continue
        top_overrides = parse_overrides(top["config_overrides"], str(top["sweep_point"]))
        duplicate_top = overrides == top_overrides
        seeds = [43, 44] if duplicate_top else [42, 43, 44]
        source = {
            "kind": "best_per_factor",
            "points": sources,
            "duplicates_stage1_top": duplicate_top,
            "note": "Bỏ seed 42 vì overrides trùng hạng 1 giai đoạn 1" if duplicate_top else "",
        }
        name = f"stage2_D_{model}"
        generated[f"stage2_D_{model}.json"] = make_config(base, model, seeds, name, overrides, source)
    return generated, warnings


def run(ranking_path: Path, factors_path: Path, base_config_path: Path, output_dir: Path) -> dict[str, dict]:
    ranking, factors, base = load_inputs(ranking_path, factors_path, base_config_path)
    generated, warnings = generate_configs(ranking, factors, base)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, config in generated.items():
        (output_dir / filename).write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("Đã sinh:", output_dir / filename)
    for warning in warnings:
        print("CẢNH BÁO:", warning)
    manifest = {
        "protocol_id": base["protocol_id"],
        "owner": base["owner"],
        "generated": sorted(generated),
        "warnings": warnings,
    }
    (output_dir / "stage2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return generated


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--best-per-factor", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.ranking, args.best_per_factor, args.base_config, args.output_dir)


if __name__ == "__main__":
    main()
