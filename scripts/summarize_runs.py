import argparse
import csv
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def summarize(repo_id, output_path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Chưa thiết lập HF_TOKEN")

    api = HfApi(token=token)

    # Cố định một phiên bản kho trong suốt lần tổng hợp.
    revision = api.repo_info(
        repo_id=repo_id,
        repo_type="dataset",
    ).sha

    files = set(api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    ))

    def read_json(filename):
        local_file = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            token=token,
        )
        return json.loads(Path(local_file).read_text(encoding="utf-8"))

    records = []
    metric_columns = set()

    info_files = sorted(
        name for name in files
        if name.startswith("runs/")
        and name.count("/") == 2
        and name.endswith("/info.json")
    )

    for info_file in info_files:
        info = read_json(info_file)
        run_path = info_file.rsplit("/", 1)[0]
        folder_id = run_path.split("/")[-1]

        is_smoke = bool(
            info.get("is_smoke_test")
            or info.get("status") == "smoke_test"
            or "_smoke_" in folder_id
        )

        base = {
            "run_id": info.get("run_id", folder_id),
            "owner": info.get("owner", ""),
            "git_commit": info.get("git_commit", ""),
            "status": info.get("status", ""),
            "is_smoke_test": is_smoke,
            "split_id": info.get("split_id", ""),
            "training_seed": info.get("training_seed", ""),
            "source_revision": revision,
        }

        stage_found = False

        for stage in ("frozen", "finetuned", "scratch"):
            metrics_file = f"{run_path}/{stage}/metrics.json"
            if metrics_file not in files:
                continue

            stage_found = True
            metrics = read_json(metrics_file)
            checkpoint = f"{run_path}/{stage}/checkpoint.pt"

            record = {
                **base,
                "stage": stage,
                "has_metrics": True,
                "checkpoint_path": checkpoint if checkpoint in files else "",
            }

            for key, value in metrics.items():
                if key in {"stage", "is_smoke_test"}:
                    continue

                column = f"metric_{key}"
                metric_columns.add(column)
                record[column] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )

            records.append(record)

        # Giữ cả lượt chưa có metrics để nhìn thấy hồ sơ còn thiếu.
        if not stage_found:
            records.append({
                **base,
                "stage": "",
                "has_metrics": False,
                "checkpoint_path": "",
            })

    fields = [
        "run_id", "owner", "git_commit", "status",
        "is_smoke_test", "split_id", "training_seed",
        "stage", "has_metrics", "checkpoint_path", "source_revision",
        *sorted(metric_columns),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"Đã tổng hợp {len(info_files)} lượt thành {len(records)} dòng")
    print("File kết quả:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="catbreed-id/catbreed-experiments",
    )
    parser.add_argument("--output", default="outputs/summary.csv")
    args = parser.parse_args()
    summarize(args.repo_id, args.output)


if __name__ == "__main__":
    main()