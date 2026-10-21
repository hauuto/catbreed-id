import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def sync_run(run_dir, repo_id):
    run_dir = Path(run_dir).resolve()

    if not run_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục: {run_dir}")

    if not (run_dir / "info.json").is_file():
        raise ValueError("Thư mục kết quả phải có info.json")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Chưa thiết lập HF_TOKEN")

    api = HfApi(token=token)
    remote_path = f"runs/{run_dir.name}"

    for attempt in range(1, 4):
        try:
            result = api.upload_folder(
                folder_path=str(run_dir),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Sync {run_dir.name}",
            )
        except Exception as exc:
            print(
                f"Upload lần {attempt}/3 thất bại: "
                f"{type(exc).__name__}",
                flush=True,
            )
            if attempt == 3:
                raise RuntimeError(
                    "Chưa đồng bộ. Giữ thư mục kết quả để upload lại."
                ) from exc
            time.sleep(3)
        else:
            print(f"Đã đồng bộ: {repo_id}/{remote_path}", flush=True)
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--repo-id",
        default="catbreed-id/catbreed-experiments",
    )
    args = parser.parse_args()
    sync_run(args.run_dir, args.repo_id)


if __name__ == "__main__":
    main()