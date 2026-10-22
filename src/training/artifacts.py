import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_REPO = "catbreed-id/catbreed-experiments"


def sync_results(run_dir, repo_id=DEFAULT_HF_REPO):
    """Upload kết quả; trả về False nếu chưa đồng bộ được."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "sync_run.py"),
                "--run-dir",
                str(Path(run_dir).resolve()),
                "--repo-id",
                repo_id,
            ],
            check=False,
        )
    except OSError as exc:
        print(f"Chưa đồng bộ: {type(exc).__name__}", flush=True)
        return False

    return result.returncode == 0


def save_stage(
    run_dir,
    stage,
    checkpoint,
    metrics,
    repo_id=DEFAULT_HF_REPO,
):
    """Lưu checkpoint được chọn của một giai đoạn rồi upload."""
    import torch

    if stage not in {"frozen", "finetuned", "scratch"}:
        raise ValueError(f"Giai đoạn không hợp lệ: {stage}")

    run_dir = Path(run_dir).resolve()

    if not (run_dir / "info.json").is_file():
        raise ValueError("Cần tạo info.json trước khi lưu checkpoint")

    # Kiểm tra metrics chuyển thành JSON được trước khi ghi file.
    metrics_text = json.dumps(
        metrics,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Ghi xong file tạm mới thay file checkpoint chính.
    temporary_checkpoint = stage_dir / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(stage_dir / "checkpoint.pt")

    temporary_metrics = stage_dir / "metrics.json.tmp"
    temporary_metrics.write_text(metrics_text, encoding="utf-8")
    temporary_metrics.replace(stage_dir / "metrics.json")

    print(f"Đã lưu checkpoint và metrics: {stage}", flush=True)

    return sync_results(run_dir, repo_id)