import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_status import summarize_workdir


NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def write_info(root: Path, name: str, **values):
    path = root / "outputs" / name / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


def test_phase_a_uses_known_total_and_does_not_treat_http_500_as_fatal(tmp_path):
    root = tmp_path / "work_tai"
    write_info(
        root,
        "study",
        kind="study",
        profile="sweep",
        status="phase_a",
        started_at_utc="2026-11-16T11:50:00+00:00",
    )
    (root / "run.log").write_text("HTTP Error 500 thrown while requesting POST\nRetrying in 2s\n", encoding="utf-8")

    row = summarize_workdir(root, NOW)

    assert row["owner"] == "Tai"
    assert row["status"] == "đang chạy"
    assert row["point"] == "- (-/9)"
    assert row["stage"] == "phase_a"
    assert "HTTP Error 500" in row["error"]


def test_running_model_reads_history_and_sweep_point(tmp_path):
    root = tmp_path / "work_khoi"
    write_info(
        root,
        "study",
        kind="study",
        status="phase_a",
        started_at_utc="2026-11-16T11:00:00+00:00",
    )
    write_info(
        root,
        "run",
        display_name="Khoi",
        status="running",
        model="mobilenet_v2",
        sweep_point="drop0",
        started_at_utc="2026-11-16T11:50:00+00:00",
        history=[{"stage": "finetuned", "epoch": 3, "max_epochs": 30}],
    )
    (root / "run.log").write_text("đang huấn luyện\n", encoding="utf-8")

    row = summarize_workdir(root, NOW)

    assert row["status"] == "đang chạy"
    assert row["point"] == "drop0 (1/15)"
    assert row["model"] == "mobilenet_v2"
    assert row["stage"] == "finetuned"
    assert row["epoch"] == "3/30"


def test_log_epoch_is_fallback_when_info_omits_history(tmp_path):
    root = tmp_path / "work_vy"
    write_info(root, "study", kind="study", status="phase_a", started_at_utc="2026-11-16T11:00:00+00:00")
    write_info(
        root,
        "run",
        display_name="Vy",
        status="running",
        model="efficientnet_b0",
        sweep_point="unf25",
        started_at_utc="2026-11-16T11:55:00+00:00",
    )
    (root / "run.log").write_text(
        "EfficientNet-B0 tinh chỉnh | epoch  4/30 | val F1 0.8\n", encoding="utf-8"
    )

    row = summarize_workdir(root, NOW)

    assert row["stage"] == "finetuned"
    assert row["epoch"] == "4/30"
    assert row["eta"] != "?"


def test_completed_study_reports_done_and_zero_eta(tmp_path):
    root = tmp_path / "work_custom"
    write_info(
        root,
        "study",
        kind="study",
        status="sweep_completed",
        started_at_utc="2026-11-16T10:00:00+00:00",
        finished_at_utc="2026-11-16T11:00:00+00:00",
        sweep_grid=[{"name": "a"}, {"name": "b"}],
    )
    for index, point in enumerate(("a", "b")):
        write_info(
            root,
            point,
            display_name="Custom",
            status="completed",
            model="resnet50",
            sweep_point=point,
            started_at_utc=f"2026-11-16T10:{index}0:00+00:00",
            finished_at_utc=f"2026-11-16T10:{index}5:00+00:00",
        )
    (root / "run.log").write_text("Sweep xong.\n", encoding="utf-8")

    row = summarize_workdir(root, NOW)

    assert row["status"] == "xong"
    assert row["point"] == "b (2/2)"
    assert row["elapsed"] == "1h00m"
    assert row["eta"] == "0s"


def test_failed_run_reports_error(tmp_path):
    root = tmp_path / "work_hau"
    write_info(
        root,
        "run",
        display_name="Hau",
        status="failed",
        model="resnet50",
        sweep_point="head512",
        started_at_utc="2026-11-16T11:50:00+00:00",
        finished_at_utc="2026-11-16T11:55:00+00:00",
    )
    (root / "run.log").write_text("Traceback (most recent call last):\nRuntimeError: CUDA OOM\n", encoding="utf-8")

    row = summarize_workdir(root, NOW)

    assert row["status"] == "lỗi"
    assert row["error"] == "RuntimeError: CUDA OOM"


def test_traceback_overrides_stale_phase_a_status(tmp_path):
    root = tmp_path / "work_tai"
    write_info(
        root,
        "study",
        kind="study",
        status="phase_a",
        started_at_utc="2026-11-16T11:50:00+00:00",
    )
    (root / "run.log").write_text(
        "Traceback (most recent call last):\nValueError: corrupt input\n", encoding="utf-8"
    )

    row = summarize_workdir(root, NOW)

    assert row["status"] == "lỗi"
    assert row["error"] == "ValueError: corrupt input"
