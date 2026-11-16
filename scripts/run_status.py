"""Theo dõi các lượt chạy VPS mà không đọc kết quả test/OOD.

Script chỉ mở ``run.log`` và các tệp ``info.json`` bên dưới
``/mnt/thanhkhoi/work*``.  Tổng số điểm của bốn lượt giai đoạn 1 được lấy
từ tên thư mục; với lượt khác, tổng chỉ hiện khi study đã ghi ``sweep_grid``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GLOB = "/mnt/thanhkhoi/work*"
ALLOWED_PARENT = Path("/mnt/thanhkhoi")
KNOWN_POINT_TOTALS = {
    "work_tai": 9,
    "work_khoi": 15,
    "work_vy": 15,
    "work_hau": 12,
}
ACTIVE_STATUSES = {"running", "phase_a", "phase_b", "training"}
DONE_STATUSES = {"completed", "sweep_completed", "official_completed"}
ERROR_STATUSES = {"failed", "error"}
ERROR_RE = re.compile(
    r"Traceback|\b(?:Error|[A-Za-z]+Error|Exception)\b|\bFAIL(?:ED)?\b",
    re.IGNORECASE,
)
EPOCH_RE = re.compile(
    r"(?P<label>[^\r\n|]+?)\s*\|\s*epoch\s*(?P<epoch>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InfoFile:
    path: Path
    data: dict[str, Any]
    started: datetime | None
    finished: datetime | None


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_info(path: Path) -> InfoFile | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return InfoFile(path, data, parse_time(data.get("started_at_utc")), parse_time(data.get("finished_at_utc")))


def read_log_tail(path: Path, limit: int = 512_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            raw = handle.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def newest(items: Iterable[InfoFile]) -> InfoFile | None:
    values = list(items)
    if not values:
        return None
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    return max(values, key=lambda item: (item.started or minimum, item.path.as_posix()))


def owner_from_directory(directory: Path) -> str:
    suffix = directory.name.removeprefix("work_")
    if suffix == directory.name or not suffix:
        return "-"
    return suffix.split("_", 1)[0].capitalize()


def stage_name(label: str) -> str:
    lowered = label.casefold()
    if "đóng băng" in lowered or "frozen" in lowered:
        return "frozen"
    if "tinh chỉnh" in lowered or "finetuned" in lowered:
        return "finetuned"
    if "từ đầu" in lowered or "scratch" in lowered:
        return "scratch"
    return label.strip() or "-"


def progress_from_history(info: InfoFile | None) -> tuple[str, str]:
    if info is None:
        return "-", "-"
    history = info.data.get("history")
    if not isinstance(history, list) or not history:
        return "-", "-"
    row = history[-1]
    if not isinstance(row, dict):
        return "-", "-"
    stage = str(row.get("stage") or "-")
    epoch = row.get("epoch")
    maximum = row.get("max_epochs") or row.get("epochs") or row.get("epoch_total")
    if epoch is None:
        return stage, "-"
    return stage, f"{epoch}/{maximum}" if maximum is not None else str(epoch)


def progress_from_log(log: str) -> tuple[str, str, int | None, int | None]:
    matches = list(EPOCH_RE.finditer(log))
    if not matches:
        return "-", "-", None, None
    match = matches[-1]
    epoch, maximum = int(match.group("epoch")), int(match.group("total"))
    return stage_name(match.group("label")), f"{epoch}/{maximum}", epoch, maximum


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    value = int(round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def duration(info: InfoFile, end: datetime) -> float | None:
    if info.started is None:
        return None
    return ((info.finished or end) - info.started).total_seconds()


def point_progress(
    directory: Path,
    study: InfoFile | None,
    runs: list[InfoFile],
    current: InfoFile | None,
) -> tuple[str, int | None, int, set[str]]:
    points: list[str] = []
    for item in sorted(runs, key=lambda row: row.started or datetime.min.replace(tzinfo=timezone.utc)):
        point = item.data.get("sweep_point")
        if isinstance(point, str) and point and point not in points:
            points.append(point)
    grid = study.data.get("sweep_grid") if study else None
    total = len(grid) if isinstance(grid, list) and grid else KNOWN_POINT_TOTALS.get(directory.name)
    current_name = current.data.get("sweep_point") if current else None
    if not isinstance(current_name, str) or not current_name:
        current_name = points[-1] if points else None
    index = points.index(current_name) + 1 if current_name in points else 0
    if total and study and study.data.get("status") in DONE_STATUSES:
        index = total
    text = f"{current_name or '-'} ({index or '-'}/{total or '?'})"
    completed = {
        str(item.data.get("sweep_point"))
        for item in runs
        if item.data.get("status") in DONE_STATUSES and item.data.get("sweep_point")
    }
    return text, total, index, completed


def estimate_remaining(
    now: datetime,
    status: str,
    total: int | None,
    completed_points: set[str],
    runs: list[InfoFile],
    current: InfoFile | None,
    epoch: int | None,
    epoch_total: int | None,
) -> float | None:
    if status == "xong":
        return 0.0
    if status == "lỗi":
        return None
    completed_durations = [duration(item, now) for item in runs if item.data.get("status") in DONE_STATUSES]
    samples = [value for value in completed_durations if value is not None and value > 0]
    if total and samples:
        average = sum(samples) / len(samples)
        remaining = max(0, total - len(completed_points))
        estimate = average * remaining
        if current and current.data.get("status") in ACTIVE_STATUSES:
            elapsed = duration(current, now) or 0.0
            estimate = max(0.0, estimate - min(elapsed, average))
        return estimate
    if current and epoch and epoch_total and epoch > 0:
        elapsed = duration(current, now)
        if elapsed is not None:
            return max(0.0, elapsed / epoch * (epoch_total - epoch))
    return None


def summarize_workdir(directory: Path, now: datetime | None = None) -> dict[str, str]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    infos = [item for path in directory.rglob("info.json") if (item := read_info(path)) is not None]
    studies = [item for item in infos if item.data.get("kind") == "study"]
    runs = [item for item in infos if item.data.get("model")]
    study = newest(studies)
    running = newest(item for item in runs if item.data.get("status") in ACTIVE_STATUSES)
    current = running or newest(runs)
    log = read_log_tail(directory / "run.log")

    latest_error = "-"
    for line in reversed(log.splitlines()):
        line = line.strip()
        if line and ERROR_RE.search(line):
            latest_error = line[:120]
            break

    study_status = study.data.get("status") if study else None
    current_status = current.data.get("status") if current else None
    fatal_traceback = "Traceback (most recent call last)" in log
    if current_status in ERROR_STATUSES or fatal_traceback:
        status = "lỗi"
    elif study_status in DONE_STATUSES:
        status = "xong"
    elif running or study_status in ACTIVE_STATUSES:
        status = "đang chạy"
    elif infos and all(item.data.get("status") in DONE_STATUSES for item in infos):
        status = "xong"
    else:
        status = "đang chạy"

    owner = "-"
    for item in (current, newest(runs), newest(infos)):
        if item is None:
            continue
        candidate = item.data.get("display_name") or item.data.get("owner")
        if isinstance(candidate, str) and candidate:
            owner = candidate
            break
    if not current or not current.data.get("display_name"):
        inferred = owner_from_directory(directory)
        if inferred != "-":
            owner = inferred

    point, total, _, completed_points = point_progress(directory, study, runs, current)
    stage, epoch = progress_from_history(current)
    parsed_stage, parsed_epoch, epoch_value, epoch_total = progress_from_log(log)
    if parsed_epoch != "-":
        stage, epoch = parsed_stage, parsed_epoch
    elif stage == "-" and study_status in ACTIVE_STATUSES:
        stage = str(study_status)

    start = study.started if study else (current.started if current else (newest(infos).started if infos else None))
    finish = study.finished if study and status == "xong" else None
    elapsed = (finish or now) - start if start else None
    eta = estimate_remaining(
        now, status, total, completed_points, runs, current, epoch_value, epoch_total
    )
    return {
        "directory": directory.name,
        "owner": owner,
        "status": status,
        "point": point,
        "model": str(current.data.get("model") or "-") if current else "-",
        "stage": stage,
        "epoch": epoch,
        "elapsed": format_duration(elapsed.total_seconds() if elapsed else None),
        "eta": format_duration(eta),
        "error": latest_error,
    }


def allowed_cli_directory(path: Path) -> bool:
    resolved = path.resolve()
    return resolved.parent == ALLOWED_PARENT and resolved.name.startswith("work")


def print_table(rows: list[dict[str, str]]) -> None:
    columns = [
        ("thư mục", "directory"),
        ("owner", "owner"),
        ("trạng thái", "status"),
        ("điểm", "point"),
        ("model", "model"),
        ("stage", "stage"),
        ("epoch", "epoch"),
        ("đã chạy", "elapsed"),
        ("còn lại", "eta"),
        ("lỗi gần nhất", "error"),
    ]
    widths = []
    for title, key in columns:
        maximum = max([len(title), *(len(row[key]) for row in rows)], default=len(title))
        widths.append(min(maximum, 48 if key == "error" else 30))
    print("  ".join(title.ljust(width) for (title, _), width in zip(columns, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        values = []
        for (_, key), width in zip(columns, widths):
            value = row[key]
            if len(value) > width:
                value = value[: max(1, width - 1)] + "…"
            values.append(value.ljust(width))
        print("  ".join(values))


def main() -> None:
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path, help="thư mục /mnt/thanhkhoi/work*")
    args = parser.parse_args()
    directories = args.directories or [Path(value) for value in sorted(glob.glob(DEFAULT_GLOB))]
    invalid = [path for path in directories if not allowed_cli_directory(path)]
    if invalid:
        parser.error("chỉ được đọc /mnt/thanhkhoi/work*: " + ", ".join(map(str, invalid)))
    directories = [path for path in directories if path.is_dir()]
    if not directories:
        print("Không tìm thấy thư mục /mnt/thanhkhoi/work*.")
        return
    print_table([summarize_workdir(path) for path in directories])


if __name__ == "__main__":
    main()
