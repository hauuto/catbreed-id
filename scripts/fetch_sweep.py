"""Collect phase-A sweep results from a VPS directory or Hugging Face study."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

import pandas as pd


REPO_ID = "catbreed-id/catbreed-experiments"
EXPECTED_PROTOCOL = "Nhom09_v2_2026-11-16"
ALLOWED_NAMES = {"sweep_results.csv", "calibration.json", "info.json"}
FORBIDDEN_TOKENS = ("test", "near_ood", "far_ood", "unseen")
SAFE_OWNER = re.compile(r"^[A-Za-z0-9._-]+$")


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def validate_allowed_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    logical = PurePosixPath(text)
    name = logical.name
    if name not in ALLOWED_NAMES:
        raise ValueError(f"Tên file không được phép: {name}")
    parts = logical.parts
    anchors = [index for index, part in enumerate(parts) if part in {"runs", "outputs"}]
    scoped = "/".join(parts[anchors[-1] :]) if anchors else "/".join(parts[-3:])
    token = next((item for item in FORBIDDEN_TOKENS if item in scoped), None)
    if token:
        raise ValueError(f"Đường dẫn chứa từ cấm '{token}': {path}")


def safe_owner(owner: object) -> str:
    value = str(owner or "").strip()
    if not value or not SAFE_OWNER.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Owner không hợp lệ để làm tên thư mục: {value!r}")
    return value


def read_source(study_dir: Path, owner_override: str | None, with_calibration: bool) -> dict:
    sweep = study_dir / "sweep_results.csv"
    info_path = study_dir / "info.json"
    for path in [sweep, info_path]:
        validate_allowed_path(path)
        if not path.is_file():
            raise ValueError(f"Thiếu file bắt buộc: {path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    owner = safe_owner(owner_override or info.get("owner") or info.get("display_name"))
    calibration = None
    if with_calibration:
        candidates = sorted(study_dir.rglob("calibration.json"))
        for path in candidates:
            validate_allowed_path(path)
        if len(candidates) != 1:
            raise ValueError(
                f"--with-calibration cần đúng một calibration.json trong {study_dir}, tìm thấy {len(candidates)}"
            )
        calibration = candidates[0]
    frame = pd.read_csv(sweep)
    required = {"sweep_point", "model"}
    if missing := required - set(frame):
        raise ValueError("sweep_results.csv thiếu cột: " + ", ".join(sorted(missing)))
    protocol = info.get("protocol_id")
    if protocol is None and "protocol_id" in frame and frame["protocol_id"].nunique(dropna=False) == 1:
        protocol = str(frame["protocol_id"].iloc[0])
    return {
        "study_dir": study_dir,
        "sweep": sweep,
        "info": info_path,
        "calibration": calibration,
        "owner": owner,
        "study_id": info.get("run_id", study_dir.name),
        "protocol_id": protocol,
        "rows": len(frame),
        "points": frame["sweep_point"].nunique(),
        "models": sorted(frame["model"].dropna().astype(str).unique()),
    }


def find_local_sources(base: Path, exclude: Path | None = None) -> list[Path]:
    if not base.is_dir():
        raise ValueError(f"--from-dir không phải thư mục: {base}")
    found = sorted(path.parent for path in base.glob("outputs/**/sweep_results.csv"))
    if not found:
        found = sorted(path.parent for path in base.rglob("sweep_results.csv"))
    if exclude is not None:
        excluded = exclude.resolve()
        found = [path for path in found if not path.resolve().is_relative_to(excluded)]
    if not found:
        raise ValueError(f"Không tìm thấy sweep_results.csv bên dưới {base}")
    for study_dir in found:
        validate_allowed_path(study_dir / "sweep_results.csv")
    return found


def download_hf_source(study_id: str, destination: Path, with_calibration: bool) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    if not os.environ.get("HF_TOKEN"):
        raise ValueError("--from-hf cần biến môi trường HF_TOKEN")
    if any(token in study_id.lower() for token in FORBIDDEN_TOKENS):
        raise ValueError(f"study_id chứa từ cấm: {study_id}")
    prefix = f"runs/{study_id.strip('/')}/"
    remotes = [prefix + "info.json", prefix + "sweep_results.csv"]
    if with_calibration:
        files = HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(REPO_ID, repo_type="dataset")
        candidates = sorted(path for path in files if path.startswith(prefix) and path.endswith("/calibration.json"))
        if len(candidates) != 1:
            raise ValueError(
                f"--with-calibration cần đúng một calibration.json trong {study_id}, tìm thấy {len(candidates)}"
            )
        remotes.append(candidates[0])
    for remote in remotes:
        validate_allowed_path(remote)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote,
            token=os.environ["HF_TOKEN"],
            local_dir=destination,
        )
    return destination / "runs" / study_id.strip("/")


def copy_source(source: dict, output_root: Path, overwrite: bool) -> Path:
    target = output_root / source["owner"]
    files = [(source["sweep"], target / "sweep_results.csv"), (source["info"], target / "info.json")]
    if source["calibration"] is not None:
        files.append((source["calibration"], target / "calibration.json"))
    existing = [path for _, path in files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Không ghi đè file đã có: " + ", ".join(map(str, existing)))
    target.mkdir(parents=True, exist_ok=True)
    for source_path, target_path in files:
        shutil.copy2(source_path, target_path)
    return target


def print_summary(source: dict, target: Path) -> None:
    print(
        pd.DataFrame(
            [
                {
                    "owner": source["owner"],
                    "study_id": source["study_id"],
                    "protocol_id": source["protocol_id"],
                    "rows": source["rows"],
                    "sweep_points": source["points"],
                    "models": ",".join(source["models"]),
                }
            ]
        ).to_string(index=False)
    )
    if source["protocol_id"] != EXPECTED_PROTOCOL:
        print(f"CẢNH BÁO: protocol_id khác {EXPECTED_PROTOCOL}: {source['protocol_id']}")
    print("Đã lưu:", target)


def run_local(base: Path, output_root: Path, owner: str | None, with_calibration: bool, overwrite: bool):
    targets = []
    sources = find_local_sources(base, output_root)
    if owner and len(sources) != 1:
        raise ValueError("--owner chỉ dùng khi --from-dir tìm thấy đúng một study")
    for study_dir in sources:
        source = read_source(study_dir, owner, with_calibration)
        target = copy_source(source, output_root, overwrite)
        print_summary(source, target)
        targets.append(target)
    return targets


def run_hf(study_id: str, output_root: Path, owner: str | None, with_calibration: bool, overwrite: bool):
    with tempfile.TemporaryDirectory(prefix="catbreed-sweep-") as temporary:
        study_dir = download_hf_source(study_id, Path(temporary), with_calibration)
        source = read_source(study_dir, owner, with_calibration)
        target = copy_source(source, output_root, overwrite)
        print_summary(source, target)
        return [target]


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-dir", type=Path)
    source.add_argument("--from-hf", metavar="STUDY_ID")
    parser.add_argument("--owner")
    parser.add_argument("--with-calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage1_inputs"))
    args = parser.parse_args()
    if args.from_dir:
        run_local(args.from_dir, args.output_root, args.owner, args.with_calibration, args.overwrite)
    else:
        run_hf(args.from_hf, args.output_root, args.owner, args.with_calibration, args.overwrite)


if __name__ == "__main__":
    main()
