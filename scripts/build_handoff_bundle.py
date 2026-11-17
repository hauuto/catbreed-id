"""Build a small, explicit remote payload from new official files and validation inputs only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(images, output):
    files = {}
    code = ["app/__init__.py", "app/api/__init__.py", "app/api/main.py", "app/api/real_model.py",
            "app/api/temp_model.py", "scripts/check_app_consistency.py", "scripts/benchmark_protocol.py",
            "scripts/benchmark_official_web.py", "scripts/run_thanhkhoi_validation.py",
            "scripts/start_thanhkhoi_api.sh"]
    code.extend(str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "src/catbreed_core").glob("*.py"))
    for name in code:
        files[name] = ROOT / name
    for name in ("checkpoint.pt", "thresholds.json", "calibration.json", "distance_parameters.npz",
                 "locked_selection.json", "val.npz", "validation_scores.csv"):
        files[f"model_package/{name}"] = ROOT / "model_package" / name
    with (ROOT / "model_package/validation_scores.csv").open(encoding="utf-8", newline="") as f:
        ids = [r["image_id"] for r in csv.DictReader(f)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate validation IDs")
    for image_id in ids:
        if not re.fullmatch(r"[A-Za-z0-9_]+", image_id):
            raise ValueError("Invalid validation image ID")
        files[f"inputs/oxford/{image_id}.jpg"] = images / f"{image_id}.jpg"
    case_dir = ROOT / "outputs/official_v2/gpu/cases"
    cases = json.loads((case_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    records = []
    for case in cases:
        kind = case["kind"]
        path = case_dir / f"{kind}.jpg"
        relative = f"inputs/cases/{path.name}"
        files[relative] = path
        records.append({"kind": kind, "image": relative, "truth": case["truth"],
                        "source": case["source"], "source_verification": "existing validated manifest"})
    external = [
        {"kind": "dtd", "name": "dtd_val_sample.jpg", "truth": "banded",
         "source": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
         "source_verification": "User reports torchvision DTD val1[0]; official val1 first ID is banded/banded_0006.jpg",
         "dataset_image_id": "banded/banded_0006.jpg", "split": "val1"},
        {"kind": "felibreed", "name": "felibreed.jpg", "truth": "Local",
         "source": "https://data.mendeley.com/datasets/ng6tv57j5c/2",
         "source_verification": "User confirms FeliBreed Local folder; original dataset filename not supplied",
         "dataset_image_id": None, "split": None},
    ]
    for case in external:
        path = case_dir / case["name"]
        relative = f"inputs/cases/{path.name}"
        files[relative] = path
        records.append({**case, "image": relative})
    for record in records:
        record["sha256"] = hashlib.sha256(files[record["image"]].read_bytes()).hexdigest()
    hashes = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"validation_images": len(ids), "files_sha256": hashes, "cases": records,
                "contains_legacy_models": False, "requires_manual_screenshots": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("bundle_manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{output}: {output.stat().st_size} bytes; {len(ids)} validation images; {len(records)} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/official_v2/thanhkhoi/handoff_bundle.tar.gz"))
    args = parser.parse_args()
    build(args.images, args.output)
