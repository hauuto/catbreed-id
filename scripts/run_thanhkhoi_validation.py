"""Verify transferred evidence and run only under the authorized thanhkhoi account."""
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run():
    if getpass.getuser() != "thanhkhoi":
        raise RuntimeError("This runner must execute as thanhkhoi")
    if not ROOT.resolve().is_relative_to(Path("/mnt/thanhkhoi")):
        raise RuntimeError("Run directory must stay under /mnt/thanhkhoi")
    os.chdir(ROOT)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    output = ROOT / "outputs/thanhkhoi"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Invalid payload entry: {name}")
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Transferred file hash mismatch: {name}")
    outcomes = {}
    for count, label in ((20, "mandatory_20"), (0, "all_validation")):
        command = [sys.executable, "scripts/check_app_consistency.py", "--images", "inputs/oxford",
                   "--device", "cuda", "--count", str(count), "--output", str(output / f"{label}.csv")]
        outcomes[label] = subprocess.run(command, check=False).returncode
    command = [sys.executable, "scripts/benchmark_official_web.py", "--images", "inputs/oxford",
               "--device", "cuda", "--expected-user", "thanhkhoi", "--output", str(output)]
    outcomes["benchmark"] = subprocess.run(command, check=False).returncode
    from PIL import Image
    import torch
    from app.api.real_model import DeploymentClassifier
    torch.set_num_threads(1)
    model = DeploymentClassifier(ROOT / "model_package", device="cuda")
    cases = []
    for case in manifest["cases"]:
        with Image.open(ROOT / case["image"]) as image:
            image.load()
            result = model.predict(image)
        case_output = output / "cases"
        case_output.mkdir(exist_ok=True)
        (case_output / f"{case['kind']}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        cases.append({**case, "status": result["status"], "top_label": result["top_label"],
                      "ood": result["ood"], "model": result["model"]})
    report = {"hostname": socket.gethostname(), "user": getpass.getuser(), "root": str(ROOT),
              "python": sys.executable, "returncodes": outcomes, "cases": cases,
              "chapter5_ready": False, "pending": ["Human browser screenshots", "Document numerical consistency limits"]}
    (output / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"returncodes": outcomes, "cases": [{k: c[k] for k in ("kind", "truth", "status", "top_label")} for c in cases]}, ensure_ascii=False, indent=2))
    return int(any(code not in (0, 1) for code in outcomes.values()) or outcomes["benchmark"] != 0)


if __name__ == "__main__":
    raise SystemExit(run())
