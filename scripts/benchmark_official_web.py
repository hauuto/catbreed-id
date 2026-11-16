"""Run exactly 50 validation images and discard the first 5 (45 measured).

This produces technical measurements; chapter5_ready remains false while consistency fails.
"""
import argparse
import csv
import ctypes
import getpass
import json
import os
from pathlib import Path
import platform
import socket
import sys

import numpy as np
from PIL import Image
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api.real_model import DeploymentClassifier
from scripts.benchmark_protocol import collect_trials


def machine_info():
    cpu = platform.processor()
    ram = None
    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo").read_text()
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                    if line.startswith("model name")), cpu)
        meminfo = Path("/proc/meminfo").read_text()
        ram = next((int(line.split()[1]) * 1024 for line in meminfo.splitlines()
                    if line.startswith("MemTotal:")), None)
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page",
                                                      "avail_page", "total_virtual", "avail_virtual", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            ram = status.total_phys
    return {"os": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "hostname": socket.gethostname(), "user": getpass.getuser(),
            "ram_bytes": ram, "device": "cpu", "torch": torch.__version__,
            "torchvision": torchvision.__version__, "python": platform.python_version(),
            "torch_threads": torch.get_num_threads()}


def run(package, images, output, device="cpu", expected_user=None):
    if expected_user and getpass.getuser() != expected_user:
        raise ValueError(f"Benchmark must run as {expected_user}; got {getpass.getuser()}")
    torch.set_num_threads(1)
    model = DeploymentClassifier(package, device=device)
    with np.load(package / "val.npz", allow_pickle=False) as data:
        ids = data["image_id"].astype(str).tolist()
        labels = data["labels"].copy()
    if len(ids) < 50:
        raise ValueError("Cần ít nhất 50 ảnh validation")
    # Evenly spread through the complete validation ID list; selection never uses predicted outcomes.
    chosen = np.linspace(0, len(ids) - 1, 50, dtype=int).tolist()
    for i in chosen:
        if not (images / f"{ids[i]}.jpg").is_file():
            raise FileNotFoundError(images / f"{ids[i]}.jpg")
    def predict_trial(i):
        with Image.open(images / f"{ids[i]}.jpg") as img:
            img.load()  # Decode/disk IO excluded; same timing boundary as API main.py.
            result = model.predict(img)
        return {"image_id": ids[i], "label_index": int(labels[i]), "status": result["status"],
                "top_label": result["top_label"], **result["timings_ms"]}
    rows = collect_trials(chosen, predict_trial)
    measured = [r for r in rows if not r["warmup"]]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "latency_raw.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "latency_measured.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(measured)
    machine = machine_info()
    machine["device"] = str(model.device)
    if model.device.type == "cuda":
        machine["gpu"] = torch.cuda.get_device_name(model.device)
        machine["gpu_memory_bytes"] = torch.cuda.get_device_properties(model.device).total_memory
        machine["cuda_runtime"] = torch.version.cuda
    summary = {"chapter5_ready": False, "reason": "Technical measurements; complete consistency and browser evidence before chapter 5",
               "protocol": "50_total_discard_first_5", "n_total": 50,
               "n_measured": 45, "n_warmup": 5, "precision": model.precision_name,
               "exif_transpose": model.exif_transpose, "machine": machine,
               "checkpoint_sha256": model.checkpoint_sha256, "run_id": model.run_id,
               "size_bytes": {name: (package / name).stat().st_size for name in
                              ("checkpoint.pt", "distance_parameters.npz")},
               "timing_scope": "PIL image already decoded; excludes upload, disk IO, JSON encoding, browser rendering",
               "gradcam_scope": "Backward and heatmap normalization; PNG encoding counted in postprocess",
               "latency_ms": {key: {"median": float(np.median([r[key] for r in measured])),
                                     "p95": float(np.quantile([r[key] for r in measured], 0.95, method="linear"))}
                              for key in rows[0] if key in ("preprocess", "inference", "rejection", "gradcam", "postprocess", "total")}}
    (output / "latency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path("model_package"))
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/official_v2"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expected-user", default=None)
    args = parser.parse_args()
    run(args.package, args.images, args.output, args.device, args.expected_user)
