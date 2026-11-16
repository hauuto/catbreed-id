import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import fetch_sweep


def make_study(root, protocol=fetch_sweep.EXPECTED_PROTOCOL, owner="Khoi"):
    study = root / "outputs" / "study_abc"
    study.mkdir(parents=True)
    pd.DataFrame(
        [
            {"sweep_point": "drop0", "model": "mobilenet_v2"},
            {"sweep_point": "drop50", "model": "mobilenet_v2"},
        ]
    ).to_csv(study / "sweep_results.csv", index=False)
    (study / "info.json").write_text(
        json.dumps({"run_id": "study_abc", "owner": owner, "protocol_id": protocol}), encoding="utf-8"
    )
    return study


def test_fetches_local_study(tmp_path, capsys):
    make_study(tmp_path)
    targets = fetch_sweep.run_local(tmp_path, tmp_path / "collected", None, False, False)
    assert (targets[0] / "sweep_results.csv").is_file()
    assert (targets[0] / "info.json").is_file()
    assert "mobilenet_v2" in capsys.readouterr().out


def test_does_not_overwrite_without_flag(tmp_path):
    make_study(tmp_path)
    output = tmp_path / "collected"
    fetch_sweep.run_local(tmp_path, output, None, False, False)
    with pytest.raises(FileExistsError, match="Không ghi đè"):
        fetch_sweep.run_local(tmp_path, output, None, False, False)
    fetch_sweep.run_local(tmp_path, output, None, False, True)


def test_warns_for_other_protocol(tmp_path, capsys):
    make_study(tmp_path, protocol="old")
    fetch_sweep.run_local(tmp_path, tmp_path / "collected", None, False, False)
    assert "CẢNH BÁO" in capsys.readouterr().out


def test_forbidden_and_unknown_paths_are_rejected():
    with pytest.raises(ValueError, match="từ cấm"):
        fetch_sweep.validate_allowed_path("runs/study/test/info.json")
    with pytest.raises(ValueError, match="không được phép"):
        fetch_sweep.validate_allowed_path("checkpoint.pt")


def test_owner_must_be_safe(tmp_path):
    make_study(tmp_path, owner="../escape")
    with pytest.raises(ValueError, match="Owner không hợp lệ"):
        fetch_sweep.run_local(tmp_path, tmp_path / "collected", None, False, False)


def test_with_calibration_requires_exactly_one(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError, match="đúng một"):
        fetch_sweep.read_source(study, None, True)
    stage = study / "run" / "frozen"
    stage.mkdir(parents=True)
    (stage / "calibration.json").write_text(json.dumps({"n_images": 10}), encoding="utf-8")
    source = fetch_sweep.read_source(study, None, True)
    assert source["calibration"] == stage / "calibration.json"


def test_discovery_excludes_its_own_output(tmp_path):
    study = make_study(tmp_path)
    output = tmp_path / "outputs" / "stage1_inputs"
    fetch_sweep.run_local(tmp_path, output, None, False, False)
    assert fetch_sweep.find_local_sources(tmp_path, output) == [study]


def test_fetches_hf_study_with_mocked_download(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    def fake_list(self, repo_id, repo_type):
        return [
            "runs/study_hf/info.json",
            "runs/study_hf/sweep_results.csv",
            "runs/study_hf/run/frozen/calibration.json",
        ]

    def fake_download(repo_id, repo_type, filename, token, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "info.json":
            path.write_text(
                json.dumps(
                    {"run_id": "study_hf", "owner": "Vy", "protocol_id": fetch_sweep.EXPECTED_PROTOCOL}
                ),
                encoding="utf-8",
            )
        elif path.name == "sweep_results.csv":
            pd.DataFrame([{"sweep_point": "unf50", "model": "efficientnet_b0"}]).to_csv(path, index=False)
        else:
            path.write_text(json.dumps({"n_images": 100}), encoding="utf-8")
        return str(path)

    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", fake_list)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    targets = fetch_sweep.run_hf("study_hf", tmp_path / "collected", None, True, False)
    assert (targets[0] / "sweep_results.csv").is_file()
    assert (targets[0] / "calibration.json").is_file()
