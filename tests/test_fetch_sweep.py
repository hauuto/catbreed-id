import json

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
