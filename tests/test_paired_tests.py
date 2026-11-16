import numpy as np
import pytest

from scripts import paired_tests


def arrays(labels, predictions, order=None):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    ids = np.array([f"image-{index}" for index in range(len(labels))])
    logits = np.full((len(labels), 12), -5.0)
    logits[np.arange(len(labels)), predictions] = 5.0
    if order is not None:
        return {"labels": labels[order], "logits": logits[order], "image_id": ids[order]}
    return {"labels": labels, "logits": logits, "image_id": ids}


def write_stage(path, data):
    path.mkdir(parents=True)
    np.savez_compressed(path / "test.npz", **data)
    return path


def test_identical_predictions_have_p_one_and_zero_delta():
    labels = np.tile(np.arange(12), 5)
    data = arrays(labels, labels)
    result = paired_tests.compare_pair("A", data, "B", data, repeats=100)
    assert result["mcnemar_p"] == 1
    assert result["delta_acc"] == 0
    assert result["ci_low"] == 0 == result["ci_high"]


def test_clearly_better_model_has_small_p_and_positive_ci():
    labels = np.tile(np.arange(12), 10)
    good = arrays(labels, labels)
    bad = arrays(labels, (labels + 1) % 12)
    result = paired_tests.compare_pair("A", good, "B", bad, repeats=200)
    assert result["mcnemar_p"] < 1e-10
    assert result["ci_low"] > 0
    assert result["ci_low_f1"] > 0


def test_alignment_accepts_different_order_and_rejects_different_ids():
    labels = np.tile(np.arange(12), 2)
    first = arrays(labels, labels)
    second = arrays(labels, labels, order=np.arange(len(labels))[::-1])
    aligned = paired_tests.align(first, second)
    assert np.array_equal(aligned[1], aligned[2])
    second["image_id"][0] = "other"
    with pytest.raises(ValueError, match="image_id khác nhau"):
        paired_tests.align(first, second)


def test_holm_and_run_all_pairs(tmp_path):
    labels = np.tile(np.arange(12), 3)
    stages = [
        write_stage(tmp_path / name, arrays(labels, prediction))
        for name, prediction in [
            ("A", labels),
            ("B", labels.copy()),
            ("C", (labels + 1) % 12),
        ]
    ]
    frame = paired_tests.run(stages, tmp_path / "paired.csv")
    assert len(frame) == 3
    assert (frame.holm_p >= frame.mcnemar_p).all()
    assert (tmp_path / "paired.csv").is_file()


def test_pair_spec_validation():
    assert paired_tests.choose_pairs(["A", "B"], "A:B") == [("A", "B")]
    with pytest.raises(ValueError, match="không tồn tại"):
        paired_tests.choose_pairs(["A", "B"], "A:C")
