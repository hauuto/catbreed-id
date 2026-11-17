# Bàn giao bổ sung cho Vỹ – web app dùng `catbreed_core` – 16/11/2026 (18:30)

File này **bổ sung** cho `BanGiao_WebApp_Vy_2026-11-16.md`. Mục tiêu, mốc giờ, mục 7–9 của bản gốc giữ nguyên. Thay đổi chính: phần dựng mô hình, tiền xử lý và điểm từ chối **không tự viết lại trong app nữa**, mà gọi module dùng chung `src/catbreed_core/`.

## 1. Tình trạng module

| Mục | Trạng thái |
|---|---|
| Nhánh | `feat/inference-core` (commit `be94d0f`) |
| Nguồn code | Chép nguyên từ `notebooks/kaggle_runner.ipynb` (`CatClassifier`, `reload_checkpoint`, `read_rgb`, `evaluation_geometry`, `logit_scores`, `DistanceScores`), đã đối chiếu từng hàm |
| Kiểm khớp | **Đạt** trên VPS (10 passed, không skip): 20 ảnh validation của checkpoint lượt 13/11, logits và 5 điểm lệch < 1e-4, Mahalanobis lệch tương đối < 1e-6 |
| Kiến trúc | Cả 4: `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `cnn_baseline` (tên baseline trong notebook là `cnn_baseline`) |

Vì module đã khớp notebook, các dòng 1–4, 7–9 trong bảng "Những gì đang sai" (mục 2 bản gốc) được giải quyết bằng cách thay code cũ bằng lời gọi module.

## 2. Đưa module vào nhánh app

```
git switch feature/thezyx
git pull
git merge origin/feat/inference-core
```

Nhánh `feat/inference-core` chỉ thêm file mới (`src/catbreed_core/*`, `tests/test_catbreed_core.py`), nên không có xung đột. Repo chưa có `pyproject.toml`; trong app thêm `src/` vào `sys.path` (giống cách `tests/test_catbreed_core.py` đang làm) hoặc chạy app với `PYTHONPATH=src`.

## 3. Ba hàm cần dùng

| Hàm | Dùng để | Thay cho |
|---|---|---|
| `load_package(dir, device="cpu") -> Package` | Nạp `checkpoint.pt`, `thresholds.json`, `calibration.json`, `distance_parameters.npz`, `locked_selection.json` (nếu có). Tự kiểm SHA-256; sai thì `ValueError` | `real_model.py` dựng ResNet50, đọc `ood_stats.npz` |
| `predict_arrays(pkg, image, exif=...) -> (logits, features, scores)` | `image` là đường dẫn hoặc `PIL.Image`. Trả logits **thô** `[1, 12]`, đặc trưng sau GAP `[1, d]`, dict 5 điểm `MSP, MaxLogit, Energy, Mahalanobis, KNN` (T = 1) | `src/preprocessing.py` và phần tính điểm của app |
| `decide(pkg, scores, logits, method=None) -> dict` | Trả `method`, `tau`, `score`, `accepted`, `probs` (= softmax(z / T), T từ `calibration.json`), `top_class` | `status: "unverified"`, `threshold: None` |

Lưu ý khi gọi:
1. **`exif`**: kiểm khớp dùng `exif=False`; chạy thật cho người dùng dùng `exif=True`. Ghi rõ điểm khác này trong Ch.5.
2. **Phương pháp từ chối**: nếu thư mục có `locked_selection.json`, `decide` tự lấy `rejection.method` và **bỏ qua** tham số `method`. Khi thử với lượt 13/11 (không có file này) thì truyền `method="KNN"`.
3. **Top-5**: lấy từ `result["probs"]` (12 giá trị, đã hiệu chỉnh).
4. **Bị từ chối**: nếu `accepted` là `False` thì app không trả `top_label` (đúng mục 3 bản gốc). Vẫn có thể hiện Top-5 ở chế độ nâng cao.
5. **5 τ ở chế độ nâng cao**: đọc `pkg.thresholds["T1"]`. Không dùng `T_star`.

## 4. Những gì module KHÔNG làm (app tự giữ)

| Phần | Làm thế nào |
|---|---|
| Grad-CAM | Giữ cách hiện tại (mục 3 bản gốc). Mô hình lấy từ `pkg.model`; tầng ra cuối là `pkg.model.features` |
| Thông tin mô hình | Lấy từ `pkg.saved`: `model_name`, `stage`, `run_id` (seed tách từ `run_id`, ví dụ `_s42_`), cùng `pkg.checkpoint_sha256`, `method`, `tau`, `pkg.calibration["temperature"]` |
| Chặn gói smoke | Module **chưa** có cờ `is_smoke_test`. Tạm thời app tự kiểm: nếu `pkg.saved["config"].get("profile") == "smoke"` thì báo lỗi, không phục vụ |
| Đường dẫn gói | Đọc từ một thư mục `model_package/` (biến môi trường hoặc tham số), không đặt cứng |

## 5. Kiểm khớp (mục 6 bản gốc) – cách làm mới

- Tính khớp **của module** đã được xác nhận (mục 1).
- Khôi viết `app/consistency_check.py` gọi `predict_arrays(..., exif=False)` và xuất `app_consistency.csv` cho Phụ lục.
- Vỹ chỉ cần kiểm **app có gọi đúng module**: chạy `consistency_check.py` với chính thư mục `model_package/` mà app đang nạp, kết quả phải đạt.
- Khi có checkpoint v2 (mục 7 bản gốc): thay file trong `model_package/`, chạy lại `consistency_check.py` với `val.npz`, `validation_scores.csv` của checkpoint v2.

Chạy test của module (cần `HF_TOKEN`, ảnh Oxford):
```
export OXFORD_PETS_ROOT=<thư mục oxford-iiit-pet có images/>
python -m pytest tests/test_catbreed_core.py -q -rs
```
Kỳ vọng: không có dòng `SKIPPED`.

## 6. Lưu ý môi trường

| Tình huống | Cách xử lý |
|---|---|
| Windows báo `DLL load failed ... Application Control policy` khi import numpy | Máy bật Smart App Control. **Không tắt** (tắt rồi không bật lại được). Chạy trên Linux/VPS/Colab |
| `pip install -r requirements.txt` lỗi trên Python 3.10 (ví dụ `filelock==4.2.4`) | `requirements.txt` ghim cho Python ≥ 3.11. Trên 3.10 cài thẳng: `torch torchvision numpy scipy scikit-learn pandas pillow huggingface_hub` |
| `requirements.txt` | UTF-16 LE, CRLF. Không mở bằng editor đổi mã hóa |
| Token HF | Chỉ đặt trong biến môi trường, không ghi vào file hay commit |

## 7. Việc của Vỹ (thay mục 3 bản gốc)

1. Merge `feat/inference-core` vào `feature/thezyx` (mục 2).
2. Trong `real_model.py`: bỏ phần tự dựng ResNet50, tự tiền xử lý, tự tính điểm; thay bằng `load_package`, `predict_arrays`, `decide`.
3. `main.py`: trả về đúng tên phương pháp notebook, `accepted`/`rejected`, Top-5 từ `probs`, thông tin mô hình (mục 4).
4. Thêm chặn gói smoke (mục 4).
5. Thử với checkpoint lượt 13/11 (`method="KNN"`), chạy `consistency_check.py` khi Khôi xong.
6. Phần còn lại (mục 7–9 bản gốc) giữ nguyên.

Không đưa số hay ảnh chụp từ checkpoint lượt 13/11 vào báo cáo.
