# Bàn giao đợt 3 của Khôi – 16/11/2026

> Trạng thái: đang cập nhật. Bốn script độc lập đã hoàn thành; thông tin VPS và consistency app đang chờ phụ thuộc bên ngoài.

## 1. Nhánh và commit

Nhánh: `feat/khoi-analysis`, tách từ `feat/sweep-profile` tại `9112481`.

| Việc | Commit | Trạng thái |
|---|---|---|
| Gom kết quả sweep | `51146a0` | Code và test xong, đã push |
| Bảng geometry validation | `7588158` | Code và test xong, đã push |
| Histogram điểm ID/OOD | `1e341b4` | Code và test giả xong, đã push; chưa chạy test/OOD thật |
| Kiểm định ghép cặp | `bf89c20` | Code và test giả xong, đã push; chưa chạy ID test thật |
| Thông tin VPS | — | Đã có GPU/CPU/RAM/OS/commit; còn thiếu Python/PyTorch trong virtualenv |
| Consistency app | — | Chờ Vỹ cung cấp hàm inference Python |

## 2. Kết quả test

Lệnh:

```powershell
python -m pytest tests -q -W error
```

Kết quả hiện tại: `37 passed`.

Đã kiểm tra thêm:

- `python -m compileall -q scripts tests`;
- `git diff --check`;
- `--help` của bốn CLI mới;
- chạy end-to-end cả bốn CLI bằng dữ liệu tổng hợp.

Artifact nghiệm thu giả được giữ local, không commit:

```text
outputs/khoi_analysis_acceptance_20261116_164936/
```

## 3. `scripts/fetch_sweep.py`

Đã hỗ trợ:

- `--from-dir`: tìm `outputs/**/sweep_results.csv` trên filesystem;
- `--from-hf`: tải đúng study từ `catbreed-id/catbreed-experiments`;
- chỉ cho phép `sweep_results.csv`, `calibration.json`, `info.json`;
- chặn đường dẫn chứa `test`, `near_ood`, `far_ood`, `unseen`;
- đọc và in owner, study ID, protocol, số dòng, số sweep point, model;
- cảnh báo protocol khác `Nhom09_v2_2026-11-16`;
- không ghi đè nếu thiếu `--overwrite`;
- kiểm tra owner an toàn và không tự thu lại output của chính script.

Ví dụ VPS/local:

```bash
python scripts/fetch_sweep.py \
  --from-dir /mnt/thanhkhoi/<base-cua-luot> \
  --owner Khoi \
  --output-root outputs/stage1_inputs
```

Ví dụ Hugging Face:

```bash
export HF_TOKEN=...
python scripts/fetch_sweep.py \
  --from-hf <study_id> \
  --owner Khoi \
  --output-root outputs/stage1_inputs
```

Tại lần kiểm tra gần nhất, kho Hugging Face chưa có `sweep_results.csv`.

## 4. `scripts/geometry_table.py`

Đã hỗ trợ:

- nhận nhiều file hoặc thư mục;
- tự tìm `**/eval_geometry.csv`;
- kiểm tra đủ metadata và 10 cột metric;
- tính delta theo từng checkpoint trước khi lấy trung bình;
- tổng hợp mean ± std theo `(model, stage)`;
- đếm số checkpoint letterbox tốt hơn, bằng hoặc kém cho accuracy và macro-F1;
- chặn đường dẫn test/OOD.

Ví dụ:

```powershell
python scripts/geometry_table.py `
  outputs/runs `
  --output outputs/geometry/eval_geometry_summary.csv
```

Tại lần kiểm tra gần nhất, kho Hugging Face chưa có `eval_geometry.csv` để chạy thật.

## 5. `scripts/score_histograms.py`

Đã hỗ trợ:

- năm phương pháp MSP, MaxLogit, Energy, Mahalanobis, KNN;
- ID test và các nhóm near-OOD, far-OOD, unseen-cat nếu có;
- cùng bins trong từng ô, `density=True`;
- vạch ngưỡng `T1`;
- tự dùng `symlog` cho Mahalanobis khi dải trị tuyệt đối vượt ba bậc độ lớn;
- PNG 300 dpi, chữ đen, không title và legend phía trên;
- `score_summary.csv` với n, median, q05, q95 và tỷ lệ `score >= τ`.

Ví dụ sau khi Hậu báo đã khóa v2:

```powershell
python scripts/score_histograms.py `
  <stage-dir-da-khoa> `
  --output-dir outputs/score_histograms
```

Chưa tải hoặc mở score test/OOD thật. Nghiệm thu hiện chỉ dùng dữ liệu Gaussian giả.

## 6. `scripts/paired_tests.py`

Đã hỗ trợ:

- ghép theo `image_id`, không phụ thuộc thứ tự dòng;
- phát hiện tập ảnh, labels hoặc ID trùng không hợp lệ;
- McNemar chính xác hai phía;
- bootstrap ghép cặp 1.000 lần, seed 42;
- CI 95% của ΔAccuracy và ΔF1 macro 12 lớp;
- so mọi cặp hoặc chọn bằng `--pairs`;
- hiệu chỉnh Holm khi có nhiều cặp.

Ví dụ sau khi Hậu báo đã khóa v2:

```powershell
python scripts/paired_tests.py `
  <stage-A> <stage-B> <stage-C> `
  --output outputs/paired_tests.csv
```

Chưa tải hoặc mở `test.npz` thật. Nghiệm thu hiện chỉ dùng NPZ giả do test tạo.

## 7. Thông tin VPS – đã có một phần

Thông tin người dùng chạy trực tiếp trên VPS đã được lưu trong `env_vps.txt`:

- GPU: NVIDIA RTX A6000, 49.140 MiB VRAM;
- driver 580.178.04, CUDA báo bởi driver 13.0;
- CPU: Intel Xeon Gold 6138, 20 core / 40 thread;
- RAM 62 GiB, available 54 GiB tại thời điểm đo;
- Ubuntu 22.04.5 LTS;
- repo VPS ở commit `902f1c7` trên `feat/sweep-profile`.

Phần Python/PyTorch/torchvision chưa có vì lệnh được chạy ngoài virtualenv và VPS không có alias `python`. Cần kích hoạt virtualenv của repo rồi chạy lại đúng một lệnh Python ghi trong `env_vps.txt`.

Kết nối tự động từ máy agent qua alias `ssh a6000` vẫn bị server từ chối public key, nên agent chưa thể tự chạy phần còn thiếu.

Không có train, GPU job hoặc file trong `--base` nào bị thay đổi.

## 8. Consistency app – chưa xong

Nhánh hiện tại mới có `/health`. Đã tìm thấy predictor của Vỹ trên `origin/feature/thezyx` (`app/api/real_model.py`, commit `0c14b48`) với `ResNet50Classifier.predict_raw()` và `predict()`.

Chưa thể dùng trực tiếp predictor này cho consistency v2 vì:

- package của Vỹ hiện nhận `weights.pth` và `ood_stats.npz`;
- bàn giao v2 yêu cầu `checkpoint.pt`, `thresholds.json`, `calibration.json`, `distance_parameters.npz`;
- implementation hiện cố định ResNet50, trong khi checkpoint triển khai v2 chưa được khóa và có thể là backbone khác;
- tên score trả về cũng chưa đồng nhất hoàn toàn với notebook.

Cần Vỹ chốt hàm Python của app, tối thiểu:

```python
predict_arrays(image) -> (logits, scores)
```

và cách:

- nạp `model_package`;
- tắt EXIF trong consistency check;
- trả đúng năm score;
- xác định logits trước hay sau temperature scaling.

Chưa viết lại preprocessing riêng hoặc ép adapter giả để tránh tạo một implementation không đại diện cho app. Cần Vỹ cập nhật predictor/package contract hoặc chốt adapter chính thức trước khi viết consistency checker.

## 9. Tuân thủ khóa dữ liệu

- Không dùng dataset `roonyvtt/catbreed-cats-dogs-all`.
- Không tải hoặc mở test/OOD thật cho histogram và paired tests.
- Không mở study 13/11 hoặc `20261116_052028_study_99d77653` để lấy dữ liệu test.
- Các file `test_scores.csv`, OOD scores và `test.npz` trong artifact nghiệm thu đều là dữ liệu tổng hợp được tạo local.

## 10. Việc tiếp theo

1. Kích hoạt virtualenv trên VPS và bổ sung phiên bản Python/PyTorch/CUDA runtime/cuDNN/torchvision vào `env_vps.txt` (hoặc cập nhật authorized key để agent tự chạy).
2. Khi đồng đội sinh sweep, chạy `fetch_sweep.py`, rồi `rank_sweep.py`.
3. Khi có `eval_geometry.csv`, chạy `geometry_table.py`.
4. Khi Vỹ push predictor, hoàn thiện `app/consistency_check.py` và test.
5. Chỉ sau khi Hậu báo đã khóa v2 mới tải dữ liệu test/OOD và chạy hai phân tích hậu khóa.
6. Cập nhật file này bằng commit, lệnh chạy thật và artifact cuối.

