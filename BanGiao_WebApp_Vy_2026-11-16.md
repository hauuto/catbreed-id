# Bàn giao web app CatBreedID cho Vỹ – 16/11/2026

Mục tiêu hôm nay: web app nạp **đúng mô hình triển khai của lượt official v2** (kiến trúc bất kỳ trong 4 mô hình), áp dụng **đúng ngưỡng từ chối, nhiệt độ, ngân hàng KNN của chính checkpoint đó**, cho kết quả **khớp notebook**, rồi chụp ảnh và đo độ trễ cho Ch.5.

Code hiện tại: nhánh `feature/thezyx` (`app/api/main.py`, `app/api/real_model.py`, `app/api/temp_model.py`, `src/preprocessing.py`).
Code notebook để đối chiếu: nhánh `feat/sweep-profile`, file `notebooks/kaggle_runner.ipynb` (các hàm `CatClassifier`, `reload_checkpoint`, `read_rgb`, `evaluation_geometry`, `logit_scores`, `DistanceScores`).

## 1. Hạn và mốc
| Giờ | Việc | Xong khi |
|---|---|---|
| 15:15–18:00 | Sửa app theo mục 3, thử với checkpoint lượt 13/11 (mục 5) | Kiểm khớp đạt (mục 6) |
| 18:00–21:00 | Phần đầu báo cáo, khung Ch.5 | Bản nháp |
| ~21:00 | Có checkpoint v2: thay file (mục 7), chạy lại kiểm khớp | Đạt |
| 21:00–22:00 | Chụp ảnh, đo độ trễ (mục 8), viết Ch.5 | Ch.5 xong |

## 2. Những gì đang sai hoặc thiếu
| # | Hiện trạng (`real_model.py`, `main.py`) | Vì sao phải sửa |
|---|---|---|
| 1 | Chỉ nhận ResNet50 (`arch != "resnet50"` thì báo lỗi), tự dựng backbone ResNet50 | Mô hình triển khai v2 có thể là EfficientNet-B0 hoặc MobileNetV2 (lượt 13/11 chọn EfficientNet-B0) |
| 2 | Đọc khóa `arch`, `preprocess`, `classes`, `checkpoint_id`, `model_cfg` | Checkpoint của notebook dùng khóa khác (mục 4) |
| 3 | Đầu phân loại cố định `Dropout → Linear` | v2 có thể chọn `head_hidden = 512` (Linear → ReLU → Dropout → Linear) |
| 4 | Đọc `ood_stats.npz` tự định dạng | Notebook lưu `distance_parameters.npz` (mục 4) |
| 5 | `status: "unverified"`, `threshold: None`, không từ chối | Phải từ chối bằng phương pháp và τ đã khóa |
| 6 | Xác suất hiển thị là softmax T = 1 | Nên hiển thị xác suất đã hiệu chỉnh softmax(z / T) với T của checkpoint |
| 7 | Nhãn phương pháp `"KNN (k=10)"`, `"Energy Score"` | Dùng đúng tên notebook: `MSP`, `MaxLogit`, `Energy`, `Mahalanobis`, `KNN` |
| 8 | `src/preprocessing.py` xoay ảnh theo EXIF; notebook **không** xoay | Phải thống nhất (mục 3, dòng tiền xử lý) |
| 9 | Ảnh có kênh alpha: app chỉ `convert("RGB")` | Notebook ghép alpha lên nền xám 128 (`read_rgb`) |

## 3. Việc cần sửa
| Phần | Làm gì |
|---|---|
| Dựng mô hình | Dựng lại đúng như `CatClassifier` của notebook: `resnet50` = `nn.Sequential(*list(net.children())[:-2])`; `mobilenet_v2`, `efficientnet_b0` = `net.features`; `cnn_baseline` = 3 khối Conv3×3 (32/64/128) [+ BatchNorm nếu `config["baseline_batchnorm"]`] + ReLU + MaxPool. Sau đó `AdaptiveAvgPool2d(1)` → đầu phân loại: `Dropout(p) → Linear(d, 12)` nếu `head_hidden = 0`, hoặc `Dropout(p) → Linear(d, h) → ReLU → Dropout(p) → Linear(h, 12)`. Tên module phải là `features`, `pool`, `head` để `load_state_dict(strict=True)` chạy được. Dựng với `weights=None` (không tải trọng số ImageNet) |
| Tiền xử lý | Giống hệt notebook: `read_rgb` (alpha → ghép nền xám 128 → RGB) → `Resize(256, bilinear, antialias=True)` → `CenterCrop(224)` → `/255` → chuẩn hóa ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). Khi **kiểm khớp** (mục 6) phải **tắt EXIF**. Khi chạy thật cho người dùng: đề xuất **bật** xoay EXIF (ảnh điện thoại thường có cờ xoay), ghi rõ trong Ch.5 là khác với lúc huấn luyện |
| Điểm từ chối | Tính đủ 5 điểm theo đúng công thức mục 4, T = 1 |
| Quyết định | `accepted` nếu điểm của phương pháp đã khóa ≥ τ của phương pháp đó; ngược lại `rejected` (không trả `top_label`). Hiển thị cả 5 điểm và 5 τ ở chế độ nâng cao |
| Xác suất | Top-5 theo softmax(z / T), T đọc từ `calibration.json` |
| Grad-CAM | Giữ cách hiện tại: tầng ra cuối của `features`, lớp đích = lớp dự đoán, gradient theo logit, ReLU, nội suy bilinear 224×224, min–max từng ảnh |
| Thông tin mô hình | Trả `model_name`, `stage`, `seed` (trong `config["training_seeds"]` không có; lấy từ `run_id`), `run_id`, SHA-256 của `checkpoint.pt`, phương pháp từ chối, τ, T |
| Cấu hình | Không đặt cứng đường dẫn hay phương pháp: đọc tất cả từ một thư mục `model_package/` (mục 7) |

## 4. Định dạng file của notebook
`checkpoint.pt` (`torch.load(..., weights_only=True)`):
| Khóa | Nội dung |
|---|---|
| `model_name` | `resnet50` / `mobilenet_v2` / `efficientnet_b0` / `cnn_baseline` |
| `state_dict` | Trọng số (tên module `features.*`, `head.*`) |
| `class_mapping` | `{tên giống: chỉ số}` 12 lớp |
| `config` | CONFIG lúc huấn luyện: `dropout`, `head_hidden`, `baseline_batchnorm`, `image_size`, `resize_size`, … |
| `stage`, `epoch`, `run_id`, `split_id`, `git_commit`, `validation` | Thông tin truy vết |

`thresholds.json`: `T1` = τ của 5 phương pháp (T = 1), dùng để quyết định; `T_star` chỉ cho phân tích, app không dùng. Ví dụ checkpoint lượt 13/11: `"KNN": -1.0221926071848497`.

`calibration.json`: `temperature` (ví dụ 0.6049…).

`distance_parameters.npz`: `means` (12 × d), `precision` (d × d), `shrinkage`, `bank` (đặc trưng train đã chuẩn hóa L2, n × d), `k` (= 10).

`locked_selection.json` (thư mục study): `deployment.stage_dir` (đường dẫn checkpoint triển khai trên HF), `deployment.checkpoint_sha256`, `rejection.method` (ví dụ `KNN`), `rejection.threshold`, `rejection.temperature`.

Công thức điểm (z = logits, h = đặc trưng sau GAP, cùng quy ước "lớn hơn = giống dữ liệu huấn luyện"):
| Phương pháp | Điểm |
|---|---|
| MSP | max softmax(z) |
| MaxLogit | max z |
| Energy | logsumexp(z) |
| Mahalanobis | −min_c (h − μ_c)ᵀ P (h − μ_c), P = `precision`, chặn dưới 0 trước khi đổi dấu |
| KNN | −‖u − b_(k)‖, u = h/‖h‖, b_(k) = láng giềng gần thứ k trong `bank` (khoảng cách Euclid) |

## 5. Thử trước với checkpoint lượt 13/11
Lấy từ HF (`catbreed-id/catbreed-experiments`, cần quyền đọc org), thư mục `runs/20261113_145326_efficientnet_b0_s42_df1edde1/frozen/`: `checkpoint.pt`, `thresholds.json`, `calibration.json`, `distance_parameters.npz`, `val.npz`, `validation_scores.csv`. Phương pháp KNN, τ = −1.0221926071848497, T = 0.6049425684624492.
Chỉ dùng để **thử code**; không đưa số hay ảnh chụp từ checkpoint này vào báo cáo.

## 6. Kiểm khớp với notebook (bắt buộc trước khi chụp ảnh)
1. Lấy 20 ảnh validation có trong `validation_scores.csv` (cột `image_id`), đọc ảnh gốc từ Oxford, **tắt EXIF**.
2. Chạy qua app (gọi hàm Python trực tiếp, không qua HTTP), so với notebook:
   - logits: so với `val.npz["logits"]` cùng `image_id` → sai khác tuyệt đối lớn nhất < 1e-4.
   - 5 điểm: so với `validation_scores.csv` → sai khác < 1e-4 (Mahalanobis có thể cần dung sai tương đối 1e-6 vì giá trị lớn).
3. Ghi kết quả vào một file `app_consistency.csv` (để đưa vào Phụ lục).
Nếu lệch: kiểm theo thứ tự tiền xử lý → tên module/đầu phân loại → công thức điểm.

## 7. Khi có checkpoint v2 (~21:00)
1. Mở `locked_selection.json` của lượt official v2-tối-ưu (Claude/Hậu gửi đường dẫn). Nếu v2-tối-ưu không xong, dùng lượt v2-mặc-định.
2. Tải về `model_package/`: `checkpoint.pt`, `thresholds.json`, `calibration.json`, `distance_parameters.npz` **của đúng `deployment.stage_dir`**, và `locked_selection.json`.
3. Kiểm SHA-256 của `checkpoint.pt` khớp `deployment.checkpoint_sha256`; phương pháp và τ trong app khớp `rejection.method`, `rejection.threshold`.
4. Chạy lại kiểm khớp (mục 6) với `val.npz`, `validation_scores.csv` của checkpoint v2.

## 8. Ảnh chụp và số đo cho Ch.5
| Mục | Nội dung |
|---|---|
| Ảnh chụp | (1) mèo đúng giống, hiện Top-5 + Grad-CAM; (2) ảnh chó bị từ chối; (3) ảnh kết cấu (DTD) bị từ chối; (4) ảnh FeliBreed (mèo ngoài 12 giống) – ghi đúng kết quả, dù được nhận hay bị từ chối; (5) một ca sai (nếu có) |
| Chọn ảnh | Lấy từ **validation** hoặc ảnh người dùng tự chụp; không lấy từ tập test để chọn ảnh đẹp |
| Độ trễ | Trên **một máy cố định**, ghi CPU/GPU, RAM, hệ điều hành; chạy 50 ảnh, bỏ 5 lần đầu (khởi động), báo trung vị và phân vị 95%, tách: tiền xử lý, suy luận, điểm từ chối, Grad-CAM |
| Dung lượng | Kích thước `checkpoint.pt` và `distance_parameters.npz` (ngân hàng KNN cũng phải tải khi chạy) |

## 9. Không làm
- Không tự chọn lại phương pháp từ chối hay τ trong app.
- Không dùng τ, T, ngân hàng KNN của checkpoint khác với checkpoint đang nạp.
- Không đưa số của checkpoint lượt 13/11 vào báo cáo.
