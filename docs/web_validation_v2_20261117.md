# Nghiệm thu web v2 theo hai tầng — 17/11/2026

## Kết luận

Theo chỉ đạo bổ sung của Hậu: tầng 1 đạt 238/238. Tầng 2 chỉ báo cáo, không chấm đạt/không đạt: 0/238 ảnh đổi top-1; 1/238 ảnh đổi quyết định KNN (Bombay_162).

Notebook tính batch 32, batch cuối 14, TF32 bật cho convolution. App tính từng ảnh, TF32 tắt. Giữ nguyên checkpoint, τ, T, app/core và tham chiếu.

## Gói triển khai

- EfficientNet-B0 / finetuned / seed 42 / epoch tốt nhất 24.
- Run: 20261117_021310_efficientnet_b0_s42_8b69c4e9.
- SHA checkpoint: 6cc4bf72641b696b6765ef73a6f772cb65d5b28340088bf44e3b4ebdaf7bd586.
- KNN τ=-1.1167204244629498; T=0.7957721604899282.
- checkpoint.pt: 16,390,595 byte; distance_parameters.npz: 21,968,575 byte.
- Máy megalab-HP-Z8-G4-Workstation, GPU RTX A6000; Python 3.13.18, PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, CUDA 13.2.
- Thư mục riêng trên VPS: /mnt/thanhkhoi/catbreed-webapp-20261116-8295e721.
- API v2: port 18001, model_package_v2. Port 18000/package mặc định trên VPS là gói 16/11, không dùng cho v2.

## Xác minh SHA qua API

Giữ nguyên app theo chỉ đạo. `/health` hiện trả `is_mock=false` nhưng chưa
hiển thị SHA; SHA v2 được xác minh bằng hash SHA-256 của `checkpoint.pt`,
đối chiếu với `thresholds.json` và `locked_selection.json`, sau đó xác nhận
lại trong metadata phản hồi `/predict`. Không sửa `/health` chỉ để thêm SHA.

## Tầng 1 — kiểm code khớp notebook

Dùng model, preprocessing và scoring của adapter app. Chế độ batch32/batch cuối14, cuDNN convolution TF32=True, matmul TF32=False. Không sửa notebook hoặc core.

Giữ dung sai tuyệt đối <1e-4; Mahalanobis giữ thêm điều kiện tương đối <1e-6 theo OR như bàn giao gốc. Thực tế toàn bộ sai số đều đạt ngưỡng tuyệt đối.

| Đại lượng | Sai số tuyệt đối lớn nhất |
|---|---:|
| Logits | 0 |
| Features | 0 |
| MSP | 0 |
| MaxLogit | 0 |
| Energy | 0 |
| Mahalanobis | 1.3642420526593924e-12 |
| KNN | 4.440892098500626e-16 |

Kết quả: 238/238. 20 ảnh đầu cũng đạt 20/20 trong đúng các batch notebook.

Bằng chứng: [tier1.csv](evidence/v2/two_tier_20261117/tier1.csv), [summary](evidence/v2/two_tier_20261117/tier1.summary.json).

## Tầng 2 — ảnh hưởng chế độ phục vụ, chỉ báo cáo

Gọi DeploymentClassifier.predict_raw từng ảnh, float32, TF32=False cho convolution và matmul, exif_transpose=True như API. So sánh theo image_id với val.npz và validation_scores.csv. Không dùng tầng này để chấm đạt/không đạt.

| Đại lượng | Kết quả |
|---|---:|
| Logits: sai số tuyệt đối lớn nhất | 0.01070261001586914 |
| Logits: trung vị trên 238×12 sai số tuyệt đối | 0.0019298791885375977 |
| Logits: trung vị của sai số lớn nhất từng ảnh | 0.004977703094482422 |
| KNN: sai số tuyệt đối lớn nhất | 0.00047434395949952357 |
| KNN: trung vị sai số tuyệt đối | 0.00008630545822463365 |
| Ảnh đổi top-1 | 0/238 |
| Ảnh đổi quyết định KNN | 1/238 |
| min khoảng cách KNN notebook tới τ | 0.00010954112248473535 |
| min khoảng cách KNN app tới τ | 0.000006030191209926983 |

Ảnh sát ngưỡng nhất ở cả hai chế độ là Bombay_162. Notebook KNN=-1.116610883340465 ≥ τ nên chấp nhận; app KNN=-1.1167264546541598 < τ nên từ chối. Sai số 0.00011557131369466234 lớn hơn biên notebook 0.00010954112248473535, đủ lật quyết định. Top-1 vẫn là Bombay. Giữ nguyên kết quả này.

Bằng chứng: [tier2_app.csv](evidence/v2/two_tier_20261117/tier2_app.csv), [summary](evidence/v2/two_tier_20261117/tier2_app.summary.json). NPZ cùng tên giữ logits/KNN thô để kiểm lại thống kê.

## Độ trễ app thật

Đã đo lại 50 lượt, bỏ 5 lượt đầu, thống kê 45 lượt. App từng ảnh, TF32 tắt. Đo sau giải mã PIL; không gồm upload, đọc đĩa, JSON hoặc render trình duyệt. PNG/Base64 nằm trong hậu xử lý.

| Giai đoạn | Median ms | P95 ms |
|---|---:|---:|
| Tiền xử lý | 56.4559 | 84.0119 |
| Suy luận | 9.8806 | 10.4581 |
| Grad-CAM | 2.4799 | 2.5501 |
| Điểm từ chối | 4.7367 | 5.0899 |
| Hậu xử lý/PNG/Base64 | 80.1250 | 110.8219 |
| Tổng | 153.4363 | 212.0182 |

Bằng chứng: [latency_summary.json](evidence/v2/two_tier_20261117/latency_summary.json), latency_raw.csv và latency_measured.csv cùng thư mục. Cờ chapter5_ready=false trong output benchmark là nhãn mặc định của công cụ đo độc lập; kết luận hai tầng được ghi ở tài liệu này.

## Năm ca minh chứng

Đã gọi lại API v2 port 18001 cho đúng 5 input. Ca id_correct.jpg dùng Abyssinian_179, SHA 84c83acc06eaf0a597a43d8752a9b341f2657b4ba7fdd532a6bb12f45a78fea0. Không dùng inputs/cases/id_correct.jpg còn sót từ bundle cũ (Abyssinian_101); giữ nguyên file input cũ.

| Ca | Status | Top label | KNN |
|---|---|---|---:|
| id_correct.jpg | accepted | Abyssinian | -0.9979786196 |
| dog.jpg | rejected | null | -1.2155317237 |
| dtd_val_sample.jpg | rejected | null | -1.3661151814 |
| felibreed.jpg | accepted | Ragdoll | -1.0595768519 |
| id_error.jpg | rejected | null | -1.1172601320 |

10 ảnh chụp v2 đã có trong docs/evidence/v2 được giữ lại; không chụp mới ở lượt hai tầng. Đã xem ảnh tổng quan/chi tiết ca ID và đối chiếu cả 5 response mới: status/top/KNN khớp bằng chứng đã lưu. Hash 10 ảnh và hash input 5 ca nằm trong [screenshots_recheck_manifest.json](evidence/v2/screenshots_recheck_manifest.json).

Bằng chứng HTTP: [cases_summary.json](evidence/v2/two_tier_20261117/cases/cases_summary.json). Response đúng ca ID là id_correct_required179_result.json. File id_correct_result.json trong thư mục kiểm mới thuộc lần phát hiện bundle cũ, không dùng cho bảng 5 ca.

## Bảng tổng hợp gửi Hậu

- SHA checkpoint: 6cc4bf72641b696b6765ef73a6f772cb65d5b28340088bf44e3b4ebdaf7bd586.
- Tầng 1: 20/20 ảnh đầu; logits/features/KNN/cả 5 điểm đạt 238/238; không có ảnh lệch ở chế độ notebook.
- Tầng 2: top-1 khác 0/238; KNN đổi quyết định 1/238; sai số và biên tới τ được ghi đầy đủ phía trên.
- Độ trễ từng giai đoạn và tổng: bảng phía trên, đúng 50/5/45 lượt.
- Dung lượng checkpoint / distance_parameters: 16,390,595 / 21,968,575 byte.
- Kết quả 5 ca: bảng phía trên; 10 ảnh tại docs/evidence/v2/.
- Giữ app từng ảnh, TF32 tắt; không sửa checkpoint, τ, T hoặc tham chiếu.

## Lệnh tái lập trên VPS

Chạy tại thư mục riêng của Vỹ; đặt PYTHONPATH=runtime_api:., OMP_NUM_THREADS=1 và MKL_NUM_THREADS=1. Dùng Python /mnt/thanhkhoi/catbreed-id/.venv/bin/python đã có, chỉ đọc môi trường này.

    python scripts/check_notebook_reference_batches.py --package model_package_v2 --images inputs/oxford --output outputs/two_tier_20261117/tier1.csv
    python scripts/report_serving_mode_effect.py --package model_package_v2 --images inputs/oxford --output outputs/two_tier_20261117/tier2_app.csv
    python scripts/benchmark_official_web.py --package model_package_v2 --images inputs/oxford --device cuda --expected-user thanhkhoi --output outputs/two_tier_20261117

## Các mục bàn giao còn lại

- Commit/push báo cáo, script và bằng chứng mới lên feature/thezyx; không đưa model_package hoặc thay đổi không liên quan vào commit.
- `/health` trả `is_mock=false` nhưng chưa hiển thị SHA. SHA đúng đã được xác minh qua metadata `/predict` và hash package; giữ nguyên app, chờ Hậu xác nhận cách chứng minh thay thế này.
- Gửi kết quả cho Hậu; chat này chưa gửi tin nhắn thay người dùng.
- Số liệu trước v2 đã lưu riêng ở outputs/study_1611_pre_v2 và docs/evidence/study_1611_pre_v2, không dùng cho báo cáo v2.

Báo cáo được ghi UTF-8; không đổi dữ liệu kiểm thử trong lần sửa dấu tiếng Việt.
