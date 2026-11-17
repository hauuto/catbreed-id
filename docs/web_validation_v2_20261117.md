# Web validation v2 — 17/11/2026

## Trạng thái

Gói v2 đã được xác minh và chạy trên thư mục riêng của Vỹ trên máy `thanhkhoi`.
Checkpoint và khóa triển khai khớp nhau. Tuy nhiên, kiểm khớp số học với
`val.npz`/`validation_scores.csv` **chưa đạt**, nên các số liệu latency và ảnh
web dưới đây là bằng chứng thực tế của gói v2, không được xem là đã qua cổng
nghiệm thu Chương 5 cho đến khi Hậu xác định nguyên nhân lệch tham chiếu.

## Gói model

- Run: `20261117_021310_efficientnet_b0_s42_8b69c4e9`
- Model/stage/seed: `efficientnet_b0` / `finetuned` / `42`
- Checkpoint SHA-256:
  `6cc4bf72641b696b6765ef73a6f772cb65d5b28340088bf44e3b4ebdaf7bd586`
- Rejection: KNN
- Threshold: `-1.1167204244629498`
- Temperature: `0.7957721604899282`
- `checkpoint.pt`: 16,390,595 bytes
- `distance_parameters.npz`: 21,968,575 bytes

## Môi trường VPS

- Hostname: `megalab-HP-Z8-G4-Workstation`
- GPU: NVIDIA RTX A6000, 49,140 MiB
- PyTorch: `2.15.0+cu132`
- CUDA runtime: `13.2`
- torchvision: `0.29.2+cu132`
- Python: `3.13.18`
- Device/precision: CUDA / float32

## Kiểm khớp

Tiêu chí giữ nguyên: logits và score tuyệt đối `< 1e-4`; Mahalanobis được
thêm điều kiện tương đối `< 1e-6` theo phép OR. Không nới dung sai.

| Bộ | Logits | MSP | MaxLogit | Energy | Mahalanobis | KNN | Cả 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20 ảnh bắt buộc | 0/20 | 9/20 | 1/20 | 0/20 | 0/20 | 11/20 | 0/20 |
| 238 ảnh validation | 0/238 | 142/238 | 8/238 | 3/238 | 0/238 | 136/238 | 0/238 |

Sai khác bắt đầu từ `Abyssinian_101`; toàn bộ 238 ảnh có logits vượt dung sai.
Max abs trên 238 ảnh: logits `0.0107026100`, MSP `0.0016804786`,
MaxLogit `0.0081534386`, Energy `0.0077732437`, Mahalanobis `3.6674880`,
KNN `0.0004743439`. Danh sách chi tiết nằm trong
[all_validation.csv](evidence/v2/all_validation.csv) và
[summary](evidence/v2/all_validation.summary.json).

## Độ trễ

Đúng 50 lượt, bỏ 5 lượt đầu, thống kê 45 lượt còn lại. Đo sau khi ảnh đã
được giải mã; không gồm upload, đọc đĩa, JSON và render trình duyệt.

| Giai đoạn | Median (ms) | P95 (ms) |
|---|---:|---:|
| Tiền xử lý | 55.1051 | 85.4536 |
| Suy luận | 10.1769 | 10.6588 |
| Grad-CAM | 2.5016 | 2.5817 |
| Điểm từ chối | 4.7151 | 5.0572 |
| Hậu xử lý/PNG/Base64 | 80.5653 | 111.4838 |
| **Tổng** | **154.7340** | **216.1719** |

Chi tiết: [latency_summary.json](evidence/v2/latency_summary.json),
[latency_raw.csv](evidence/v2/latency_raw.csv),
[latency_measured.csv](evidence/v2/latency_measured.csv).

## Năm ca minh chứng

| Ca | Kết quả app | Top label | KNN |
|---|---|---|---:|
| Ảnh đúng (`id_correct.jpg`) | accepted | Abyssinian | -0.9979786 |
| Chó (`dog.jpg`) | rejected | null | -1.2155317 |
| DTD (`dtd_val_sample.jpg`) | rejected | null | -1.3661152 |
| FeliBreed (`felibreed.jpg`) | accepted | Ragdoll | -1.0595769 |
| Ảnh sai (`id_error.jpg`) | rejected | null | -1.1172601 |

Mỗi ca có ảnh tổng quan và chi tiết trong
[evidence/v2](evidence/v2/). Kết quả được lưu trong
[cases_summary.json](evidence/v2/cases_summary.json) và các JSON cùng tên.

## Tách số liệu cũ

Kết quả trước v2 ngày 16/11 đã chuyển sang
`outputs/study_1611_pre_v2/` và `docs/evidence/study_1611_pre_v2/`.
Các số liệu đó không được dùng cho báo cáo v2 hoặc Chương 5.
