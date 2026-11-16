# Kiểm tra web official v2 — 16/11/2026

**Trạng thái nghiệm thu cuối:** adapter gói thử đã truyền method tường minh vào decide; 15 test hồi quy đạt không skip, cả gói 13/11 và official v2 đạt 20/20 qua adapter, 10 ca predict đầy đủ đạt hợp đồng. Bộ ảnh đã đủ và Word/PDF được cập nhật theo kết quả này. Bản sao bằng chứng để lưu Git nằm trong `docs/evidence/official_v2/`. Các ghi chú chờ test, chờ ảnh hoặc thiếu tham chiếu bên dưới chỉ mô tả lịch sử trước nghiệm thu cuối.

## Kiểm thử gói thử 13/11

- Gói thử có checkpoint SHA-256 `d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`; các file `val.npz` và `validation_scores.csv` trong hai thư mục `fbf86953` và `d90bca48` có cùng SHA-256.
- Gói không có `locked_selection.json` được nạp qua adapter với `method="KNN"` và `exif=False`; 20/20 ảnh đạt logits cùng đủ năm điểm OOD.
- Bằng chứng: [CSV](../outputs/trial_d90bca48/trial_d90bca48_20_exif_false.csv) và [tóm tắt](../outputs/trial_d90bca48/trial_d90bca48_20_exif_false.summary.json). Đây là kiểm thử adapter, không dùng số liệu gói thử cho báo cáo official v2.

**Cập nhật nghiệm thu core và gói 13/11:** đã có merge d649df4; đã chạy bộ test core bằng nguồn tham chiếu cục bộ: 11 passed, không skip; kiểm khớp gói 13/11 đạt 20/20 trên thanhkhoi. Xem [biên bản và đường dẫn log](ket_qua_nghiem_thu_core_1311.md). Các ghi chú thiếu file/gói 13/11 ở dưới là lịch sử trước khi người dùng cung cấp hai thư mục theo SHA. Chưa cập nhật hồi tố báo cáo Word/PDF cũ.

**Cập nhật sau xác minh ảnh:** đã xem đủ 10 ảnh màn hình của 5 ca trong `outputs/official_v2/screenshots/module_review/`, tất cả đạt nội dung cần chụp. Xem [biên bản kiểm ảnh và việc còn lại](kiem_tra_anh_minh_chung.md). Bộ ảnh đã đủ để viết Chương 5; các ghi chú chờ ảnh và cờ `chapter5_ready=false` của phiên đo bên dưới là trạng thái lịch sử, không phải yêu cầu chụp thêm. Báo cáo cuối vẫn cần hoàn thiện và kiểm bản xuất.

## Gói triển khai

- Nguồn gói mới: `C:\Users\vyx\Downloads\EfficientNet-B0`; bản sao chạy trong `model_package/`.
- EfficientNet-B0, frozen, seed 42, epoch 9; run `20261116_060136_efficientnet_b0_s42_b8c9dba0`.
- SHA-256 checkpoint: `985e40d4036ae06845503687f77e976cb734dea63801c928ca970081b081d660`. Checkpoint, thresholds, calibration và distance parameters khớp khóa trong `locked_selection.json`.
- KNN ở T=1, nhận nếu điểm ≥ τ = -1.0221926071848497. Top-5 dùng nhiệt độ 0.6049425684624492. `top_label` bằng null khi từ chối.

## Kiểm khớp trên máy thanhkhoi

Máy `megalab-HP-Z8-G4-Workstation`, tài khoản `thanhkhoi`, NVIDIA RTX A6000; Linux 6.8, Xeon Gold 6138, RAM 67,098,542,080 byte, Python 3.13.18, torch 2.15.0+cu132, torchvision 0.29.2+cu132. Chạy CUDA float32, tắt TF32. Dữ liệu và mã được đối chiếu SHA-256 trước khi chạy trong thư mục riêng `/mnt/thanhkhoi/catbreed-webapp-20261116-8295e721`.

- Phiên tích hợp `src/catbreed_core`: 20 ảnh bắt buộc **20/20 đạt**. [CSV](../outputs/official_v2/thanhkhoi/module_review/mandatory_20.csv) và [tóm tắt](../outputs/official_v2/thanhkhoi/module_review/mandatory_20.summary.json).
- Toàn bộ 238 ảnh của phiên tích hợp: logits và KNN đạt 238/238; cả 5 điểm đạt **236/238**. `Bengal_143` và `Siamese_167` vượt dung sai Mahalanobis; giữ nguyên trong [CSV](../outputs/official_v2/thanhkhoi/module_review/all_validation.csv) và [tóm tắt](../outputs/official_v2/thanhkhoi/module_review/all_validation.summary.json).
- Tiêu chí: `|Δlogits| < 1e-4`, `|Δscore| < 1e-4`; Mahalanobis cho phép thêm `relative < 1e-6` theo phép OR. Đây là kiểm khớp số học, không phải độ chính xác phân loại.
- Kiểm khớp tắt xoay EXIF như notebook; web bật xoay EXIF. Alpha ghép nền xám 128.

## Năm ca minh chứng

[Manifest và kết quả HTTP của phiên đã kiểm lại](../outputs/official_v2/thanhkhoi/module_review/review_summary.json) lưu ảnh, nguồn, SHA-256, nhãn thật và quyết định. API đã được khởi động lại để nạp code tích hợp mới. Ảnh đầu vào ở `outputs/official_v2/gpu/cases/`:

- `id_correct.jpg`: Oxford validation `Abyssinian_179`, nhận đúng Abyssinian.
- `dog.jpg`: Oxford chó `american_bulldog_138`, bị từ chối.
- `dtd_val_sample.jpg`: người dùng báo tạo bằng `torchvision.datasets.DTD(split="val", partition=1)[0]`; annotation val1 chính thức ghi phần tử đầu `banded/banded_0006.jpg`. Bị từ chối. Chưa so byte ảnh với bản gốc DTD.
- `felibreed.jpg`: người dùng xác nhận lấy từ thư mục `Local` của FeliBreed. Ảnh được **nhận là Ragdoll**; giữ nguyên ca này. Chưa có tên file gốc để đối chiếu trực tiếp.
- `id_error.jpg`: Oxford validation `Abyssinian_121`, nhãn thật Abyssinian, app nhận sai Maine Coon.

Không chỉnh ngưỡng hay thay ảnh để ép từ chối. Ảnh chụp trình duyệt cho năm ca còn chờ người dùng. [Nguồn ngoại lai và mức xác minh](../outputs/official_v2/provenance/external_sources.json).

## Độ trễ và dung lượng trên thanhkhoi

Đúng **50 lượt tổng**, bỏ 5 lượt đầu làm nóng, thống kê **45 lượt** còn lại. Ảnh chọn đều từ validation. Đo `predict` sau giải mã ảnh; không gồm đọc file, upload, mã hóa JSON hay render trình duyệt. GPU được đồng bộ tại mốc đo. PNG/Base64 thuộc hậu xử lý. [50 lượt gốc](../outputs/official_v2/thanhkhoi/module_review/latency_raw.csv), [45 lượt đo](../outputs/official_v2/thanhkhoi/module_review/latency_measured.csv), [tóm tắt](../outputs/official_v2/thanhkhoi/module_review/latency_summary.json).

| Giai đoạn | Median (ms) | p95 (ms) |
| --- | ---: | ---: |
| Tiền xử lý | 61.78 | 89.51 |
| Suy luận | 11.08 | 13.38 |
| Grad-CAM | 2.91 | 3.88 |
| Điểm từ chối | 5.86 | 6.43 |
| Hậu xử lý/PNG/Base64 | 92.13 | 122.14 |
| **Tổng** | **169.75** | **226.83** |

Không cộng median từng phần để suy ra median tổng. `checkpoint.pt` 16,390,147 byte; `distance_parameters.npz` 21,994,386 byte; tổng 38,384,533 byte (~38.38 MB dung lượng file).
[CSV 50 lượt](../outputs/official_v2/thanhkhoi/module_review/latency_raw.csv), [45 lượt đo](../outputs/official_v2/thanhkhoi/module_review/latency_measured.csv), [tóm tắt](../outputs/official_v2/thanhkhoi/module_review/latency_summary.json).

Bộ `gpu/module_integration` và `thanhkhoi/results` được giữ làm lịch sử. **Bộ kết quả cuối sau sửa phép đo nằm trong `outputs/official_v2/thanhkhoi/module_review/`**: mandatory_20.csv (20/20), all_validation.csv (236/238), latency_raw.csv (50), latency_measured.csv (45), latency_summary.json và review_summary.json. Bảng trên dùng bộ này. Đã đồng bộ CUDA trước/sau suy luận; tổng đo trực tiếp từ đầu hàm đến hết mã hóa Base64, bao gồm quyết định và tạo ảnh giải thích. Không dùng số 153.23/213.79 ms của phép đo trước cho báo cáo cuối.

Đã đọc và kiểm SHA gói `C:\Users\vyx\Downloads\model_package_1311`: checkpoint khớp hash trong thresholds (`d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`). README của gói ghi rõ chỉ thử code ngày 13/11, không dùng báo cáo. Gói thiếu locked_selection, val.npz và validation_scores.csv; không thay gói official v2 hiện tại bằng gói này.

Hướng dẫn ảnh đầu vào, nội dung chụp và tên file: [hướng dẫn chụp minh chứng](huong_dan_chup_minh_chung.md).

Kết quả Windows RTX 3050 cũ dùng giao thức 5 lượt làm nóng **cộng** 50 lượt đo nên không dùng làm số đo bàn giao 50 lượt tổng. CSV cũ trong `outputs/official_v2/gpu/` được giữ làm lịch sử.

## Mở web để chụp minh chứng

Trong thư mục dự án trên Windows, mở hai cửa sổ PowerShell:

```powershell
# Cửa sổ 1: nối cổng 8000 tới API GPU thanhkhoi; để cửa sổ mở
.\scripts\connect_thanhkhoi_web.ps1
```

```powershell
# Cửa sổ 2: chỉ chạy giao diện Flask trên Windows
.\.venv\Scripts\python.exe -m flask --app app run --port 5000
```

Kiểm `http://127.0.0.1:8000/health` có `is_mock=false`, rồi mở `http://127.0.0.1:5000`. Chọn lần lượt năm ảnh ở trên; mỗi ca chụp tổng quan gồm đầu vào, trạng thái, Top-5, Grad-CAM và chụp phần “Chi tiết mô hình và 5 điểm từ chối”. Lưu vào `outputs/official_v2/screenshots/`; ghi rõ ảnh của phiên thanhkhoi. FeliBreed phải thể hiện kết quả Ragdoll hiện tại. Ảnh chụp Windows GPU cũ không phải minh chứng cho phiên thanhkhoi.

`chapter5_ready=false` cho đến khi kiểm đủ ảnh chụp và ghi hai giới hạn Mahalanobis cùng mức xác minh nguồn DTD/FeliBreed. Bản nháp chương 5 ở `docs/chapter5_webapp_draft.md`.

## Sai khác so với bàn giao bổ sung

`BanGiao_WebApp_Vy_BoSung_2026-11-16.md` yêu cầu app gọi `src/catbreed_core` từ nhánh `feat/inference-core`. Yêu cầu này đã được tích hợp và xác nhận bằng kiểm khớp 20/20, kiểm mở rộng 236/238 (hai sai khác Mahalanobis đã biết) và benchmark 50 lượt trên cùng máy. Phần chụp ảnh trình duyệt vẫn do người dùng thực hiện; `chapter5_ready=false` cho đến khi đủ ảnh và hoàn tất xác minh nguồn DTD/FeliBreed.

Ảnh mèo đúng đã đổi sang validation `Abyssinian_179` theo yêu cầu bỏ watermark. File hiện tại `gpu/cases/id_correct.jpg` là bản sao nguyên ảnh này; phản hồi GPU mới nằm ở `thanhkhoi/module_review/id_correct_replacement_http.json`. Các manifest/run_summary và benchmark lịch sử vẫn ghi ảnh Abyssinian_101 đúng với phiên đo cũ; không diễn giải chúng thành phép đo cho ảnh thay thế. Ảnh và kết quả cũ được lưu trong `gpu/cases/history_Abyssinian_101/`.
