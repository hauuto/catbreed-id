# Nội dung nháp Chương 5 – triển khai CatBreedID

Trạng thái: đã có bộ model, kiểm khớp bắt buộc, 5 ca đầu vào và số đo trên GPU `thanhkhoi`.
Đã xác minh đủ 10 ảnh chụp trình duyệt; xem `docs/kiem_tra_anh_minh_chung.md`.
Cần chèn các hình vào báo cáo và kiểm bản xuất cuối; đây vẫn là bản nháp.

## Mô hình triển khai

Ứng dụng gồm giao diện Flask tại cổng 5000 và API FastAPI tại cổng 8000. API nạp EfficientNet-B0
giai đoạn frozen, seed 42, checkpoint epoch 9 của lượt `20261116_060136_efficientnet_b0_s42_b8c9dba0`.
Lựa chọn mô hình, phương pháp từ chối và ngưỡng được đọc từ locked_selection.json. SHA-256 của
checkpoint và các file hiệu chỉnh được kiểm tra trước khi nạp.

Ảnh đầu vào được chuyển về RGB; alpha ghép với nền xám 128. Ảnh được resize cạnh ngắn 256,
crop giữa 224×224 và chuẩn hóa theo ImageNet. Khi dùng web, ảnh được xoay theo EXIF để hỗ trợ
ảnh điện thoại. Kiểm khớp với notebook tắt xoay EXIF; đây là khác biệt tiền xử lý cần nêu rõ.

API tính MSP, MaxLogit, Energy, Mahalanobis và KNN ở T=1. Quyết định sử dụng KNN với
τ=-1.0221926071848497; điểm ≥ τ được nhận. Trường top_label bằng null khi ảnh bị từ chối.
Xác suất Top-5 dùng nhiệt độ T=0.6049425684624492. Grad-CAM dùng gradient của logit lớp dự đoán
tại đầu ra cuối của features, ReLU và nội suy song tuyến về 224×224.

## Kiểm khớp và giới hạn số học

20 ảnh validation ban đầu đạt tiêu chí sai số theo bàn giao trên PyTorch 2.15.0+cu132,
torchvision 0.29.2+cu132, RTX A6000, float32 và tắt TF32. File
outputs/official_v2/thanhkhoi/module_review/mandatory_20.csv lưu từng ảnh,
sai số logits và cả 5 điểm. 20 ảnh đầu đều thuộc giống Abyssinian, do đó đã kiểm tra bổ sung
toàn bộ 238 ảnh validation để đánh giá phạm vi rộng hơn.

Toàn bộ 238 ảnh đạt dung sai logits và điểm KNN. Sai số logits tuyệt đối lớn nhất khoảng
6.44×10⁻⁶; sai số KNN lớn nhất khoảng 5.59×10⁻⁷. Cả 5 điểm đạt trên 236/238 ảnh;
Bengal_143 và Siamese_167 vượt nhẹ dung sai tương đối Mahalanobis 10⁻⁶, lần lượt khoảng
1.116×10⁻⁶ và 1.452×10⁻⁶. Giữ nguyên ngưỡng/dung sai và ghi nhận hai ca này trong Phụ lục.
Không diễn giải 236/238 là độ chính xác phân loại. Không suy rộng kết quả sang profile CPU.

## Các tình huống minh họa

Các ảnh được chọn từ validation hoặc chó trainval trong manifest near_val, không chọn từ test.
Nguồn, SHA-256 và phản hồi GPU có trong outputs/official_v2/thanhkhoi/module_review/review_summary.json.

- Abyssinian_179: được nhận đúng là Abyssinian. [Chèn ảnh Top-5 và Grad-CAM sau khi xác minh UI.]
- american_bulldog_138: bị từ chối, top_label=null. [Chèn ảnh giao diện.]
- Abyssinian_121: nhãn thật Abyssinian, hệ thống nhận thành Maine Coon. Đây là ca sai được giữ nguyên.
- DTD: người dùng tạo từ torchvision DTD val1[0], lớp banded theo annotation chính thức;
  hệ thống từ chối. Chưa so byte ảnh với bản gốc DTD.
- FeliBreed: người dùng xác nhận ảnh từ thư mục Local; hệ thống nhận sai thành Ragdoll.
  Chưa có tên file gốc để đối chiếu với kho ảnh.

## Hiệu năng triển khai

Đo trên Linux 6.8, CPU Xeon Gold 6138, RAM 67,098,542,080 byte, GPU RTX A6000 48 GB.
Suy luận dùng PyTorch 2.15.0 CUDA 13.2, một luồng CPU cho các bước tính toán CPU.
Chạy đúng 50 lượt tổng trên ảnh validation được lấy đều theo danh sách ID; bỏ 5 lượt đầu,
thống kê 45 lượt còn lại.

Tổng thời gian có median 169.75 ms và p95 226.83 ms. Các cặp median/p95 theo ms là:
tiền xử lý 61.78/89.51, suy luận 11.08/13.38, điểm từ chối 5.86/6.43,
Grad-CAM 2.91/3.88 và hậu xử lý 92.13/122.14. Không cộng các phân vị từng phần để suy ra phân vị tổng.

Phạm vi đo bắt đầu sau giải mã ảnh, kết thúc sau tạo PNG/Base64; không gồm đọc file,
upload, mã hóa JSON hay render trình duyệt. GPU được đồng bộ tại các mốc đo. Grad-CAM gồm
backward, chuẩn hóa và chuyển dữ liệu về CPU; mã hóa PNG tính vào hậu xử lý.
Đây không phải phép đo độ trễ từ thao tác người dùng đến khi trình duyệt hiện kết quả.

Checkpoint có kích thước 16,390,147 byte; ngân hàng/tham số khoảng cách có kích thước
21,994,386 byte. Tổng hai file là 38,384,533 byte (~38.38 MB). Cần cấp cả file KNN khi triển khai;
kích thước file không đồng nghĩa lượng RAM/VRAM sử dụng.

## Phụ lục cần đính kèm

outputs/official_v2/thanhkhoi/module_review/mandatory_20.csv và summary; kết quả mở rộng 238 ảnh và hai ca Mahalanobis;
latency_raw.csv, latency_measured.csv, latency_summary.json; review_summary.json và ảnh chụp giao diện được xác minh;
thông tin SHA-256/checkpoint/run và nguồn các ảnh minh họa.
