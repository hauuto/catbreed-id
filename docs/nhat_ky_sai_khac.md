# Phụ lục D – Nhật ký sai khác so với giao thức thực nghiệm 19–20/10/2026

Bản nháp lập lúc 16/11/2026, khoảng 20:15. Giờ ghi theo múi giờ Việt Nam (UTC+7).

Cột "Thời điểm" ghi giờ quyết định khi có căn cứ. Nếu chưa có căn cứ, cột này ghi giờ commit đầu tiên thể hiện thay đổi. Giờ commit có thể muộn hơn giờ quyết định.

Ô trong ngoặc vuông `[…]` là thông tin còn thiếu, cần bổ sung trước khi đưa vào báo cáo.

## Bảng D.1. Các thay đổi so với giao thức

| # | Thời điểm | Thay đổi so với giao thức 19–20/10 | Lý do | Căn cứ (commit/file) | Đã xem kết quả test khi quyết định chưa |
|---|---|---|---|---|---|
| 1 | 13/11 (lượt chạy); 21:10 commit giao thức hai pha | Lượt chính thức 1 (Tài, GPU T4) đã tính kết quả test/OOD. | [lý do chạy lượt này] | `505700f` (13/11 21:10) | Có. Nhóm trưởng đã xem một phần kết quả |
| 2 | 16/11, khoảng 12:20 | Study `20261116_052028_study_99d77653` (Tài, giao thức cũ, GPU T4, có FeliBreed Mixed Breed + Local) được chạy, và Tài đã xem toàn bộ kết quả test/OOD. | [lý do chạy study] | Tên study (052028 UTC = 12:20 giờ VN) | Có. Tài xem trong khoảng [giờ – chờ câu trả lời việc 3] |
| 3 | 16/11, commit 11:50–12:26 | Thêm cấu hình dò siêu tham số theo kiến trúc (lưới chung). Mở khóa theo tỷ lệ tham số ở mức khối con. | [góp ý giảng viên / kết quả validation] | `73bc73e` 11:50, `e85b345` 12:05, `9d1e229` 12:26 | [cần xác nhận; `9d1e229` commit sau 12:20] |
| 4 | 16/11, commit 12:48 | Xây dựng dữ liệu từ archive gốc và kiểm MD5, bỏ bản sao trên Kaggle. Tiền xử lý huấn luyện đổi thành Resize 256 → RandomCrop 224, thay cho RandomResizedCrop. | [góp ý giảng viên / kết quả validation] | `c45e3bc` 12:48 | [cần xác nhận] |
| 5 | 16/11, khoảng 13:00–14:40 | Chốt v2: số epoch 30/30/80; tách pha A/B; chọn phương pháp từ chối bằng tập thăm dò, thay cho AUROC near-OOD trên test. | [góp ý giảng viên / kết quả validation] | [commit/file] | [cần xác nhận] |
| 6 | 16/11, quyết định khoảng 14:40 [cần xác nhận]; commit 16:28 | Thêm FeliBreed làm tập unseen_cat. Chỉ dùng Mixed Breed + Local; loại Tabby vì Tabby là kiểu lông, không phải giống. | Tabby là kiểu lông, không phải giống mèo | `5a96e2d` 16:28; `1e9cc07` 16:36 (tách thư mục gốc FeliBreed) | [chờ việc 3: Tài đề xuất trước hay sau khi xem kết quả study 12:20] |
| 7 | 16/11, commit 14:31 | D3/D4/D7: dùng điểm kết hợp D; so sánh letterbox với center crop trên validation, chỉ để phân tích độ nhạy. | [lý do] | `90db4b5` 14:31 | [cần xác nhận] |
| 8 | 16/11, khoảng 14:50 | Áp dụng điểm D cho cả 3 backbone. | [lý do] | [commit/file] | [cần xác nhận] |
| 9 | 16/11, commit 14:18–14:25; quyết định khoảng 15:05 [cần xác nhận] | Chuyển sang VPS A6000 và chạy lại toàn bộ, không trộn với kết quả trên T4. | [lý do] | `7f44dbb` 14:18, `6862802` 14:22, `4ad5731` 14:25 | [cần xác nhận] |
| 10 | 16/11, khoảng 17:00 | Hạn nộp đổi sang 17/11, nên bỏ phương án cắt giảm thí nghiệm. | Thay đổi hạn nộp | [thông báo/tin nhắn] | Không liên quan đến kết quả test |
| 11 | 16/11, 18:02 | FeliBreed dùng bản đã xoay theo EXIF (27/94 ảnh). | Ảnh gốc bị lệch hướng do thẻ EXIF | [commit/file] | [cần xác nhận] |
| 12 | 16/11, 18:16 (commit 18:18) | Ủy quyền Khôi chạy theo quy tắc đăng ký trước. Thêm script `make_official_tuned.py`, chọn cấu hình theo F1 macro validation trung bình 3 seed (42–44). | Chọn tự động theo luật đăng ký trước | `ddd2b54` 18:18 | Luật chọn chỉ dùng validation |
| 13 | 16/11, 18:32 | Token HF bị thu hồi, nên `sweep_tai` chưa đồng bộ lên HF. | Sự cố vận hành | [tin nhắn] | Không liên quan |
| 14 | 16/11, 19:32 | Chạy lại `khoi`, `vy`, `hau` ở các thư mục `_2`, cùng mã băm notebook `cb124dd2…`. | [lý do chạy lại] | `run.log` trong `work_*_2` | Không liên quan |
| 15 | 16/11, 19:56 | `sweep_khoi` (MobileNetV2, 15 điểm) chạy xong và đã đồng bộ lên HF. | Ghi nhận tiến độ | Log VPS | Không liên quan |

## D.2. Đánh giá ảnh hưởng của việc đã xem kết quả test

Các quyết định trong Bảng D.1 dựa trên góp ý của giảng viên hoặc kết quả trên tập validation. [Cần xác nhận từng dòng 3–9 dựa trên căn cứ nào.]

Trước khi khóa v2, nhóm đã xem kết quả test/OOD hai lần. Lần thứ nhất là lượt chính thức ngày 13/11: nhóm trưởng xem một phần kết quả. Lần thứ hai là study lúc 12:20 ngày 16/11: Tài xem toàn bộ kết quả.

Quyết định có nguy cơ bị ảnh hưởng nhiều nhất là việc loại Tabby khỏi tập unseen_cat. Study lúc 12:20 có dùng FeliBreed. Nếu Tài đề xuất loại Tabby sau khi đã xem kết quả study này, quyết định có thể đã bị ảnh hưởng bởi kết quả test. [Kết luận phụ thuộc câu trả lời việc 3.]

Nhóm áp dụng ba biện pháp để hạn chế ảnh hưởng này. Thứ nhất, quy tắc chọn cấu hình được đăng ký trước khi chạy thực nghiệm. Thứ hai, việc chọn cấu hình được thực hiện tự động bằng script và chỉ dựa trên validation. Thứ ba, Tài không tham gia vào bước chọn cấu hình.