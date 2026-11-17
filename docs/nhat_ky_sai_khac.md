# Phụ lục D – Nhật ký sai khác so với giao thức thực nghiệm 19–20/10/2026

Bản nháp lập lúc 16/11/2026, khoảng 20:15. Giờ ghi theo múi giờ Việt Nam (UTC+7).

Cột "Thời điểm" ghi giờ quyết định khi có căn cứ. Nếu chưa có căn cứ, cột này ghi giờ commit đầu tiên thể hiện thay đổi. Giờ commit có thể muộn hơn giờ quyết định.

Ô trong ngoặc vuông `[…]` là thông tin còn thiếu, cần bổ sung trước khi đưa vào báo cáo.

## Bảng D.1. Các thay đổi so với giao thức

| # | Thời điểm | Thay đổi so với giao thức 19–20/10 | Lý do | Căn cứ (commit/file) | Đã xem kết quả test khi quyết định chưa |
|---|---|---|---|---|---|
| 1 | 13/11 (lượt chạy); 21:10 commit giao thức hai pha | Lượt chính thức 1 (Tài, GPU T4) đã tính kết quả test/OOD. | Lượt đánh giá chính thức theo giao thức 19–20/10 | `505700f` (13/11 21:10) | Có. Nhóm trưởng đã xem một phần kết quả |
| 2 | 16/11, khoảng 12:20 | Study `20261116_052028_study_99d77653` (Tài, giao thức cũ, GPU T4, có FeliBreed Mixed Breed + Local) được chạy, và Tài đã xem toàn bộ kết quả test/OOD. | Chạy thử quy trình có thêm tập unseen_cat (FeliBreed), theo giao thức cũ | Tên study (052028 UTC = 12:20 giờ VN) | Có. Tài xem khoảng 18:30 ngày 16/11, sau khi mọi quyết định giao thức v2 (dòng 3–12) đã chốt |
| 3 | 16/11, commit 11:50–12:26 | Thêm cấu hình dò siêu tham số theo kiến trúc (lưới chung). Mở khóa theo tỷ lệ tham số ở mức khối con. | Giảng viên yêu cầu số liệu chứng minh cho các lựa chọn; bộ tham số chung có thể thiên vị một kiến trúc; cụm "2 khối cuối" ứng với 93,9% tham số ở ResNet50 nhưng 28,5% ở EfficientNet-B0 (đếm từ code) | `73bc73e` 11:50, `e85b345` 12:05, `9d1e229` 12:26 | Không. Các commit do phiên điều phối thực hiện; phiên này không có và không mở kết quả study 12:20. `9d1e229` sửa lỗi unf25 = unf50 phát hiện bằng test đơn vị |
| 4 | 16/11, commit 12:48 | Xây dựng dữ liệu từ archive gốc và kiểm MD5, bỏ bản sao trên Kaggle. Tiền xử lý huấn luyện đổi thành Resize 256 → RandomCrop 224, thay cho RandomResizedCrop. | Bảo đảm nguồn gốc dữ liệu (yêu cầu của nhóm trưởng: không dùng bản sao của thành viên); tiền xử lý huấn luyện dùng cùng cách cắt ảnh với tập đánh giá (rà code) | `c45e3bc` 12:48 | Không. Do phiên điều phối thực hiện, căn cứ từ code và yêu cầu về nguồn dữ liệu |
| 5 | 16/11, khoảng 13:00–14:40 | Chốt v2: số epoch 30/30/80; tách pha A/B; chọn phương pháp từ chối bằng tập thăm dò, thay cho AUROC near-OOD trên test. | Epoch: nhiều lượt 13/11 dừng đúng epoch cuối (đường validation). Tách pha A/B và chọn phương pháp từ chối bằng tập thăm dò: góp ý của giảng viên về việc truy cập tập test | `QuyTrinh_Sweep_v2_2026-11-16.md`; `configs/v2/*.json` | Không. Căn cứ là validation và góp ý của giảng viên |
| 6 | 16/11: (a) Tài đề xuất bỏ Tabby 11:12, nhóm trưởng đồng ý 11:50, commit 12:11; lượt 12:20 chỉ dùng Mixed Breed + Local (`b7d2748`); (b) đưa vào cấu hình v2 lúc 16:28 sau EDA của Tài (16:27) | Thêm FeliBreed làm tập unseen_cat. Chỉ dùng Mixed Breed + Local; loại Tabby vì Tabby là kiểu lông, không phải giống. | Tabby là kiểu lông, không phải giống mèo | `b7d2748`; `5a96e2d` 16:28; `1e9cc07` 16:36 (tách thư mục gốc FeliBreed); nguồn [41] (CFA) | Không. Quyết định có trước khi lượt 12:20 bắt đầu và trước khi Tài xem kết quả (18:30) |
| 7 | 16/11, commit 14:31 | D3/D4/D7: dùng điểm kết hợp D; so sánh letterbox với center crop trên validation, chỉ để phân tích độ nhạy. | Thiết kế thí nghiệm: D4 kiểm tra tương tác giữa các yếu tố; D7 là phân tích độ nhạy cho phép cắt vùng trung tâm | `90db4b5` 14:31 | Không. Căn cứ từ rà phương pháp |
| 8 | 16/11, khoảng 14:50 | Áp dụng điểm D cho cả 3 backbone. | Đủ tài nguyên GPU để kiểm tra tương tác cho cả 3 kiến trúc (quyết định của nhóm trưởng 14:49) | `scripts/make_stage2.py` (Khôi) | Không |
| 9 | 16/11, khoảng 14:15 (quyết định); commit 14:18–14:25 | Chuyển sang VPS A6000 và chạy lại toàn bộ, không trộn với kết quả trên T4. | Đủ tài nguyên để chạy lại toàn bộ trên cùng một loại phần cứng | `7f44dbb` 14:18, `6862802` 14:22, `4ad5731` 14:25 | Không |
| 10 | 16/11, khoảng 17:00 | Hạn nộp đổi sang 17/11, nên bỏ phương án cắt giảm thí nghiệm. | Thay đổi hạn nộp | [thông báo/tin nhắn] | Không liên quan đến kết quả test |
| 11 | 16/11, 18:02 | FeliBreed dùng bản đã xoay theo EXIF (27/94 ảnh). | 27/94 ảnh gốc có thẻ xoay EXIF; notebook không xoay theo EXIF nên sẽ đọc ảnh nằm nghiêng | `felibreed_rotation.csv` (Tài); danh sách SHA-256 trên VPS | Không. Quyết định của nhóm trưởng, căn cứ từ EDA |
| 12 | 16/11, 18:16 (commit 18:18) | Ủy quyền Khôi chạy theo quy tắc đăng ký trước. Thêm script `make_official_tuned.py`, chọn cấu hình theo F1 macro validation trung bình 3 seed (42–44). | Chọn tự động theo luật đăng ký trước | `ddd2b54` 18:18 | Luật chọn chỉ dùng validation |
| 13 | 16/11, 18:32 | Token HF bị thu hồi, nên `sweep_tai` chưa đồng bộ lên HF. | Sự cố vận hành | [tin nhắn] | Không liên quan |
| 14 | 16/11, 19:32 | Chạy lại `khoi`, `vy`, `hau` ở các thư mục `_2`, cùng mã băm notebook `cb124dd2…`. | Token HF mới; lượt trước dừng vì token bị thu hồi và vì HEAD trên VPS chưa khớp origin | `run.log` trong `work_*_2` | Không liên quan |
| 15 | 16/11, 19:56 | `sweep_khoi` (MobileNetV2, 15 điểm) chạy xong và đã đồng bộ lên HF. | Ghi nhận tiến độ | Log VPS | Không liên quan |

## D.2. Đánh giá ảnh hưởng của việc đã xem kết quả test

Các quyết định trong Bảng D.1 dựa trên góp ý của giảng viên, kết quả trên tập validation, rà code hoặc yêu cầu về nguồn dữ liệu. Căn cứ của từng dòng được ghi ở cột "Lý do".

Trước khi khóa v2, kết quả test/OOD đã được xem hai lần. Lần thứ nhất là lượt chính thức ngày 13/11: nhóm trưởng xem một phần kết quả. Lần thứ hai là study lúc 12:20 ngày 16/11: Tài xem toàn bộ kết quả vào khoảng 18:30 cùng ngày.

Lần xem thứ hai diễn ra sau khi mọi quyết định của giao thức v2 đã được chốt. Cụ thể, việc bỏ Tabby được đề xuất lúc 11:12 và commit lúc 12:11, trước khi lượt 12:20 bắt đầu. Lưới dò, quy tắc chọn và cách tách pha A/B được chốt trong buổi họp 13:00–14:40. Việc dùng bản FeliBreed đã xoay được chốt lúc 18:02, và quy tắc chọn tự động được chốt lúc 18:16. Sau 18:30, nhóm không thay đổi giao thức; các bước còn lại chỉ thực thi theo quy tắc đã chốt.

Nhóm áp dụng ba biện pháp để hạn chế ảnh hưởng còn lại. Thứ nhất, quy tắc chọn cấu hình được đăng ký trước khi chạy thực nghiệm. Thứ hai, việc chọn cấu hình được thực hiện tự động bằng script và chỉ dựa trên validation và tập thăm dò. Thứ ba, Tài không tham gia vào bước chọn cấu hình.
