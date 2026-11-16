# 4.3. Quy trình dò và chọn cấu hình (bản nháp)

Mục này mô tả quy trình nhóm sử dụng để dò siêu tham số và chọn cấu hình cho từng mạng xương sống (backbone).
Quy trình gồm bốn bước: dò theo lưới, kiểm tra độ ổn định theo seed, khóa cấu hình và thực thi tự động.
Mọi quyết định trong quy trình chỉ dựa trên tập validation và tập thăm dò.
Tập test và các tập ngoài phân phối (out-of-distribution, OOD) chỉ được đánh giá sau khi cấu hình đã được khóa.

## 4.3.1. Giai đoạn 1: dò từng yếu tố theo lưới

Ở giai đoạn 1, nhóm dò siêu tham số riêng cho từng backbone.
Mỗi lượt dò chỉ thay đổi một yếu tố so với cấu hình gốc.
Cách làm này giúp xác định tác động riêng của từng yếu tố.

Lưới dò gồm các yếu tố sau.
Tốc độ học (learning rate) được dò trên 9 điểm.
Tỷ lệ tham số được mở khóa để tinh chỉnh nhận ba mức: 25%, 50% và 100%.
Đầu phân loại có thêm một tầng ẩn 512 chiều (Dropout → Linear(d, 512) → ReLU → Dropout → Linear(512, 12)). Điểm này được so sánh với đầu phân loại mặc định chỉ gồm Dropout và một tầng tuyến tính 12 chiều.
Tỷ lệ dropout nhận hai giá trị: 0 và 0,5.
Với mô hình CNN cơ sở (baseline), nhóm dò kết hợp giữa việc có hoặc không dùng chuẩn hóa theo lô (Batch Normalization) và tốc độ học.

Tất cả các lượt ở giai đoạn 1 dùng chung seed 42.
Các lượt này chỉ được đánh giá trên tập validation.

## 4.3.2. Tiêu chí xếp hạng

Tiêu chí chính để xếp hạng cấu hình là F1 macro trên tập validation.
Hai cấu hình được xem là hòa khi F1 macro chênh nhau dưới 10⁻⁴.
Khi hòa, nhóm xét tiếp độ chính xác (Accuracy).
Nếu vẫn hòa, cấu hình đạt kết quả tốt nhất ở epoch sớm hơn được xếp trên.

Nhóm kiểm tra độ nhạy của thứ hạng theo hai cách.
Cách thứ nhất là xếp hạng lại theo Accuracy và theo NLL (negative log-likelihood).
Mức tương đồng giữa các bảng xếp hạng được đo bằng hệ số Kendall τ.
Cách thứ hai là tính lại thứ hạng sau khi bỏ 3 ảnh validation trùng với tập huấn luyện.

Tập validation có 238 ảnh.
Vì vậy, cặp cấu hình có Accuracy chênh nhau dưới 1/238 (≈ 0,42 điểm phần trăm, tức ít hơn một ảnh validation) được đánh dấu là không phân biệt được.

[Bảng 4.x. Kết quả giai đoạn 1 theo từng backbone: …]

## 4.3.3. Giai đoạn 2: kiểm tra độ ổn định theo seed

Ở giai đoạn 2, cấu hình hạng 1 của mỗi backbone được huấn luyện thêm với seed 43 và 44.
Điểm kết hợp D được đánh giá trên cả ba backbone, mỗi backbone với 3 seed.
Mô hình CNN cơ sở cũng được huấn luyện thêm với 2 seed.

[Bảng 4.x. Kết quả giai đoạn 2, trung bình và độ lệch theo 3 seed: …]

## 4.3.4. Khóa cấu hình

Cấu hình cuối cùng của mỗi backbone được chọn theo F1 macro trung bình trên 3 seed.
Khi hòa, nhóm áp dụng cùng luật như ở giai đoạn 1.
Nếu vẫn hòa, cấu hình có ít khóa ghi đè hơn so với cấu hình mặc định được ưu tiên.
Việc chọn được thực hiện bằng script `make_official_tuned.py`, không có thao tác chọn thủ công.

Sau khi chọn, lượt chính thức (official) huấn luyện lại cấu hình đã chọn với 3 seed.
Lượt này đồng thời chạy thí nghiệm tắt tăng cường dữ liệu (ablation).
Checkpoint dùng để triển khai được chọn dựa trên tập validation.
Phương pháp từ chối được chọn dựa trên tập thăm dò.
Kết quả lựa chọn được ghi vào tệp khóa `locked_selection.json` trước pha B. Pha B là pha đánh giá trên tập test và các tập OOD; pha này chỉ chạy sau khi tệp khóa đã được ghi.

[Bảng 4.x. Cấu hình đã khóa cho từng backbone: …]

## 4.3.5. Thực thi tự động

Toàn bộ quy trình được thực thi tự động theo các quy tắc đã đăng ký trước.
Người vận hành chỉ can thiệp trong hai trường hợp.
Trường hợp thứ nhất là khi có xung đột ở điểm kết hợp D.
Trường hợp thứ hai là khi lượt chạy gặp lỗi.
Cách làm này nhằm hạn chế khả năng người thực hiện điều chỉnh lựa chọn theo kết quả.