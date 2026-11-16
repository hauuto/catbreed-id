# Kiểm tra ảnh minh chứng — 16/11/2026

Đã xem trực tiếp toàn bộ 10 PNG trong `outputs/official_v2/screenshots/module_review/`. Kết luận: đủ 5 ca, mỗi ca có ảnh tổng quan và ảnh chi tiết; không cần chụp lại để đáp ứng nội dung bàn giao.

| Ca | Nội dung đã xác nhận trên ảnh chụp | Kết luận |
| --- | --- | --- |
| 01 mèo đúng | Đúng ảnh Abyssinian_179 thay thế không thấy watermark; Abyssinian 96%, KNN -0.9415 ≥ -1.0222; có Grad-CAM và bốn khả năng khác | Đạt |
| 02 chó | Đúng ảnh chó, trạng thái từ chối; KNN -1.2891 < -1.0222; có Top-5 tham khảo và Grad-CAM | Đạt |
| 03 DTD | Đúng ảnh vạch dọc; bị từ chối; KNN -1.3201 < -1.0222; có Top-5 và Grad-CAM | Đạt |
| 04 FeliBreed | Đúng ảnh người dùng báo từ Local; app nhận Ragdoll 37%; KNN -1.0043 ≥ -1.0222; giữ nguyên kết quả nhận sai theo nhãn nguồn | Đạt |
| 05 mèo sai | Đúng ảnh Abyssinian_121; app nhận Maine Coon 59%; KNN -0.9116 ≥ -1.0222; có Top-5 và Grad-CAM | Đạt |

Cả năm ảnh chi tiết đều nhìn thấy đủ 5 điểm và 5 ngưỡng, EfficientNet-B0 / frozen / seed 42, T = 0.6049425684624492, cuda / float32, run `20261116_060136_efficientnet_b0_s42_b8c9dba0` và SHA checkpoint đã khóa. Top-5 của ca nhận gồm lớp chính và bốn khả năng khác; không yêu cầu năm hàng riêng dưới lớp chính.

## Lưu ý khi đưa vào báo cáo

- Ảnh tổng quan hẹp khoảng 440–470 px; có thể dùng nguyên bản trong phần minh họa, nhưng trình bày ảnh chi tiết riêng để bảng và metadata đọc được. Không kéo giãn ảnh vượt độ phân giải gốc.
- Nhãn thật Abyssinian_121 và Local không thể suy ra từ tên dự đoán trên web; ghi nhãn nguồn trong chú thích hình.
- FeliBreed hiển thị 4578 ms. Giữ nguyên ảnh; đây là thời gian yêu cầu đơn lẻ, không phải median/p95 benchmark. Không chọn lại thời gian đẹp để thay thế phép đo chuẩn.
- Thống kê hiệu năng dùng `outputs/official_v2/thanhkhoi/module_review/latency_summary.json`: 50 lượt tổng, bỏ 5 đầu, 45 lượt đo; median 169.75 ms, p95 226.83 ms trên RTX A6000. Phạm vi đo là hàm predict sau giải mã ảnh, không bao gồm mạng và render trình duyệt.
- Kiểm bắt buộc 20/20; mở rộng 236/238. Hai ca Mahalanobis vượt dung sai là Bengal_143 và Siamese_167. Không gọi 236/238 là độ chính xác phân loại.
- DTD có quy trình tải do người dùng cung cấp và annotation val1 đối chiếu; chưa đối chiếu byte ảnh gốc. FeliBreed Local do người dùng xác nhận; chưa có tên file gốc trong kho để đối chiếu độc lập.

## Việc còn lại

1. Hoàn thiện Chương 5 từ bộ kết quả cuối; chèn 5 cặp ảnh, chú thích nhãn thật/kết quả, bảng hiệu năng, dung lượng và các giới hạn trên.
2. Đóng phụ lục kiểm khớp, benchmark, metadata/hash và nguồn ảnh; tránh lấy số liệu từ gói thử model_package_1311 hoặc các phiên đo lịch sử.
3. Kiểm bản báo cáo xuất cuối: ảnh/bảng đọc được, số liệu thống nhất và không còn chỗ trống chờ chèn hình. Sau đó mới đánh dấu báo cáo/bàn giao hoàn tất.

Bộ ảnh đã đủ để viết báo cáo. Các JSON của phiên đo gốc giữ `chapter5_ready=false` phản ánh trạng thái khi đo; tài liệu này ghi nhận bước xác minh ảnh sau đó, không sửa hồi tố bằng chứng gốc.
