# Kiểm tra thay đổi adapter gói thử 13 tháng 11

Đã rà soát phần bổ sung `method` vào DeploymentClassifier và CLI checker. Constructor đã chọn đúng phương pháp nhưng hàm predict chưa truyền phương pháp này vào decide. Vì vậy test chỉ nạp model có thể đạt, trong khi dự đoán đầy đủ của gói không có locked_selection vẫn lỗi.

## Sửa chữa

- Truyền `method=self.method` vào decide trong predict.
- Khi locked_selection tồn tại, chỉ lấy phương pháp trong khóa; không dùng tham số bên ngoài để bù cho khóa thiếu phương pháp.
- Bổ sung test gói không có khóa: không chỉ kiểm constructor mà gọi predict thật và kiểm KNN, trạng thái từ chối, top_label=null, năm xác suất và Grad-CAM.

## Kết quả trên thanhkhoi

- 15 tests passed, không skip, không failure/error. Gồm 11 test core, 2 test adapter và 2 test giao thức benchmark.
- Gói d90bca48 của ngày 13/11: kiểm logits và năm điểm qua adapter đạt 20/20, exif=False, CUDA float32.
- Gói official v2: kiểm qua adapter đạt 20/20. Truyền method=MSP vẫn chọn KNN từ locked_selection đúng yêu cầu.
- Mỗi gói chạy predict đầy đủ trên năm ca; tổng 10 ca, đủ Top-5, năm điểm, metadata và ảnh Grad-CAM 224×224. Các ca official giữ đúng kết quả đã ghi nhận.

Chạy trong thư mục thử riêng `/mnt/thanhkhoi/catbreed-webapp-20261116-8295e721/trial_1311_review_20261116`. Không thay gói model hay restart API web đang chạy. Những sửa chữa này không đổi công thức, tham số hay đường tính điểm của gói official đã khóa; không đo lại benchmark và không yêu cầu chụp lại ảnh. Dự đoán gói thử được gọi trực tiếp qua adapter, không tuyên bố đã chuyển API đang phục vụ sang gói 13/11.

## Bằng chứng

Trong `outputs/official_v2/adapter_trial_review/`:

- `pytest.log`, `pytest.xml`;
- `trial_adapter_20.csv`, `trial_adapter_20.summary.json`;
- `official_adapter_20.csv`, `official_adapter_20.summary.json`;
- `adapter_review.json` ghi cả 10 ca dự đoán và metadata.

Phần thiếu về đường kiểm gói 13/11 qua adapter nay đã được xử lý. Còn review và commit thay đổi mới, cập nhật Word/PDF cũ. Giữ nguyên hai giới hạn Mahalanobis mở rộng và nguồn ảnh như biên bản trước.
