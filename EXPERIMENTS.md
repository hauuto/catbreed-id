# Hồ sơ thực nghiệm CatBreedID

## Mỗi lượt chạy

- Có run_id duy nhất, được tạo tự động.
- Thư mục local: outputs/<run_id>/
- Thư mục Hugging Face: runs/<run_id>/
- Upload lại dùng nguyên run_id.
- Lượt chạy mới dùng run_id mới.

## Các tệp chung

- info.json: run_id, người chạy, thời gian, commit GitHub,
  phiên bản dữ liệu, split_id, seed huấn luyện và trạng thái.
- config.yaml: cấu hình thực tế đã dùng.
- history.csv: metrics theo epoch, có cột stage.
- class_mapping.json: ánh xạ chỉ số lớp sang tên giống.
- requirements_freeze.txt: phiên bản thư viện của môi trường chạy.

## Checkpoint theo giai đoạn

Mỗi giai đoạn có thư mục riêng:

- frozen/: giai đoạn đóng băng backbone.
- finetuned/: giai đoạn tinh chỉnh backbone.
- scratch/: CNN huấn luyện từ đầu.

Mỗi thư mục giai đoạn gồm:

- checkpoint.pt: checkpoint được chọn bằng validation.
- metrics.json: metric lựa chọn, giá trị và epoch tương ứng.
- predictions.csv: thêm khi cần lưu dự đoán theo mẫu.

Chỉ tạo các thư mục giai đoạn thực sự áp dụng cho lượt chạy.
Giữ nguyên checkpoint frozen khi chuyển sang finetuned.

## Đồng bộ

- Lưu checkpoint và metrics xong mới gọi upload.
- Tối thiểu upload cuối mỗi giai đoạn và cuối lượt.
- Upload lỗi thì giữ file để thử lại.
- Chỉ báo "Đã đồng bộ" sau khi upload thành công.
- Không xóa kết quả của lượt trước.

## Trạng thái lượt chạy

- running: đang thực hiện.
- completed: đã hoàn thành tính toán.
- failed: gặp lỗi; lưu thêm error.txt nếu bắt được lỗi.

Trạng thái tính toán và trạng thái đồng bộ là hai thông tin riêng.
Lượt completed vẫn có thể chưa đồng bộ nếu upload thất bại.

## Dữ liệu

- Cách chia train/validation hiện chưa chốt.
- Khi chốt, lưu danh sách chia tập dùng chung và gán split_id.
- Mỗi lượt chính thức phải ghi đúng split_id đã sử dụng.