# Quy tắc làm việc - CatBreedID

## 1. Quản lý code

- GitHub lưu toàn bộ code xử lý dữ liệu, train, inference và Flask.
- Nhánh main chứa code dùng chung đã được kiểm tra.
- Mỗi thay đổi thực hiện trên branch riêng.
- Đặt tên branch theo dạng feat/ten-noi-dung hoặc fix/ten-noi-dung.
- Tạo Pull Request vào main và có một thành viên khác review trước khi merge.
- PR ghi rõ thay đổi gì và đã kiểm tra bằng cách nào.
- Có thể chạy thử nghiệm từ branch riêng trước khi merge.

## 2. Chạy thực nghiệm

- Dùng runner mẫu của nhóm.
- Mỗi lượt có mã riêng và thư mục kết quả riêng.
- Ghi commit thực tế, cấu hình, người chạy và phiên bản dữ liệu.
- Nếu sửa code trên Kaggle để debug, đưa bản sửa về GitHub
  trước khi chạy thực nghiệm chính thức.

## 3. Lưu và đồng bộ kết quả

- Giữ riêng checkpoint đóng băng và checkpoint tinh chỉnh.
- Upload sau khi lưu checkpoint thành công, tối thiểu cuối
  giai đoạn đóng băng, cuối tinh chỉnh và cuối lượt.
- Chỉ báo "Đã đồng bộ" sau khi upload thành công.
- Khi upload lỗi, giữ kết quả và chạy lại bước upload.
- Lần chạy mới không ghi đè lần chạy cũ.
- Không xóa hồ sơ cũ chỉ vì có kết quả tốt hơn.
- Giữ hồ sơ và thông tin lỗi của những lượt thất bại.

## 4. Tài khoản và file

- Mỗi người dùng tài khoản và token riêng.
- Lưu token trong Kaggle Secrets hoặc môi trường local.
- Không commit token, mật khẩu, .env, dataset hoặc checkpoint lên GitHub.
- Checkpoint và kết quả thực nghiệm được lưu trên Hugging Face.
- Hậu quản lý backup.

## 5. Inference và ứng dụng

- Ứng dụng dùng checkpoint đã được nhóm lựa chọn.
- Ghi rõ phiên bản checkpoint trong cấu hình inference.
- Dùng đúng kiến trúc, tiền xử lý và ánh xạ nhãn đi kèm.
- Không tự thay model ứng dụng sau mỗi lượt train mới.