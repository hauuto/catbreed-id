# Nghiệm thu module core với checkpoint 13 tháng 11

Cập nhật sau kiểm adapter: đã xử lý thiếu truyền method vào decide; cả gói thử và official v2 đạt 20/20 qua adapter app, 15 test hồi quy đạt không skip và 10 ca predict đầy đủ đã kiểm. Xem `docs/kiem_tra_adapter_1311_sau_sua.md` và `outputs/official_v2/adapter_trial_review/`.

Ngày thực hiện: 16/11/2026. Chạy trên tài khoản thanhkhoi, GPU RTX A6000, torch 2.15.0+cu132, float32, tắt TF32. Bộ thử đặt riêng tại `/mnt/thanhkhoi/catbreed-webapp-20261116-8295e721/trial_1311_review_20261116`.

## Nguồn dữ liệu và nhận dạng gói

- Checkpoint gốc: `C:\Users\vyx\Downloads\model_package_1311\checkpoint.pt`.
- SHA-256: `d90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052`, khớp thresholds.json.
- Run: `20261113_145326_efficientnet_b0_s42_df1edde1`, frozen.
- Hai file tham chiếu do người dùng tải về: `C:\Users\vyx\Downloads\d90bca48\val.npz` và `validation_scores.csv`. Không dùng bộ trong thư mục fbf86953.
- Đã ghép thành gói cục bộ đầy đủ tại `outputs/reference_packages/model_package_1311/`, kèm reference_manifest.json ghi nguồn, SHA và kích thước từng file. Không xác minh được SHA từ server private; nguồn gói tham chiếu dựa trên thư mục người dùng cấp và kiểm nội dung số.
- NPZ và CSV chứa cùng 238 ID, đúng thứ tự; logits có 12 lớp, điểm hữu hạn. Mọi file chuyển sang máy GPU được kiểm SHA trước khi chạy.

## Kết quả

1. `tests/test_catbreed_core.py`: **11 passed**, **0 skipped**, **0 failures/errors**. Đã đối chiếu cả log pytest và JUnit XML.
2. Kiểm trực tiếp module với 20 ảnh validation đầu tiên, exif=False: **20/20 đạt**. Logits và năm điểm đối chiếu NPZ/CSV. Không thay đổi dung sai; bộ test gốc còn giữ kiểm Mahalanobis theo sai số tương đối.
3. Tất cả kết quả model 13/11 chỉ phục vụ nghiệm thu code; không thay số benchmark, ảnh chụp hay kết quả official v2 trong Chương 5.

## Thay đổi để chạy offline và sửa lỗi phát hiện

- Bộ test nhận thêm CATBREED_TEST_PACKAGE và OXFORD_PETS_IMAGES để dùng gói và ảnh cục bộ; CATBREED_TEST_DEVICE chọn thiết bị. Giữ các phép assert gốc, không dùng token giả hay mock HF để bỏ qua kiểm dữ liệu thật.
- Khôi phục `DistanceScores.save` còn thiếu trong module đã tích hợp, theo định dạng gốc của module chung. Hai test cần lưu gói tổng hợp đã đạt sau sửa.
- `decide` hỗ trợ đối tượng Package thật có class_mapping và đối tượng cấu hình tối thiểu của test gốc. Với Package thật, nhãn vẫn lấy từ checkpoint.
- Pytest được cài riêng trong runtime_tests của thư mục thử, không cài vào môi trường dùng chung. Model official v2 và API web không bị thay gói.

## Bằng chứng local

- `outputs/official_v2/trial_1311/pytest.log`
- `outputs/official_v2/trial_1311/pytest.xml`
- `outputs/official_v2/trial_1311/trial_consistency_20.csv`
- `outputs/official_v2/trial_1311/trial_summary.json`
- `outputs/official_v2/trial_1311/acceptance.json`

## Cập nhật trạng thái bàn giao

Merge nhánh đã có commit d649df4. Hai mục còn thiếu trước đây là bộ test core đầy đủ và kiểm khớp checkpoint 13/11 nay đã có bằng chứng đạt. Các bản Word/PDF tạo trước bước này vẫn ghi chúng là chưa xác nhận; cần cập nhật báo cáo theo biên bản này trước khi nộp.

Hai ca Mahalanobis của kiểm mở rộng official v2 và giới hạn xác minh nguồn DTD/FeliBreed vẫn còn như đã công bố. Những thay đổi mới trong module/test cần được review và lưu vào Git theo quy trình nhóm. Không cần chụp lại năm ca chỉ vì các sửa chữa ở hàm save và đối tượng cấu hình test không thay đường tính kết quả official v2.
