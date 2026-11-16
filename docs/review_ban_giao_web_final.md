# Review chốt bàn giao web official v2

Phạm vi review gồm adapter, API, module chung, test, script kiểm khớp, giao thức benchmark, phần chi tiết trên giao diện và báo cáo. Đã đọc diff của các file sửa và đối chiếu log nghiệm thu; không phát hiện lỗi chức năng mới sau bản sửa method đã được kiểm trên thanhkhoi.

## Bằng chứng kiểm tra

- 15 test hồi quy đạt, không skip: `docs/evidence/official_v2/adapter/pytest.xml` và `pytest.log`.
- Cả gói thử 13/11 và official v2 đạt 20/20 qua adapter; bộ khóa official vẫn ưu tiên KNN khi tham số thử là MSP.
- 10 lần predict đầy đủ trên hai gói có Top-5, năm điểm, Grad-CAM và metadata; official giữ kết quả mong đợi.
- Kiểm mở rộng official 236/238; hai ca Mahalanobis đã công bố, không nới dung sai.
- Đủ 10 ảnh chụp, đối chiếu SHA với bản gốc đã xem trước; được nhúng đầy đủ trong Word.
- Benchmark giữ đúng 50 lượt tổng, 5 làm nóng, 45 thống kê; không trộn kết quả gói 13/11 vào số đo official.
- Đã kiểm cú pháp Python, JavaScript và `git diff --check` trong quá trình chốt. Kết quả GPU được tái dùng từ phiên nghiệm thu đã lưu vì sửa cuối chỉ ở dependency, tài liệu, gói bằng chứng và dàn trang.

## Những điểm được hoàn thiện khi review

Đã khai báo SciPy trong requirements của API vì module chung dùng scipy.special. Thư mục model_package, môi trường notebook và thư mục pytest tạm được bỏ qua trong Git để tránh đưa gói model hay cache vào commit.

Nguồn số liệu và ảnh để dựng lại báo cáo nay ở `docs/evidence/official_v2/`. Report builder đọc từ bản sao này. Các cờ trạng thái cũ trong kết quả đo được giữ nguyên như lịch sử; trạng thái hiện tại được ghi trong tài liệu nghiệm thu và báo cáo cuối.

## Phạm vi lưu Git và giới hạn

Chỉ stage danh sách file thuộc bàn giao web, hai file Word/PDF cuối và bằng chứng liên quan. Không stage model, môi trường Python, bộ dữ liệu, các báo cáo tuần cũ hoặc những phần phân tích/huấn luyện chưa thuộc phạm vi review. Nội dung thay đổi sẵn có trong lời giới thiệu index.html được giữ nguyên và review cùng giao diện.

Hai ca Mahalanobis và mức xác minh nguồn ảnh DTD/FeliBreed tiếp tục là giới hạn được công bố. Không có xác nhận độ chính xác toàn bộ dữ liệu ngoại lai, độ trễ từ đầu đến cuối qua mạng, hoặc khớp từng bit giữa phần cứng. Việc commit cục bộ không đồng nghĩa đã push hoặc đã được người khác duyệt trên remote.
