# Bằng chứng nghiệm thu web official v2

Bản sao các kết quả nhỏ để lưu Git và tạo lại báo cáo. Không chứa trọng số hoặc bộ dữ liệu huấn luyện.

- `deployment/`: kiểm mandatory 20/20, mở rộng 236/238, benchmark 50 lượt tổng bỏ 5 đầu trên RTX A6000. Các tệp phản ánh phiên đo gốc, giữ nguyên cờ trạng thái khi đo.
- `adapter/`: nghiệm thu cuối sau sửa method, 15 test không skip; gói 13/11 và official v2 đều 20/20 qua adapter, tổng 10 ca predict đầy đủ.
- `core_1311/`: nghiệm thu 11 test core và đối chiếu 20 ảnh của gói thử.
- `screenshots/`: 10 ảnh người dùng chụp đã kiểm trực tiếp; mèo đúng là Abyssinian_179.
- `screenshots_manifest.json`: SHA-256 từng ảnh, giữ trạng thái lúc kiểm ảnh trước khi hoàn thiện báo cáo.
- `provenance/`: nguồn ảnh DTD/FeliBreed và mức xác minh.

Các run_summary lịch sử có thể ghi mèo đúng cũ Abyssinian_101; ảnh chụp cuối dùng Abyssinian_179. Không thay lịch sử benchmark theo ảnh minh họa mới. Không diễn giải kết quả model 13/11 thành số đo official v2.

Tạo lại Word bằng `scripts/build_official_web_report.py` với Python có python-docx và Pillow. Bộ tạo đọc số liệu và ảnh từ thư mục bằng chứng này. Chạy `render_docx.py` theo quy trình documents để kiểm bản xuất; bản PDF bàn giao nằm cùng Word trong BaoCao.
