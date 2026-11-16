# Chụp minh chứng web official v2

Chỉ chụp khi API đã được khởi động lại với code tích hợp `catbreed_core`. Dùng model official v2 run `20261116_060136_efficientnet_b0_s42_b8c9dba0`, frozen, seed 42, thiết bị cuda. Gói `C:\Users\vyx\Downloads\model_package_1311` là gói thử ngày 13/11, không dùng chụp cho báo cáo.

## Mở web

Tại `D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id`, nếu web và SSH tunnel đã chạy thì dùng luôn. Khi cần mở lại, chạy ở hai cửa sổ PowerShell:

```powershell
.\scripts\connect_thanhkhoi_web.ps1
```

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --port 5000
```

Mở http://127.0.0.1:5000 và nhấn Ctrl+F5. Mục chi tiết phải ghi đúng run ở trên, KNN, τ = -1.0221926071848497, T = 0.6049425684624492, cuda/float32. Không chạy API GPU Windows.

## Ảnh đầu vào và kết quả cần giữ

Tất cả ảnh có sẵn tại **`D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id\outputs\official_v2\gpu\cases`**. Chọn ảnh gốc JPG dưới đây để tải lên; `dtd.png` là ảnh chụp màn hình cũ, không phải ảnh đầu vào.

| Thứ tự | File tải lên | Nhãn nguồn và kết quả thực tế | Tên ảnh chụp |
| --- | --- | --- | --- |
| 01 | `id_correct.jpg` | Abyssinian_179; nhận đúng Abyssinian | `01_meo_dung_tongquan.png`, `01_meo_dung_chitiet.png` |
| 02 | `dog.jpg` | Chó Oxford; bị từ chối | `02_cho_tongquan.png`, `02_cho_chitiet.png` |
| 03 | `dtd_val_sample.jpg` | DTD val1[0], banded theo quy trình người dùng; bị từ chối | `03_dtd_tongquan.png`, `03_dtd_chitiet.png` |
| 04 | `felibreed.jpg` | Người dùng xác nhận thư mục Local; được nhận là Ragdoll | `04_felibreed_tongquan.png`, `04_felibreed_chitiet.png` |
| 05 | `id_error.jpg` | Abyssinian_121; được nhận sai thành Maine Coon | `05_meo_sai_tongquan.png`, `05_meo_sai_chitiet.png` |

## Chụp mỗi ca

1. Tải đúng ảnh, nhấn phân tích và chờ kết quả cùng Grad-CAM tải xong.
2. Chụp tổng quan: ảnh đầu vào, tên dự đoán hoặc trạng thái từ chối, Top-5 và Grad-CAM. Nếu trang dài, chụp toàn trang hoặc thêm ảnh phần dưới.
3. Mở “Chi tiết mô hình và 5 điểm từ chối”, chụp đủ bảng năm phương pháp/điểm/ngưỡng và thông tin model. Có thể chụp thêm khi bảng không vừa màn hình.
4. Lưu vào **`D:\LEARNING\Nam4_Ki1\ThiGiacMayTinh\Project\catbreed-id\outputs\official_v2\screenshots\module_review`** theo tên trong bảng. Hai ảnh cũ nằm ở thư mục screenshots gốc chưa phải bộ minh chứng cuối.

Mục tiêu tối thiểu là 10 ảnh màn hình đủ nội dung. Giữ nguyên ca sai và FeliBreed được nhận. Nếu kết quả khác bảng, giữ ảnh chụp thực tế và báo lại trước khi viết báo cáo. Nhãn Local do người dùng xác nhận; chưa đối chiếu tên file gốc FeliBreed. DTD chưa so byte ảnh với kho gốc. Nêu rõ hai giới hạn nguồn này trong báo cáo.
