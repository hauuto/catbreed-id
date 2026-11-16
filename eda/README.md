# EDA – khảo sát dữ liệu CatBreedID

Notebook `eda.ipynb` tạo số liệu và hình cho Chương 3 (dữ liệu). Chỉ dùng ảnh, nhãn, trimap, hộp bao; không dùng mô hình.

## Dữ liệu cần có

Mặc định notebook đọc `../../eda_data` (tính từ thư mục `eda/`, tức nằm cạnh thư mục repo). Đổi bằng biến môi trường `EDA_DATA`.

```
eda_data/
  archives/      images.tar.gz, annotations.tar.gz, dtd-r1.0.1.tar.gz (để kiểm MD5; không bắt buộc)
  oxford/        giải nén images.tar.gz và annotations.tar.gz  -> oxford/images, oxford/annotations
  dtd/           giải nén dtd-r1.0.1.tar.gz                    -> dtd/dtd/images
  felibreed/     FeliBreed/Mixed Breed, FeliBreed/Local, manifest.csv (bản đã xoay theo EXIF, như dataset roonyvtt/catbreed-felibreed-unseen)
  splits/        fixed_split.csv, data_audit.json do notebook huấn luyện sinh ra (split_id 3b283d42…)
                 crop_retention_trainval.csv (không bắt buộc; có thì mục C2 dùng số của notebook)
```

MD5 archive gốc: `images.tar.gz` 5c4f3ee8e5d25df40f4fd59a7f44e54c, `annotations.tar.gz` 95a8c909bbe2e81eed6a22bccdf3f68f, `dtd-r1.0.1.tar.gz` fff73e5086ae6bdbea199a49dfb8a4c1.

## Chạy lại

```bash
pip install numpy pandas matplotlib pillow scipy imagehash opencv-python-headless jupyter
cd eda
jupyter nbconvert --to notebook --execute --inplace eda.ipynb
```

Chạy bằng CPU, khoảng 2 phút. Lần đầu đọc toàn bộ ảnh và lưu `tables/image_inspection.csv` (không commit vì nặng); các lần sau đọc lại tệp này.

## Cờ `ALLOW_TEST`

Ở ô cấu hình đầu notebook. Mặc định `False`: notebook không đọc `test.txt` và không mở ảnh test. Chỉ đổi thành `True` sau khi nhóm báo đã khóa cấu hình; khi đó phần D (số ảnh test theo giống, trùng lặp giữa test và trainval, giữa FeliBreed và Oxford test, chất lượng và kích thước ảnh test) mới chạy.

## Đầu ra

- `tables/*.csv`: các bảng số liệu; `tables/captions.csv`: số và tiêu đề “Bảng 3.x / Hình 3.x”.
- `figures/*.png`: hình 300 dpi, không tiêu đề trên hình (không commit, chạy lại để có).
- `tables/manual_review.csv`: kết quả duyệt bằng mắt 30 ảnh mỗi giống (mục C5). Tệp này **không bị ghi đè** khi chạy lại.
- Repo chỉ lưu notebook với output dạng bảng/chữ; hình đã được bỏ khỏi output để nhẹ.
