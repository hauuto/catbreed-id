# CatBreedID

Ứng dụng phân loại giống mèo qua ảnh bằng CNN.

## 1. Clone project

```powershell
git clone https://github.com/hauuto/catbreed-id.git
cd catbreed-id
```

## 2. Cài đặt

Yêu cầu máy đã cài Python và Git. Các lệnh dưới đây dùng PowerShell trên Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Chạy ứng dụng local

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --debug
```

Mở http://127.0.0.1:5000/health để kiểm tra. Kết quả mong đợi:

```json
{"status": "ok"}
```

Nhấn `Ctrl+C` để dừng.

Hiện ứng dụng mới có endpoint kiểm tra; chức năng dự đoán sẽ được bổ sung.

## 4. Chạy runner trên Kaggle

1. Import file `notebooks/kaggle_runner.ipynb` vào notebook Kaggle.
2. Bật **Internet**.
3. Thêm secret `HF_TOKEN` và cho phép notebook sử dụng.
4. Chạy các cell theo thứ tự.

Tài khoản Hugging Face cần có quyền ghi vào repo
`catbreed-id/catbreed-experiments`.

Runner hiện chạy kiểm tra lưu và đồng bộ checkpoint, chưa train model thật. Không cần bật GPU cho bước này.

## 5. Cập nhật code

Tại thư mục project:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Quy tắc đóng góp: xem `CONTRIBUTING.md`.