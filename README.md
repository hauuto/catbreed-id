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

## 4. Tự kiểm tra runner trên Kaggle

Thực hiện sau khi đã chấp nhận lời mời vào Organization
`catbreed-id` trên Hugging Face.

### Tạo token cá nhân

1. Đăng nhập Hugging Face bằng tài khoản của mình.
2. Mở https://huggingface.co/settings/tokens và chọn tạo token.
3. Đặt tên `kaggle-catbreed`, chọn preset **Write**.
4. Cấp quyền ghi nội dung repository cho Organization `catbreed-id`.
5. Tạo và sao chép token. Không chia sẻ token cho người khác.

### Chuẩn bị notebook

1. Tạo notebook Kaggle riêng tư.
2. Import file `notebooks/kaggle_runner.ipynb` từ project đã clone.
3. Trong Settings, bật **Internet**; chưa cần GPU hoặc dataset.
4. Vào **Add-ons → Secrets**, thêm:
   - Label: `HF_TOKEN`
   - Value: token Hugging Face của mình.
5. Bật quyền cho notebook sử dụng secret đó.

### Chạy kiểm tra

Chạy lần lượt cả bốn cell của notebook.

Thành công khi output cuối có:

    frozen: Đã đồng bộ
    finetuned: Đã đồng bộ
    final: Đã đồng bộ
    Workflow thử hoàn thành

Mở kho kết quả:
https://huggingface.co/datasets/catbreed-id/catbreed-experiments

Trong `Files`, kiểm tra thư mục `runs/<run_id>/` vừa tạo có:

- `info.json`: đúng tài khoản của mình, trạng thái `completed`.
- `frozen/`: có `checkpoint.pt` và `metrics.json`.
- `finetuned/`: có `checkpoint.pt` và `metrics.json`.

Gửi `run_id` cho Hậu để xác nhận đã tự test thành công.
Đây là smoke test bằng tensor nhỏ, chưa huấn luyện model thật.

### Nếu gặp lỗi

- Không kết nối được GitHub: kiểm tra **Internet: On**.
- Không đọc được token: kiểm tra tên `HF_TOKEN` và quyền sử dụng secret.
- Hugging Face báo 401/403: kiểm tra đã nhận lời mời, có quyền
  Write trong Organization và token có quyền ghi.
- Khi gửi lỗi cho nhóm, không gửi kèm token.

## 5. Cập nhật code

Tại thư mục project:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Quy tắc đóng góp: xem `CONTRIBUTING.md`.