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

API mặc định sử dụng checkpoint ResNet50 fine-tuned tại `model_package/weights.pth`.
File này bị Git bỏ qua theo quy tắc `*.pth`; khi chuyển project sang máy khác,
cần sao chép checkpoint tương ứng vào cùng đường dẫn. Checkpoint chứa thứ tự
12 nhãn và cấu hình tiền xử lý. API trả dự đoán thật cùng Grad-CAM của lớp có
điểm cao nhất. Do chưa chốt ngưỡng OOD của checkpoint, API trả
`status: unverified` và `ood: null`; không sử dụng ngưỡng giả để từ chối ảnh.
Nếu có `model_package/ood_stats.npz` được tạo từ `export.npz` bằng
`python scripts/prepare_ood.py`, API hiển thị điểm thực của 5 phương pháp trong
`ood_all`, nhưng ngưỡng và kết luận vẫn để trống. Script dùng đặc trưng train,
tính các ngưỡng gợi ý trên validation và không dùng test để chọn ngưỡng;
`ood_candidate_thresholds.json` chỉ để đối chiếu, chưa được tự động áp dụng.
Checkpoint hiện dùng resize cạnh ngắn 256 (song tuyến), crop giữa 224 × 224
và chuẩn hóa ImageNet. Khởi động lại API sau khi thay checkpoint hoặc mã.
Ảnh theo EXIF có thể hiển thị khác hướng khi so với trình duyệt; Grad-CAM được
vẽ trên ảnh 224 × 224 sau tiền xử lý giống đường đánh giá huấn luyện.

Để dùng riêng mô hình giả lập cho các bài kiểm tra cũ, đặt biến môi trường
`CATBREED_MODEL=mock` trước khi khởi động API. Mặc định là `real`; nếu checkpoint
vắng mặt hoặc không tương thích, API báo lỗi khi khởi động.

Ứng dụng gồm hai dịch vụ:

- **API mô hình FastAPI:** cổng `8000`, nạp mô hình và xử lý ảnh.
- **Giao diện web Flask:** cổng `5000`, cho phép chọn ảnh và xem kết quả.

Có thể mở giao diện trước, nhưng API phải khởi động xong trước khi nhấn
**Phân tích ảnh**. Nếu giao diện đang mở mà API chưa chạy, giao diện sẽ báo
“Không kết nối được dịch vụ”.

### Cài dependencies cho API

Nếu chưa cài dependencies của API, chạy thêm lệnh sau tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m pip install -r app\api\requirements.txt
```

### Mở giao diện và API

Mở **hai cửa sổ PowerShell** tại thư mục gốc của project.

**Cửa sổ 1 – khởi động API FastAPI:**

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000 --reload
```

Đợi terminal hiện `Application startup complete` và tiếp tục để cửa sổ này
chạy. Lần đầu nạp checkpoint có thể mất một lúc. Mở
http://127.0.0.1:8000/health để kiểm tra: phản hồi có `"status": "ok"` và
`"is_mock": false` khi đang dùng mô hình thật. Tài liệu API tương tác ở
http://127.0.0.1:8000/docs.

**Cửa sổ 2 – khởi động giao diện Flask:**

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --port 5000
```

Sau đó mở trình duyệt và truy cập **http://127.0.0.1:5000** để sử dụng giao diện.
Endpoint kiểm tra của Flask là http://127.0.0.1:5000/health, kết quả mong đợi:

```json
{"status": "ok"}
```

Có thể thử hai ảnh mẫu có sẵn ngay trên giao diện:
`Abyssinian_1.jpg` (mèo hợp lệ) và `american_bulldog_2.jpg` (ảnh ngoại lai).
Để tải ảnh mới từ máy và nhận kết quả dự đoán, cần khởi động cả hai dịch vụ.
Nhấn `Ctrl+C` trong từng cửa sổ PowerShell để dừng dịch vụ tương ứng.

### Khi giao diện báo không kết nối được dịch vụ

1. Kiểm tra cửa sổ API vẫn đang chạy và đã hiện `Application startup complete`.
   Nếu chưa, chạy lại lệnh API ở trên từ thư mục gốc của project.
2. Mở http://127.0.0.1:8000/health. Nếu không truy cập được, xem lỗi trong
   terminal API; kiểm tra đã cài dependencies và có file
   `model_package/weights.pth`.
3. Mở giao diện qua http://127.0.0.1:5000 và nhấn **Ctrl+F5** để tải lại
   JavaScript mới, sau đó thử phân tích ảnh. Giao diện và API phải được truy
   cập trên cùng máy vì `127.0.0.1` chỉ máy đang mở trình duyệt.
4. Nếu terminal báo cổng `8000` hoặc `5000` đang được sử dụng, hãy kiểm tra
   dịch vụ tại địa chỉ tương ứng trước; nếu dịch vụ đã chạy đúng thì không cần
   khởi động thêm một bản nữa.

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
