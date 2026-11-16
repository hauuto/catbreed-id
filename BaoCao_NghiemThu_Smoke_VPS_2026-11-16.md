# Báo cáo nghiệm thu lượt chạy thử VPS

**Ngày thực hiện:** 16/11/2026  
**Người vận hành:** Khôi  
**Phạm vi:** Lượt chạy thử theo `configs/v2/smoke_trial.json` trên VPS RTX A6000  
**Kết luận:** **ĐẠT CÓ LƯU Ý**

## 1. Mục tiêu nghiệm thu

Xác nhận pipeline VPS có thể chạy trọn luồng pha A trước khi khởi động sweep giai đoạn 1, bao gồm:

- chạy notebook bằng `scripts/run_vps.py` từ commit sạch;
- xác thực đúng dữ liệu và split đã chốt;
- không mở các tập bị khóa trước pha B;
- sinh đủ artifact retention và geometry;
- đồng bộ kết quả lên Hugging Face;
- kiểm tra các script hậu xử lý của Khôi trên kết quả thật.

## 2. Môi trường và phiên bản

| Hạng mục | Giá trị |
|---|---|
| Máy | `megalab-HP-Z8-G4-Workstation` |
| Hệ điều hành | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA RTX A6000, 49.140 MiB VRAM |
| Driver | 580.178.04 |
| PyTorch | 2.15.0+cu132 |
| Python | 3.13.18 trong `.venv` |
| Repo VPS | `/mnt/thanhkhoi/catbreed-id` |
| Branch | `feat/sweep-profile` |
| Commit | `deb0d34e5fab6ae3bf0a97086165e4f8138f66b5` |
| Notebook SHA-256 | `cb124dd2d14cbd6739833ae9c7eb9879a682ae0160cb8276fdbfc2cf7487f070` |
| Base chạy thử | `/mnt/thanhkhoi/work` |
| Mount lưu dữ liệu | `/mnt/thanhkhoi` |

Repo VPS sạch trước và sau lượt chạy. Tại thời điểm chốt báo cáo, `/mnt/thanhkhoi` còn khoảng 123 GiB; không ghi artifact vào phân vùng `/` hoặc mount của người khác.

## 3. Lệnh thực hiện

```bash
python scripts/run_vps.py \
  --config configs/v2/smoke_trial.json \
  --base /mnt/thanhkhoi/work \
  --input /mnt/thanhkhoi/input \
  --data-dir /mnt/thanhkhoi/input/official
```

Lệnh được chạy trong `tmux`, sau khi `source ~/.bashrc`, kích hoạt `.venv` và kiểm tra token Hugging Face có độ dài 37, tiền tố `hf_`.

## 4. Diễn biến

| Lần | Thời gian (+07:00) | Kết quả | Ghi chú |
|---|---|---|---|
| 1 | 17:18:58–17:19:03 | Không đạt, mã 1 | Token nhập sai, dài 46 ký tự; Hugging Face trả 401. Không phát sinh kết quả huấn luyện hợp lệ. |
| 2 | 17:24:02–17:28:00 | Đạt, mã 0 | Token đã sửa thành 37 ký tự. Log sau đó bị một thao tác notebook khác ghi đè. |
| 3 | 17:44:25–17:47:49 | Đạt, mã 0 | Chạy lại để tạo bộ log và artifact nghiệm thu nhất quán. Đây là lượt dùng trong báo cáo. |

Study chính thức của lượt nghiệm thu:

- Study ID: `20261116_104455_study_9c0b71a2`
- Thời gian study: `17:44:55–17:47:09 +07:00`
- Trạng thái trong `info.json`: `sweep_completed`
- Protocol: `Nhom09_v2_2026-11-16`
- `split_id`: `3b283d422973d497d268d935ef0fe4a76bbd2d10f4b5efdef20a8d859949cfcb`

## 5. Đối chiếu tiêu chí nghiệm thu

| Tiêu chí | Kết quả | Bằng chứng / nhận xét |
|---|---|---|
| Runner kết thúc với mã 0 | **Đạt** | Terminal in `Xong, mã thoát 0`; `run.log` và `executed.ipynb` được ghi lúc 17:47:49. |
| Commit sạch và đúng phiên bản | **Đạt** | `deb0d34`; repo VPS không có thay đổi chưa commit. |
| MD5 archive chính thức khớp | **Đạt có lưu ý** | Cell chuẩn bị dữ liệu bắt buộc so sánh MD5 và raise lỗi nếu sai; pipeline đã chạy hết với mã 0. Tuy nhiên `run.log` chưa in trực tiếp giá trị MD5 đã tính, nên thiếu bằng chứng log theo nghĩa đen của tiêu chí bàn giao. |
| `split_id` bắt đầu bằng `3b283d42` | **Đạt** | Giá trị đầy đủ trong `info.json` bắt đầu bằng `3b283d42`. |
| Không giải mã dữ liệu bị khóa | **Đạt** | Không giải mã test, near-OOD, far-OOD hoặc unseen. Pha A chỉ dùng train, validation và `probe_ood` được phép phục vụ tuning. |
| Có `crop_retention_trainval.csv` | **Đạt** | File 48.777 byte trong thư mục study. |
| Có `eval_geometry.csv` | **Đạt** | Có bảng tổng study và bảng theo cả frozen/finetuned. |
| Đồng bộ Hugging Face | **Đạt có cảnh báo** | Các lần upload cuối báo `Đã đồng bộ`. API commit có lúc trả HTTP 500 nhưng retry tự động thành công; runner vẫn kết thúc mã 0. |
| `fetch_sweep.py` đọc được kết quả thật | **Đạt** | Xuất 2 dòng, 1 sweep point, model EfficientNet-B0 vào `khoi_outputs/smoke_inputs/smoke`. |
| `geometry_table.py` tổng hợp được geometry | **Đạt** | Sinh `/mnt/thanhkhoi/khoi_outputs/smoke_geometry.csv`. |

Không có file test/OOD bị mở để lập báo cáo này. Chỉ `run.log`, `executed.ipynb`, `info.json`, bảng validation/geometry và nhật ký vận hành được dùng làm bằng chứng.

## 6. Artifact nghiệm thu

### Artifact gốc

- `/mnt/thanhkhoi/work/run.log`
- `/mnt/thanhkhoi/work/executed.ipynb`
- `/mnt/thanhkhoi/work/outputs/20261116_104455_study_9c0b71a2/info.json`
- `/mnt/thanhkhoi/work/outputs/20261116_104455_study_9c0b71a2/sweep_results.csv`
- `/mnt/thanhkhoi/work/outputs/20261116_104455_study_9c0b71a2/crop_retention_trainval.csv`
- `/mnt/thanhkhoi/work/outputs/20261116_104455_study_9c0b71a2/eval_geometry.csv`

### Artifact hậu kiểm của Khôi

- `/mnt/thanhkhoi/khoi_outputs/smoke_inputs/smoke/sweep_results.csv`
- `/mnt/thanhkhoi/khoi_outputs/smoke_inputs/smoke/info.json`
- `/mnt/thanhkhoi/khoi_outputs/smoke_geometry.csv`
- `/mnt/thanhkhoi/khoi_outputs/run_journal.md`

Lệnh bàn giao ban đầu truyền cả `/mnt/thanhkhoi/work` cho `fetch_sweep.py`. Base này đã chứa nhiều study từ các lần chạy lại nên script từ chối gán một owner chung. Hậu kiểm được chạy lại đúng cách với study mới nhất, không xóa hoặc ghi đè các study cũ.

## 7. Kết quả kỹ thuật của lượt thử

| Stage | Epoch tốt nhất / đã chạy | Thời gian train | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|---:|
| Frozen | 2 / 2 | 5,52 giây | 0,873950 | 0,871099 |
| Finetuned | 2 / 2 | 5,17 giây | 0,886555 | 0,883248 |

So sánh geometry:

| Stage | Center accuracy | Letterbox accuracy | Chênh lệch | Center macro-F1 | Letterbox macro-F1 | Chênh lệch |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 0,873950 | 0,865546 | -0,008403 | 0,871099 | 0,863330 | -0,007769 |
| Finetuned | 0,886555 | 0,810924 | -0,075630 | 0,883248 | 0,808926 | -0,074321 |

Các số geometry chỉ là kết quả validation của smoke trial, không được dùng thay cho kết luận từ sweep đầy đủ.

## 8. Ước lượng và quyết định chuyển bước

Tổng thời gian train hai stage trong smoke là khoảng 10,70 giây. Theo công thức bàn giao, phần train của lượt 30+30 epoch được ước lượng khoảng 160 giây; cộng chuẩn bị dữ liệu, validation, ghi artifact và đồng bộ, một điểm EfficientNet-B0 dự kiến khoảng 4–5 phút. ResNet50 và lỗi mạng retry có thể lâu hơn.

Ước lượng vận hành cho 51 điểm giai đoạn 1 là khoảng 4–7 giờ. Sau khi smoke đạt, sweep giai đoạn 1 đã được khởi động tuần tự theo thứ tự Tài → Khôi → Vỹ → Hậu.

## 9. Kết luận và kiến nghị

Lượt chạy thử **đủ điều kiện chuyển sang sweep giai đoạn 1**: runner, GPU, split, artifact, đồng bộ và các script hậu kiểm đều hoạt động trên dữ liệu thật mà không mở dữ liệu bị khóa.

Hai lưu ý cần ghi nhận nhưng không làm sai kết quả:

1. Bổ sung việc in MD5 thực tế và MD5 kỳ vọng vào log runner để tiêu chí checksum có bằng chứng trực tiếp, thay vì chỉ suy ra từ cơ chế hard-fail và mã thoát 0.
2. Tiếp tục theo dõi HTTP 500 từ Hugging Face. Nếu một lượt hết retry hoặc `sync_receipt.json` báo thất bại thì chạy lại phần đồng bộ trước khi coi lượt đó hoàn tất.
