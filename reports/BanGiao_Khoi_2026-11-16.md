# Bàn giao kết quả P1–P2 – Khôi – 16/11/2026

## 1. Trạng thái bàn giao

- Nhánh: `feat/khoi-scripts`.
- Commit đã push: `f8800b855bdb3df2c3c2d2ca93800a3ebf9088b4` (`feat: P1 complete`).
- Local và `origin/feat/khoi-scripts` đang cùng commit trên.
- Ba script phân tích, dependency và bộ kiểm thử đã hoàn thành.
- Nghiệm thu bằng dữ liệu giả đã hoàn thành.
- Phân tích KNN bằng dữ liệu thật hiện có đã hoàn thành cho checkpoint triển khai của protocol cũ.
- Chưa xếp hạng sweep thật và chưa sinh cấu hình stage 2 chính thức vì kho kết quả chưa có `sweep_results.csv`.
- Đồng đội sẽ bổ sung code/chạy sweep trong các lượt tiếp theo; khi có kết quả chỉ cần tiếp tục theo Mục 7, không viết lại script.

## 2. Các quyết định đã chốt

1. `factor_group` đại diện cho yếu tố được khảo sát, không phải từng mức của yếu tố:
   - `lr_*` → `lr`;
   - `unf25`, `unf50`, `unf100` → `unf`;
   - `drop0`, `drop50` → `drop`;
   - `head512` → `head`;
   - `bl_*` → `bl`.
2. Cho phép đọc `info.json` để lấy `model`, `training_seed` và metadata run.
3. Cho phép đọc `calibration.json` để lấy `n_images`; cũng có thể truyền `--n-val`.
4. Đã khai báo trực tiếp `numpy`, `pandas`, `scipy`, `scikit-learn` trong `requirements.txt`.
5. `make_stage2.py` tự kiểm tra cấu hình theo luật của notebook. `run_vps.py --dry-run` là kiểm tra tích hợp sau commit/push, không thay thế validator.
6. Không mở hoặc dùng `test.npz`, `near_ood.npz`, `far_ood.npz`, file có `test`, hay dữ liệu unseen-cat.

## 3. Phần đã triển khai

### 3.1. `scripts/rank_sweep.py`

Chức năng đã có:

- Nhận một hoặc nhiều file có tên `sweep_results.csv`.
- Kiểm tra đủ cột, JSON trong `config_overrides`, khóa trùng `(sweep_point, model, seed, stage)` và `protocol_id` đồng nhất.
- Chọn checkpoint theo đúng QĐ9 trong notebook:
  - F1 macro cao hơn;
  - nếu `abs(delta F1) < 1e-4`, accuracy cao hơn;
  - nếu vẫn hòa, epoch nhỏ hơn;
  - nếu hòa hoàn toàn, giữ bản ghi xuất hiện trước.
- Xếp hạng trong từng model và phân nhóm `lr/unf/drop/head/bl`.
- Xếp hạng độ nhạy theo accuracy và NLL nếu có `val_nll`.
- Tính Kendall τ giữa các tiêu chí.
- In bước accuracy `1/n` và đánh dấu top 1–2 không phân biệt được khi chênh accuracy `< 1/n`.
- In top 3 và khoảng cách F1 giữa hạng 1–2; cảnh báo khi khoảng cách `< 0,005`.

Đầu ra:

- `stage1_ranking.csv`;
- `stage1_top.csv`;
- `stage1_best_per_factor.csv`;
- `stage1_criterion_sensitivity.csv`.

### 3.2. `scripts/make_stage2.py`

Chức năng đã có:

- Sinh A/B/C từ hạng 1 của ResNet50/MobileNetV2/EfficientNet-B0, seed 43–44.
- Sinh Baseline từ hạng 1 của CNN baseline, seed 43–44.
- Sinh D riêng cho mỗi backbone bằng cách ghép điểm tốt nhất của từng `factor_group`, seed 42–44.
- Nếu D có overrides trùng hoàn toàn top 1 giai đoạn 1 thì bỏ seed 42 để không chạy lặp.
- Nếu hai nhóm ghi đè cùng khóa bằng hai giá trị khác nhau thì dừng và báo rõ xung đột.
- Không sinh file cho model thiếu dữ liệu và in cảnh báo.
- Giữ `owner`, `protocol_id`, epochs và khóa chung từ cấu hình gốc.
- Ghi `stage2_source` trong từng cấu hình và tạo `stage2_manifest.json`.
- Kiểm tra model, seed, sweep grid, khóa lạ, khóa cố định và giá trị kiến trúc trước khi ghi file.

Đầu ra tối đa:

- `stage2_A.json`;
- `stage2_B.json`;
- `stage2_C.json`;
- `stage2_Baseline.json`;
- `stage2_D_resnet50.json`;
- `stage2_D_mobilenet_v2.json`;
- `stage2_D_efficientnet_b0.json`;
- `stage2_manifest.json`.

### 3.3. `scripts/knn_k_sensitivity.py`

Chức năng đã có:

- Nhận một hoặc nhiều stage directory.
- Đọc đúng `train.npz`, `val.npz`, `probe_ood.npz`, `validation_scores.csv`, `probe_results.csv` và `info.json`.
- Dùng đúng công thức notebook:
  - chuẩn hóa L2;
  - `NearestNeighbors(metric="euclidean", algorithm="brute")`;
  - điểm bằng âm khoảng cách tới láng giềng thứ k;
  - ngưỡng là phân vị 5%, `method="linear"`;
  - AUROC, FPR@95TPR và các metric tại ngưỡng giữ nguyên công thức notebook.
- Chạy `k ∈ {1, 5, 10, 20, 50}`.
- Với k=10, bắt buộc điểm validation khớp trong sai số `1e-6` và `fpr_at_threshold` khớp `probe_results.csv`.
- In trung bình ± độ lệch chuẩn theo k khi có nhiều stage directory.

Đầu ra: `knn_k_sensitivity.csv`.

## 4. Kết quả kiểm thử và nghiệm thu

### 4.1. Unit test

Lệnh:

```powershell
python -m pytest tests -q -W error
```

Kết quả: `18 passed`.

Các ca đã phủ gồm:

- hòa F1 trong dung sai, phân xử bằng accuracy và epoch;
- khóa sweep trùng, protocol lệch, NLL có/không có;
- nhóm `unf/drop/lr`;
- xung đột overrides, model thiếu, D trùng top 1;
- khóa override lạ hoặc cố định;
- k=1 với ảnh validation trùng train;
- train bank nhỏ hơn k;
- k=10 khớp và không khớp file gốc;
- tổng hợp nhiều seed và chặn tên đầu vào bị cấm.

`python -m pip check` cũng đã qua, không có dependency hỏng.

### 4.2. Nghiệm thu tích hợp bằng dữ liệu giả

Đã tạo dữ liệu giả gồm 4 model, các nhóm `lr/unf/head/drop`, rồi chạy nối tiếp hai CLI.

Kết quả:

- sinh đủ 4 CSV stage 1;
- mỗi model có đúng top 3;
- nhóm yếu tố đúng `drop/head/lr/unf`;
- sinh đủ 7 cấu hình stage 2 và manifest;
- cả 7 cấu hình đều qua `scripts/run_vps.py --dry-run` trên commit đã push;
- KNN giả chạy đủ hai seed và năm giá trị k;
- mọi assertion nghiệm thu thủ công đều qua.

Artifact nghiệm thu giả được giữ local, không commit:

```text
outputs/p2_acceptance_20261116_155546/
```

Các số liệu từ dữ liệu giả chỉ dùng kiểm tra code, không dùng trong báo cáo nghiên cứu.

## 5. Kết quả KNN trên dữ liệu thật hiện có

Tại thời điểm bàn giao, kho Hugging Face chưa có study v2 hoặc `sweep_results.csv`. Bản khóa mới nhất sẵn có là:

- Study: `20261116_052028_study_99d77653`;
- Profile: `official`;
- Protocol: `Nhom09_QD1-QD10_2026-10-18`;
- Checkpoint triển khai: `efficientnet_b0`, stage `frozen`, seed 42;
- Các seed cùng cấu hình: 42, 43, 44.

Kết quả sensitivity:

| k | τ | AUROC | FPR@95TPR | TPR tại τ | FPR tại τ | Ảnh probe được nhận |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.942418 | 0.999538 | 0.002 | 0.949580 | 0.002 | 1/500 |
| 5 | -0.997502 | 0.999983 | 0 | 0.949580 | 0 | 0/500 |
| 10 | -1.022193 | 0.999966 | 0 | 0.949580 | 0 | 0/500 |
| 20 | -1.051887 | 0.999983 | 0 | 0.949580 | 0 | 0/500 |
| 50 | -1.087239 | 0.999983 | 0 | 0.949580 | 0 | 0/500 |

Ba seed cho kết quả giống nhau vì stage frozen dùng cùng backbone đặc trưng. Kiểm tra chéo k=10 đã khớp file gốc. Kết quả cho thấy k=10 ổn định trong dải đã đăng ký; không dùng phân tích này để đổi k sau khi xem kết quả.

Artifact local, không commit:

```text
outputs/p2_real_20261116/knn_k_sensitivity.csv
```

Đây là bằng chứng nghiệm thu trên protocol cũ, chưa thay thế kết quả v2 cần chạy sau khi có khóa mới.

## 6. Phần chưa thể hoàn thành

Tại thời điểm kiểm tra kho `catbreed-id/catbreed-experiments`:

- không có file nào tên `sweep_results.csv`;
- chỉ có một study ngày 16/11 và đó là lượt `official` của protocol cũ;
- chưa có `locked_selection.json` của protocol v2.

Do đó chưa có cơ sở để:

- lập `stage1_ranking.csv` chính thức;
- chọn top thật của từng model/yếu tố;
- sinh A/B/C/D/Baseline chính thức;
- chạy KNN sensitivity trên checkpoint triển khai v2.

Không dùng kết quả official cũ để giả làm kết quả sweep v2.

## 7. Quy trình tiếp tục khi đồng đội gửi `sweep_results.csv`

### Bước 1. Tập hợp kết quả

Giữ nguyên tên mỗi file là `sweep_results.csv` và đặt trong thư mục riêng, ví dụ:

```text
outputs/stage1_inputs/hau/sweep_results.csv
outputs/stage1_inputs/khoi/sweep_results.csv
outputs/stage1_inputs/tai/sweep_results.csv
outputs/stage1_inputs/vy/sweep_results.csv
```

Trước khi chạy, xác nhận các file cùng `protocol_id`. Không tải hoặc mở file test/OOD test.

### Bước 2. Xếp hạng giai đoạn 1

Nếu biết số ảnh validation:

```powershell
python scripts/rank_sweep.py `
  outputs/stage1_inputs/hau/sweep_results.csv `
  outputs/stage1_inputs/khoi/sweep_results.csv `
  outputs/stage1_inputs/tai/sweep_results.csv `
  outputs/stage1_inputs/vy/sweep_results.csv `
  --output-dir outputs/stage1_official `
  --n-val <N_VAL>
```

Hoặc thay `--n-val` bằng:

```powershell
--calibration <duong-dan>/calibration.json
```

Sau khi chạy, kiểm tra:

- mỗi model có top 3;
- khoảng cách F1 hạng 1–2;
- cờ top 1–2 không phân biệt được theo accuracy;
- factor group đúng;
- NLL được ghi là không khả dụng nếu `val_nll` không có;
- không có cảnh báo protocol hoặc khóa trùng.

### Bước 3. Sinh cấu hình stage 2

```powershell
python scripts/make_stage2.py `
  --ranking outputs/stage1_official/stage1_ranking.csv `
  --best-per-factor outputs/stage1_official/stage1_best_per_factor.csv `
  --base-config configs/v2/sweep_khoi.json `
  --output-dir configs/v2/stage2_generated
```

Kiểm tra `stage2_manifest.json`, các cảnh báo model thiếu và nguồn `stage2_source` trong từng file. Nếu D báo xung đột cùng khóa thì dừng, không tự chọn giá trị.

### Bước 4. Kiểm tra runner

`run_vps.py` yêu cầu worktree sạch và commit đã push. Sau khi đưa các cấu hình chính thức vào Git:

```powershell
python scripts/run_vps.py `
  --config configs/v2/stage2_generated/<file>.json `
  --base <thu-muc-rieng> `
  --input <thu-muc-archive> `
  --branch feat/khoi-scripts `
  --dry-run
```

Chạy cho từng file được sinh. Validator trong `make_stage2.py` mới là kiểm tra cấu hình chính; dry-run xác nhận tích hợp với runner và commit.

### Bước 5. KNN sau khi khóa v2

Từ `locked_selection.json` v2, lấy stage directory của checkpoint triển khai và các seed cùng model/stage. Chỉ tải:

- `info.json`;
- `train.npz`;
- `val.npz`;
- `probe_ood.npz`;
- `validation_scores.csv`;
- `probe_results.csv`;
- `calibration.json` khi cần.

Chạy:

```powershell
python scripts/knn_k_sensitivity.py `
  <stage-dir-seed-42> `
  <stage-dir-seed-43> `
  <stage-dir-seed-44> `
  --output outputs/stage2_official/knn_k_sensitivity.csv
```

Nếu kiểm tra chéo k=10 không khớp, dừng và kiểm tra lại checkpoint/stage; không bỏ qua lỗi.

## 8. Checklist cho lần bàn giao tiếp theo

- [ ] Đã nhận đủ `sweep_results.csv` từ các lượt của đồng đội.
- [ ] Các file cùng protocol v2 và không trùng khóa.
- [ ] Đã sinh và rà bốn CSV stage 1 chính thức.
- [ ] Đã chốt mapping A/B/C/D/Baseline từ kết quả thật.
- [ ] Tất cả config stage 2 qua validator và dry-run.
- [ ] Đã commit/push config chính thức trước khi chạy train.
- [ ] Đã có `locked_selection.json` v2.
- [ ] KNN k=10 khớp file gốc cho mọi seed.
- [ ] Đã lưu `knn_k_sensitivity.csv` chính thức.
- [ ] Không đọc hoặc dùng dữ liệu test/OOD test trước bước được phép.

