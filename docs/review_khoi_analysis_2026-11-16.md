# Review feat/khoi-analysis (2026-11-16)

Chỉ đọc. Không sửa script. Working tree: `feat/sweep-profile` @ deb0d34. `git diff --stat HEAD origin/feat/khoi-analysis -- scripts tests` rỗng, nên `scripts/` và `tests/` giống hệt `origin/feat/khoi-analysis`.

## 1. Trạng thái repo
- `git status --short`: rỗng (sạch) trước khi tôi tạo file mới.
- `.venv` nằm trong `.gitignore`: có (`.gitignore:228:.venv/`, theo `git check-ignore -v .venv`).
- Nhánh hiện tại là `feat/sweep-profile`. Nhánh `feat/hau-core` không có ở local lẫn remote.

## 2. `scripts/paired_tests.py`: CLI
`main()` dùng argparse, chỉ có 3 tham số:
```python
parser.add_argument("stage_dirs", nargs="+", type=Path)
parser.add_argument("--pairs", help="Danh sách A:B,C:D; mặc định so mọi cặp")
parser.add_argument("--output", type=Path, required=True)
...
run(args.stage_dirs, args.output, args.pairs)
```
- Không có `--seed` hay `--repeats`. `run()` gọi `compare_pair(a, arrays[a], b, arrays[b])` mà không truyền hai tham số này, nên luôn dùng mặc định.
- Mặc định trong `compare_pair`: `repeats: int = 1000`, `seed: int = 42`.
- Muốn đổi seed hay repeats phải sửa code. Tôi chỉ ghi nhận, không đề xuất sửa.

## 3. `FORBIDDEN_TOKENS` khớp chuỗi con
Định nghĩa ở `fetch_sweep.py:20` và `geometry_table.py:15`: `FORBIDDEN_TOKENS = ("test", "near_ood", "far_ood", "unseen")`.

Phép kiểm tra là toán tử `in` trên chuỗi, tức khớp chuỗi con, không khớp theo thành phần đường dẫn:
```python
# fetch_sweep.py:31-39 (geometry_table.py:25-29 tương tự)
text = str(path).replace("\\", "/").lower()
...
anchors = [index for index, part in enumerate(parts) if part in {"runs", "outputs"}]
scoped = "/".join(parts[anchors[-1] :]) if anchors else "/".join(parts[-3:])
token = next((item for item in FORBIDDEN_TOKENS if item in scoped), None)
```
**Kết luận: có chặn nhầm.** `"test" in "runs/latest/frozen/eval_geometry.csv"` là `True`, tương tự với `contest`, `attestation`, `pytest`... Đường dẫn chứa các từ này sẽ ném `ValueError("Đường dẫn chứa từ cấm 'test'...")`. Chưa chạy thật vì không import được numpy/pandas, kết luận dựa trên đọc code.

Phạm vi bị chặn nhầm:
- Chuỗi được kiểm tra là `scoped`, đã chuyển sang chữ thường. Nếu có thành phần `runs` hoặc `outputs`, `scoped` lấy từ thành phần cuối cùng khớp trở đi. Nếu không, chỉ lấy 3 thành phần cuối.
- Thư mục như `C:\Users\x\latest\...` bên ngoài `scoped` thì không bị chặn.
- Tên thư mục `study_id` cũng bị chặn theo cùng cách (`fetch_sweep.py:112`): `if any(token in study_id.lower() for token in FORBIDDEN_TOKENS)`.
- Chiều ngược lại (chặn thiếu) chưa được đánh giá.

## Điểm chưa chắc chắn
- Chưa chạy test hay script nào (Application Control chặn DLL của numpy).
- Mục 3 chưa kiểm chứng bằng thực thi.
