"""Chạy notebooks/kaggle_runner.ipynb trên máy ngoài Kaggle (VPS), không cần mở Jupyter.

Ví dụ:
    export HF_TOKEN=hf_...            # token có quyền ghi vào org catbreed-id
    python scripts/run_vps.py --config configs/v2/sweep_tai.json --base /data/runs/tai \
        --input /data/official_archives

Script làm bốn việc:
1. Kiểm tra repo local sạch và commit HEAD đã có trên GitHub (runner sẽ clone đúng commit này).
2. Tự tính SUBMISSION (git_commit, notebook_source_sha256, git_dirty) cho commit HEAD.
3. Tạo bản sao notebook, chèn một cell ngay sau cell CONFIG: CONFIG.update(<file cấu hình>) và SUBMISSION.
4. Chạy bản sao bằng `jupyter nbconvert --execute`, log ghi vào <base>/run.log, notebook đã chạy ở <base>/executed.ipynb.

Mỗi lượt chạy song song phải có --base riêng. --input là thư mục chứa images.tar.gz, annotations.tar.gz,
dtd-r1.0.1.tar.gz gốc (và thư mục unseen_cat nếu có); notebook vẫn kiểm MD5 lúc chạy.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "kaggle_runner.ipynb"


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def notebook_source_hash(notebook):
    # Cùng công thức với notebook_source_hash trong notebook
    content = [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def submission(branch):
    if git("status", "--porcelain"):
        sys.exit("Repo local còn thay đổi chưa commit. Commit và push trước khi chạy.")
    commit = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/" + branch).split()
    if not remote or remote[0] != commit:
        sys.exit(f"HEAD {commit[:7]} chưa khớp origin/{branch}. Push trước khi chạy.")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": False,
        "notebook_source_sha256": notebook_source_hash(notebook),
        "runner": "run_vps.py",
        "profile": None,  # điền từ cấu hình bên dưới
    }


def build_notebook(overrides, submit):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    index = next(
        i for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith("CONFIG = {")
    )
    source = (
        "# Cell do scripts/run_vps.py chèn: cấu hình của lượt chạy\n"
        f"CONFIG.update(json.loads({json.dumps(json.dumps(overrides, ensure_ascii=False))}))\n"
        f"SUBMISSION = json.loads({json.dumps(json.dumps(submit))})\n"
    )
    # json chưa được import ở các cell trước CONFIG
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "import json\n" + source}
    notebook["cells"].insert(index + 1, cell)
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON: các khóa ghi đè CONFIG của lượt chạy")
    parser.add_argument("--base", required=True, help="Thư mục làm việc riêng của lượt chạy")
    parser.add_argument("--input", required=True, help="Thư mục chứa archive gốc (và unseen_cat)")
    parser.add_argument("--branch", default="feat/sweep-profile")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ tạo notebook, không chạy")
    args = parser.parse_args()

    overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if overrides.get("profile") not in ["smoke", "sweep", "official"]:
        sys.exit("Cấu hình phải có profile smoke, sweep hoặc official.")
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        sys.exit("Đặt biến môi trường HF_TOKEN trước khi chạy.")
    submit = submission(args.branch)
    submit["profile"] = overrides["profile"]

    base = Path(args.base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    notebook_path = base / "run.ipynb"
    notebook_path.write_text(
        json.dumps(build_notebook(overrides, submit), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("SUBMISSION:", json.dumps(submit))
    print("Notebook:", notebook_path)
    if args.dry_run:
        return

    env = {
        **os.environ,
        "CATBREED_BASE": str(base),
        "CATBREED_INPUT": str(Path(args.input).resolve()),
        "CATBREED_BRANCH": args.branch,
        "CATBREED_RUNNER": "1",
    }
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1", "--output", "executed.ipynb", str(notebook_path),
    ]
    with open(base / "run.log", "w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=base)
    print("Xong, mã thoát", result.returncode, "| log:", base / "run.log")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
