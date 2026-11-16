#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test "$(id -un)" = "thanhkhoi" || { echo 'Expected thanhkhoi account' >&2; exit 1; }
case "$ROOT" in /mnt/thanhkhoi/*) ;; *) echo 'Run directory must be under /mnt/thanhkhoi' >&2; exit 1;; esac
cd "$ROOT"
if [ "${1:-}" = "--background" ]; then
    if [ -f api.pid ]; then
        PID=$(cat api.pid)
        if kill -0 "$PID" 2>/dev/null; then
            test "$(readlink "/proc/$PID/cwd")" = "$ROOT" || { echo 'PID belongs to a different working directory' >&2; exit 1; }
            echo "API already running: $PID"
            exit 0
        fi
    fi
    nohup sh "$0" > "$ROOT/api.log" 2>&1 < /dev/null &
    echo $! > "$ROOT/api.pid"
    echo "Started remote API: $(cat "$ROOT/api.pid")"
    exit 0
fi
export PYTHONPATH="$ROOT/runtime_api:$ROOT"
export TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CATBREED_MODEL=real CATBREED_DEVICE=cuda CATBREED_PRECISION=float32
export CATBREED_MODEL_PACKAGE="$ROOT/model_package"
exec /mnt/thanhkhoi/catbreed-id/.venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 18000
