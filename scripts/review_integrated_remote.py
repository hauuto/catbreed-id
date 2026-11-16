"""Recheck the integrated app and restart only this run's owned API."""
import base64
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if getpass.getuser() != 'thanhkhoi' or not ROOT.is_relative_to(Path('/mnt/thanhkhoi')):
    raise RuntimeError('Expected thanhkhoi workspace')
os.chdir(ROOT)
os.environ.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
output = ROOT / 'outputs/module_review'
output.mkdir(parents=True, exist_ok=True)
codes = {}
for count, label in ((20, 'mandatory_20'), (0, 'all_validation')):
    codes[label] = subprocess.run([sys.executable, 'scripts/check_app_consistency.py', '--images', 'inputs/oxford', '--device', 'cuda', '--count', str(count), '--output', str(output / (label + '.csv'))]).returncode
codes['benchmark'] = subprocess.run([sys.executable, 'scripts/benchmark_official_web.py', '--images', 'inputs/oxford', '--device', 'cuda', '--expected-user', 'thanhkhoi', '--output', str(output)]).returncode
pid_file = ROOT / 'api.pid'
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    proc = Path('/proc') / str(pid)
    if proc.exists():
        if (proc / 'cwd').resolve() != ROOT or b'uvicorn' not in (proc / 'cmdline').read_bytes():
            raise RuntimeError('Refusing to stop unrelated process')
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not proc.exists():
                break
            time.sleep(.1)
        if proc.exists():
            raise RuntimeError('API has not stopped')
subprocess.run(['sh', 'scripts/start_thanhkhoi_api.sh', '--background'], check=True)
for _ in range(60):
    try:
        with urllib.request.urlopen('http://127.0.0.1:18000/health', timeout=2) as response:
            health = json.load(response)
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError('API startup timeout')
manifest = json.loads((ROOT / 'bundle_manifest.json').read_text())
records = []
for case in manifest['cases']:
    path = ROOT / case['image']
    boundary = 'catbreed-review-boundary'
    body = ('--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="' + path.name + '"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + path.read_bytes() + ('\r\n--' + boundary + '--\r\n').encode()
    request = urllib.request.Request('http://127.0.0.1:18000/predict', data=body, headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    from PIL import Image
    for field in ('heatmap', 'cam_input'):
        assert Image.open(io.BytesIO(base64.b64decode(result[field]))).size == (224, 224)
    assert len(result['top5']) == len(result['ood_all']) == 5
    assert result['model']['device'] == 'cuda' and not result['is_mock']
    (output / (case['kind'] + '_http.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    records.append({**case, 'status': result['status'], 'top_label': result['top_label'], 'model': result['model']})
files = ['app/api/real_model.py', *[str(p.relative_to(ROOT)) for p in (ROOT / 'src/catbreed_core').glob('*.py')]]
report = {'codes': codes, 'health': health, 'cases': records, 'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}, 'chapter5_ready': False, 'pending': ['Human browser screenshots', 'Document two Mahalanobis failures and source verification limits']}
(output / 'review_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
if codes['mandatory_20'] != 0 or codes['benchmark'] != 0 or codes['all_validation'] not in (0, 1):
    raise SystemExit(1)
print(json.dumps({'codes': codes, 'http_cases': len(records)}))
