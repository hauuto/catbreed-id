"""Offline trial parity and original core tests in an isolated thanhkhoi directory."""
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if getpass.getuser() != 'thanhkhoi' or not ROOT.is_relative_to(Path('/mnt/thanhkhoi')):
    raise RuntimeError('Expected thanhkhoi workspace')
os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.path.insert(0, str(ROOT))
for name, expected in json.loads((ROOT/'payload_sha256.json').read_text()).items():
    path=(ROOT/name).resolve()
    assert path.is_relative_to(ROOT) and hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
import numpy as np
from PIL import Image
import torch
torch.set_num_threads(1)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
from src.catbreed_core import load_package,predict_arrays,decide
package=ROOT/'model_package'
images=ROOT.parent/'inputs/oxford'
assert images.is_dir()
os.environ.update(CATBREED_TEST_PACKAGE=str(package), OXFORD_PETS_IMAGES=str(images), CATBREED_TEST_DEVICE='cuda')
out=ROOT/'results'
out.mkdir(exist_ok=True)
(ROOT/'tmp').mkdir(exist_ok=True)
# Run pytest in this interpreter so the CUDA precision settings apply to its real-data test.
import pytest
with (out/'pytest.log').open('w',encoding='utf-8') as stream:
    class Tee:
        def __getattr__(self, name):
            return getattr(sys.__stdout__, name)
        def write(self,text):
            stream.write(text); stream.flush(); sys.__stdout__.write(text); sys.__stdout__.flush()
        def flush(self): stream.flush()
    original=sys.stdout
    sys.stdout=Tee()
    try:
        code=pytest.main(['tests/test_catbreed_core.py','-q','-rs','--junitxml='+str(out/'pytest.xml'),'--basetemp='+str(ROOT/'tmp/pytest'),'-o','cache_dir='+str(ROOT/'tmp/cache')])
    finally: sys.stdout=original
p=load_package(package,device='cuda')
with np.load(package/'val.npz',allow_pickle=False) as saved:
    ids=saved['image_id'].astype(str)
    logits=saved['logits'].copy()
with (package/'validation_scores.csv').open(encoding='utf-8',newline='') as f:
    reference={row['image_id']:row for row in csv.DictReader(f)}
rows=[]
for i,image_id in enumerate(ids[:20]):
    with Image.open(images/(image_id+'.jpg')) as image:
        z,features,scores=predict_arrays(p,image,exif=False)
    decision=decide(p,scores,z,method='KNN')
    row={'image_id':image_id,'logits_max_abs':float(np.max(np.abs(z[0]-logits[i]))),'accepted':decision['accepted']}
    passed=row['logits_max_abs']<1e-4
    for method,values in scores.items():
        expected=float(reference[image_id][method]); actual=float(values[0])
        error=abs(actual-expected); relative=error/max(abs(expected),1e-12)
        row[method+'_abs']=error; row[method+'_relative']=relative
        passed=passed and (error<1e-4 or (method=='Mahalanobis' and relative<1e-6))
    row['passed']=bool(passed); rows.append(row)
with (out/'trial_consistency_20.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={'purpose':'13 November trial only; not official v2 report metrics','checkpoint_sha256':p.checkpoint_sha256,'run_id':p.saved['run_id'],'device':'cuda','torch':torch.__version__,'exif':False,'n':len(rows),'passed':sum(r['passed'] for r in rows),'pytest_exit_code':int(code),'max_logits_abs':max(r['logits_max_abs'] for r in rows),'max_scores_abs':{m:max(r[m+'_abs'] for r in rows) for m in scores},'offline_inputs':True,'test_assertions':'Original numeric assertions retained; local package and image-directory inputs added'}
(out/'trial_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
raise SystemExit(int(code!=0 or summary['passed']!=summary['n']))
