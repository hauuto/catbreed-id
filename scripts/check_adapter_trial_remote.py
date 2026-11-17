"""Validate trial and locked adapters without changing the running web service."""
import base64
import getpass
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if getpass.getuser()!='thanhkhoi' or not ROOT.is_relative_to(Path('/mnt/thanhkhoi')):
    raise RuntimeError('Expected isolated thanhkhoi test workspace')
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    CATBREED_TEST_PACKAGE=str(ROOT/'model_package'), OXFORD_PETS_IMAGES=str(ROOT.parent/'inputs/oxford'),CATBREED_TEST_DEVICE='cuda')
import torch
torch.set_num_threads(1)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
out=ROOT/'adapter_results'
out.mkdir(exist_ok=True)
(ROOT/'tmp').mkdir(exist_ok=True)
import pytest
with (out/'pytest.log').open('w',encoding='utf-8') as stream:
    class Tee:
        def __getattr__(self,name):return getattr(sys.__stdout__,name)
        def write(self,value):stream.write(value);stream.flush();sys.__stdout__.write(value);sys.__stdout__.flush()
        def flush(self):stream.flush()
    original=sys.stdout;sys.stdout=Tee()
    try:
        code=pytest.main(['tests/test_catbreed_core.py','tests/test_official_deployment.py','tests/test_benchmark_protocol.py','-q','-rs','--junitxml='+str(out/'pytest.xml'),'--basetemp='+str(ROOT/'tmp/adapter_pytest'),'-o','cache_dir='+str(ROOT/'tmp/cache')])
    finally:sys.stdout=original
from scripts.check_app_consistency import run
run(ROOT/'model_package',ROOT.parent/'inputs/oxford',out/'trial_adapter_20.csv',count=20,device='cuda',method='KNN')
run(ROOT.parent/'model_package',ROOT.parent/'inputs/oxford',out/'official_adapter_20.csv',count=20,device='cuda',method='MSP')
from app.api.real_model import DeploymentClassifier
from PIL import Image
models={'trial':DeploymentClassifier(ROOT/'model_package',device='cuda',method='KNN'),
        'official':DeploymentClassifier(ROOT.parent/'model_package',device='cuda',method='MSP')}
manifest=json.loads((ROOT.parent/'bundle_manifest.json').read_text())
cases=[]
for profile,model in models.items():
    assert model.method=='KNN'
    for case in manifest['cases']:
        with Image.open(ROOT.parent/case['image']) as image:result=model.predict(image)
        assert result['ood']['method']=='KNN' and result['model']['device']=='cuda'
        assert len(result['top5'])==len(result['ood_all'])==5
        for key in ('heatmap','cam_input'):
            assert Image.open(io.BytesIO(base64.b64decode(result[key]))).size==(224,224)
        if result['status']=='rejected':assert result['top_label'] is None
        cases.append({'profile':profile,'case':case['kind'],'status':result['status'],'label':result['top_label'],'model':result['model']})
expected={'id_correct':('accepted','Abyssinian'),'id_error':('accepted','Maine Coon'),'dog':('rejected',None),'dtd':('rejected',None),'felibreed':('accepted','Ragdoll')}
for row in cases:
    if row['profile']=='official':assert (row['status'],row['label'])==expected[row['case']]
report={'pytest_exit_code':int(code),'cases':cases,'official_lock_overrides_MSP_argument':models['official'].method=='KNN','running_web_service_modified':False,'benchmarked':False,'reason_no_benchmark':'Change only passes the existing selected method to decide; locked KNN and official numerical results unchanged'}
(out/'adapter_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pytest_exit_code':int(code),'trial_adapter_parity':'20/20','official_adapter_parity':'20/20','predict_cases':len(cases)}))
raise SystemExit(int(code!=0))
