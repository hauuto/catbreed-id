"""Fetch the two reference files for the SHA-verified November 13 trial package."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import argparse

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'C:\Users\vyx\Downloads\model_package_1311')
DEST = ROOT / 'outputs/reference_packages/model_package_1311'
REPO = 'catbreed-id/catbreed-experiments'
STAGE = 'runs/20261113_145326_efficientnet_b0_s42_df1edde1/frozen'
EXPECTED = 'd90bca482b4367579a70b3158ad62e331644cbc9731229a39b4c15233bfbe052'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(local_references=None):
    if sha(SOURCE / 'checkpoint.pt') != EXPECTED:
        raise ValueError('Local checkpoint SHA changed; refusing to select references')
    if local_references:
        sources = {name: Path(local_references) / name for name in ('val.npz', 'validation_scores.csv')}
    else:
        info = HfApi().get_paths_info(REPO, [STAGE + '/checkpoint.pt'], repo_type='dataset')
        if len(info) != 1 or not info[0].lfs or info[0].lfs.sha256 != EXPECTED:
            raise ValueError('Remote checkpoint SHA does not match local trial package')
        sources = {name: Path(hf_hub_download(REPO, STAGE + '/' + name, repo_type='dataset'))
                   for name in ('val.npz', 'validation_scores.csv')}
    with np.load(sources['val.npz'], allow_pickle=False) as data:
        ids = data['image_id'].astype(str).tolist()
        logits_shape = list(data['logits'].shape)
        features_shape = list(data['features'].shape)
        if len(ids) != len(set(ids)) or logits_shape != [len(ids), 12]:
            raise ValueError('Invalid validation arrays')
        if not np.isfinite(data['logits']).all() or not np.isfinite(data['features']).all():
            raise ValueError('Nonfinite reference arrays')
    with sources['validation_scores.csv'].open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [r['image_id'] for r in rows] != ids:
        raise ValueError('Reference CSV IDs differ from NPZ IDs')
    for row in rows:
        if not all(np.isfinite(float(row[name])) for name in ('MSP','MaxLogit','Energy','Mahalanobis','KNN')):
            raise ValueError('Invalid reference scores')
    DEST.mkdir(parents=True, exist_ok=True)
    payload = {**{name: SOURCE / name for name in ('checkpoint.pt','thresholds.json','calibration.json','distance_parameters.npz','README.txt')}, **sources}
    for name,path in payload.items():
        target = DEST / name
        if target.exists() and sha(target) != sha(path):
            raise ValueError('Refusing to overwrite different file: ' + name)
        shutil.copyfile(path,target)
    manifest = {'purpose':'November 13 trial references only; separate from official v2', 'source':str(SOURCE),
                'repository':REPO,'stage_dir':STAGE,'checkpoint_sha256':EXPECTED,
                'remote_checkpoint_sha_verified':not bool(local_references),
                'reference_source':str(local_references) if local_references else 'Hugging Face',
                'validation_images':len(ids),
                'logits_shape':logits_shape,'features_shape':features_shape,
                'files':{name:{'sha256':sha(DEST/name),'bytes':(DEST/name).stat().st_size} for name in payload}}
    (DEST/'reference_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'destination':str(DEST),'checkpoint_sha_verified':True,'n':len(ids),'downloaded':list(sources)},ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-references', type=Path)
    main(parser.parse_args().local_references)
