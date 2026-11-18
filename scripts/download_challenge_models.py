"""Download official checkpoints into the notebook's local model directories.

사용: python scripts/download_challenge_models.py [repo ...]
인자가 없으면 실험 A의 두 모델을 받습니다. E4는 `Qwen/Qwen3.5-27B`(약 52 GiB)를 넘기세요.
"""

from _project import ROOT
import os, sys, json, hashlib
from pathlib import Path

sys.path.insert(0, str(ROOT / 'downloads/hf_download_tools'))
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ.pop(key, None)
os.environ['HF_XET_CACHE'] = str(ROOT / 'downloads/hf_xet_cache')
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
repos = sys.argv[1:] or ['PaddlePaddle/PaddleOCR-VL-1.6', 'Qwen/Qwen3.5-4B']
for repo in repos:
    folder = ROOT / 'downloads/models' / repo.split('/')[-1]
    info = api.model_info(repo, files_metadata=True)
    print(f'Start {repo} revision={info.sha}', flush=True)
    snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(folder), max_workers=4)
    manifest = []
    for item in info.siblings:
        path = folder / item.rfilename
        if not path.is_file():
            raise RuntimeError(f'Missing: {path}')
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(f'Size mismatch: {path}')
        manifest.append({'path': item.rfilename, 'bytes': path.stat().st_size})
    indexes = list(folder.glob('*.safetensors.index.json'))
    for index in indexes:
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
        assert all((folder / shard).is_file() for shard in shards)
    assert (folder / 'config.json').is_file() and list(folder.glob('*.safetensors'))
    (folder / 'download_manifest.json').write_text(json.dumps(
        {'repo_id': repo, 'revision': info.sha, 'files': manifest}, indent=2), encoding='utf-8')
    print(f'COMPLETE {repo}: {len(manifest)} files, {sum(f["bytes"] for f in manifest)/1024**3:.2f} GiB', flush=True)
