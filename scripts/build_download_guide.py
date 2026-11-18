
from _project import ROOT, prepare_notebook
import ast
import json
from pathlib import Path

cells = []
def add(kind, source):
    cell = dict(cell_type=kind, id=f'download-guide-{len(cells):02d}', metadata={}, source=source.strip().splitlines(True))
    if kind == 'code':
        ast.parse(source.strip())
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
def md(text): add('markdown', text)
def code(text): add('code', text)

md('''# PaddleOCR-VL-1.6 다운로드 및 실행 환경 준비

대상: `AI_Challenge2_PaddleOCR_VL16_halftrain_analysis.ipynb`를 실행할 Jupyter 커널.

이 가이드는 **공식 모델 다운로드 → 파일 검증 → 패키지 버전 확인**까지 수행합니다. 학습·추론·데이터 업로드는 하지 않습니다.
Paddle 단독 실험은 Paddle 모델만, `ocr_llm` 파이프라인은 Qwen3.5-4B도 필요합니다.
이 PC에는 두 모델을 이미 다운로드했습니다. 동일 revision을 사용하면 Hub 다운로드 캐시를 재사용합니다.

공식 참고:
- [Hugging Face 다운로드 가이드](https://huggingface.co/docs/huggingface_hub/guides/download)
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

**권장 순서:** 1 설정 → 2 설치(필요할 때) → 커널 재시작 → 1 설정 재실행 → 3 점검 → 4 다운로드 → 5 검증 → 6 실행 환경 점검.
패키지 설치 셀은 기본 비활성입니다. 모델 다운로드 셀 실행에는 인터넷이 필요하며, 파일 검증은 오프라인에서도 가능합니다.
''')
md('''## 1. 경로와 다운로드할 모델 선택
이 PC 경로가 기본입니다. 다른 PC/Colab에서는 PROJECT_DIR을 데이터와 분석 노트북이 있는 폴더로 변경하세요.
아래 revision은 2026-09-21 이 PC에 실제 다운로드한 커밋으로 고정했습니다. 최신 버전이 필요하면 `main`으로 변경하되 새 revision은 별도 기록하세요.
''')
code('''import os, sys, json, hashlib, subprocess
from pathlib import Path

PROJECT_DIR = Path("C:/ssafy-16-2-challenge")
MODEL_ROOT = PROJECT_DIR / "downloads/models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
DOWNLOAD_QWEN = True  # Paddle 단독 direct_vqa만 실행하면 False
MODELS = {
    "PaddlePaddle/PaddleOCR-VL-1.6": "c5630abae1d940eafe0697512a0325494b02ab42",
}
if DOWNLOAD_QWEN:
    MODELS["Qwen/Qwen3.5-4B"] = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# 이전 분석 노트북에서 켜 둔 offline 환경변수를 새 다운로드 프로세스에 전달하지 않습니다.
DOWNLOAD_ENV = dict(os.environ)
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    DOWNLOAD_ENV.pop(key, None)
DOWNLOAD_ENV["HF_XET_CACHE"] = str(PROJECT_DIR / "downloads/hf_xet_cache")
DOWNLOAD_ENV["PYTHONIOENCODING"] = "utf-8"
print("현재 커널 Python:", sys.executable)
print("저장 경로:", MODEL_ROOT.resolve())
print("대상:", list(MODELS))
''')
md('''## 2. 다운로드 도구 / Transformers 설치
`INSTALL_PACKAGES=True`로 바꾸면 **현재 커널의 Python**에 설치합니다.
- `INSTALL_PROFILE="download_only"`: 다운로드 도구만 준비.
- `INSTALL_PROFILE="analysis"`: 분석 노트북용 Transformers 5.x 및 답변 모델 양자화 의존성도 준비.
- `INSTALL_SOURCE="offline"`: 이 PC에 준비된 wheel 사용. wheel은 Windows Python 3.12 기준이며 다른 환경에서는 온라인 설치 또는 해당 플랫폼용 wheel이 필요합니다.
- `INSTALL_SOURCE="online"`: PyPI에서 설치. 모델 가중치는 이후 Hugging Face에서 받습니다.

현재 오류였던 Transformers 4.57.6은 모델 다운로드만으로 바뀌지 않습니다. 기본 analysis 프로필은 실제 다운로드한 5.17.0으로 고정합니다.
torch/CUDA 자체는 이 셀의 직접 설치 대상이 아닙니다. 기존 torch 의존성이 충족되지 않으면 pip가 추가 설치를 요청할 수 있습니다.
**설치 성공 후 반드시 Kernel → Restart Kernel**, `INSTALL_PACKAGES=False`로 되돌리고 1번부터 다시 실행하세요.
''')
code('''INSTALL_PACKAGES = False
INSTALL_PROFILE = "analysis"  # 또는 download_only
INSTALL_SOURCE = "offline"  # 다른 PC/Colab은 online

if INSTALL_PACKAGES:
    if INSTALL_PROFILE == "download_only":
        packages = ["huggingface_hub==1.32.0"]
    elif INSTALL_PROFILE == "analysis":
        packages = ["transformers==5.17.0", "huggingface_hub==1.32.0"]
        if DOWNLOAD_QWEN:
            packages += ["accelerate", "bitsandbytes"]
    else:
        raise ValueError("INSTALL_PROFILE: download_only 또는 analysis")
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed"]
    if INSTALL_SOURCE == "offline":
        folders = [PROJECT_DIR / "downloads/libs_transformers5", PROJECT_DIR / "downloads/libs"]
        command += ["--no-index"]
        for folder in folders:
            if not folder.is_dir():
                raise FileNotFoundError(f"wheel 폴더 없음: {folder}")
            command += ["--find-links", str(folder)]
    elif INSTALL_SOURCE != "online":
        raise ValueError("INSTALL_SOURCE: offline 또는 online")
    subprocess.check_call(command + packages)
    print("설치 완료. 커널을 재시작한 뒤 INSTALL_PACKAGES=False로 되돌리고 1번부터 실행하세요.")
else:
    print("설치 건너뜀. 필요한 경우에만 INSTALL_PACKAGES=True로 변경하세요.")
''')
md('''## 3. 다운로드 도구 점검
별도 자식 프로세스를 사용하여 이전에 import된 Hub의 offline 설정 영향을 피합니다.
이 PC의 `downloads/hf_download_tools`는 도구가 커널에 없을 때만 사용하는 보조 경로입니다.
버전 변경 이후 분석 모델을 import하기 전 커널 재시작은 여전히 필요합니다.
''')
code('''bootstrap = """
import os, sys, json
from pathlib import Path
try:
    import huggingface_hub
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    for name in list(sys.modules):
        if name == 'huggingface_hub' or name.startswith('huggingface_hub.'):
            del sys.modules[name]
    sys.path.insert(0, sys.argv[1])
    import huggingface_hub
    from huggingface_hub import HfApi, snapshot_download
"""
TOOL_DIR = PROJECT_DIR / "downloads/hf_download_tools"
def run_download_process(program):
    subprocess.run([sys.executable, "-u", "-c", bootstrap + program, str(TOOL_DIR)],
                   env=DOWNLOAD_ENV, check=True)
run_download_process("\\nprint('huggingface_hub:', huggingface_hub.__version__, flush=True)\\n")
''')
md('''## 4. 공식 저장소에서 다운로드
`snapshot_download(local_dir=...)`를 사용하여 config·processor·tokenizer·가중치를 함께 받습니다.
다운로드 전 파일 총용량과 여유 공간을 표시합니다. 이미 받은 파일도 총용량에는 포함됩니다.
완료한 파일은 재사용되며 중단 시 같은 셀을 다시 실행할 수 있습니다. `.cache/huggingface` 메타데이터는 유지하세요.
다운로드는 공개 저장소 읽기만 수행합니다. 토큰을 노트북에 직접 적을 필요는 없습니다.
''')
code('''download_program = r"""
import shutil
from huggingface_hub import HfApi, snapshot_download
settings = json.loads(SETTINGS_JSON)
root = Path(settings['root'])
api = HfApi()
for repo, revision in settings['models'].items():
    info = api.model_info(repo, revision=revision, files_metadata=True)
    dest = root / repo.split('/')[-1]
    total = sum(item.size or 0 for item in info.siblings)
    free = shutil.disk_usage(root).free
    print(f'[{repo}] revision={info.sha}', flush=True)
    print(f'Repository size: {total / 1024**3:.2f} GiB; free disk: {free / 1024**3:.2f} GiB', flush=True)
    snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(dest), max_workers=4)
    files = []
    for item in info.siblings:
        path = dest / item.rfilename
        if not path.is_file():
            raise RuntimeError(f'Missing file: {path}')
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(f'Size mismatch: {path}')
        files.append({'path': item.rfilename, 'bytes': path.stat().st_size})
    (dest / 'download_manifest.json').write_text(json.dumps(
        {'repo_id': repo, 'revision': info.sha, 'files': files}, indent=2), encoding='utf-8')
    print(f'COMPLETE: {dest}', flush=True)
"""
settings = json.dumps({"root": str(MODEL_ROOT.resolve()), "models": MODELS})
run_download_process("\\nSETTINGS_JSON = " + repr(settings) + "\\n" + download_program)
''')
md('''## 5. 다운로드 파일 오프라인 검증
manifest와 파일 크기, config, tokenizer, 가중치 shard를 확인합니다. 이는 모델 로딩/정확도 검증과 다릅니다.
이 PC처럼 이미 다운로드한 경우 4번을 생략하고 이 셀을 실행할 수 있습니다.
''')
code('''for repo, revision in MODELS.items():
    folder = MODEL_ROOT / repo.split("/")[-1]
    manifest = json.loads((folder / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo_id"] == repo
    if revision != "main":
        assert manifest["revision"] == revision, "다운로드 revision이 설정과 다릅니다."
    for item in manifest["files"]:
        path = folder / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"], f"파일 누락/크기 불일치: {path}"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert (folder / "tokenizer_config.json").is_file()
    assert (folder / "tokenizer.json").is_file()
    weights = list(folder.glob("*.safetensors"))
    assert weights, f"가중치 없음: {folder}"
    for index in folder.glob("*.safetensors.index.json"):
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        assert all((folder / shard).is_file() for shard in shards)
    print(repo, "OK", config.get("architectures"), "revision:", manifest["revision"])
''')
md('''## 6. 분석 커널 상태 확인
모델 다운로드는 GPU 없이도 가능합니다. 이 단계만 기존 분석 노트북의 CUDA 환경을 점검합니다.
설치된 배포판 버전과 현재 import된 버전이 다르면 반드시 커널을 재시작하세요.
''')
code('''from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

for package in ["transformers", "torch", "accelerate", "bitsandbytes"]:
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, "미설치")
import transformers
required = "5.3.0" if DOWNLOAD_QWEN else "5.0.0"
print("현재 import 버전:", transformers.__version__)
print("설치 위치:", transformers.__file__)
assert Version(transformers.__version__) >= Version(required), "2번 설치 셀 실행 후 커널 재시작 필요"
from transformers import AutoProcessor, AutoModelForImageTextToText
if DOWNLOAD_QWEN:
    from transformers import Qwen3_5ForConditionalGeneration
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("다운로드는 완료할 수 있지만 분석 노트북 실행에는 CUDA 환경 설정이 필요합니다.")
''')
md('''## 7. 분석 노트북에 연결
분석 노트북 첫 설정 셀에서 다음과 같이 지정하세요. 다른 PC에서는 실제 경로로 변경합니다.
```python
MODEL_ROOT = Path("C:/ssafy-16-2-challenge/downloads/models")
OCR_MODEL_DIR = MODEL_ROOT / "PaddleOCR-VL-1.6"
ANSWER_MODEL_DIR = MODEL_ROOT / "Qwen3.5-4B"
RUN_MODES = ["ocr_llm"]
SMOKE_TEST = True
```
먼저 소량 실행으로 실제 모델 로딩·추론을 확인하고, 성공하면 `SMOKE_TEST=False`로 본 평가를 실행하세요.

### 오류별 대응
| 오류 | 대응 |
|---|---|
| Transformers=4.57.6 | 2번 analysis 설치 후 커널 재시작 |
| OfflineModeIsEnabled | 3~4번 셀의 새 프로세스로 다운로드; 분석용 offline 환경은 다운로드에 사용하지 않음 |
| No matching distribution | 현재 Python/OS에 맞는 wheel 확인; 이 PC의 wheel은 Windows Python 3.12용 |
| 401/403/429 또는 연결 실패 | 네트워크/접근권한/요청 제한 확인; 토큰이 필요하면 환경변수 또는 별도 로그인 사용 |
| config/가중치 누락 | 4번 재실행 후 5번 검증 |
| CUDA=False/OOM | 다운로드 오류와 별개; CUDA 설치 및 GPU 메모리 확인 |

모델 파일 검증 완료가 GPU 추론 성공을 뜻하지는 않습니다. 분석 커널의 설치/재시작 상태는 6번 셀에서 확인하세요.
''')
nb = dict(cells=cells, metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
                               'language_info': {'name':'python'}}, nbformat=4, nbformat_minor=5)
target = (ROOT / 'notebooks/setup/AI_Challenge2_PaddleOCR_VL16_download_guide.ipynb')
prepare_notebook(nb)
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(target.resolve(), len(cells), 'cells; AST valid')
