"""Create the E7 Colab notebook: LoRA on Qwen3.5-27B for one training configuration.

사용: python scripts/build_e7_colab_notebook.py A|B|C   (먼저 make_e7_trainset.py로 목록을 만드세요)

정답이 사람 합의라면 E1 오답 분류의 "라벨 오류·복수 정답"도 일반인 상식·눈에 띄는 쪽·맥락 해석이라는
학습 가능한 패턴입니다. 제로샷은 이 패턴을 알 수 없으므로 train의 정답으로 학습시킵니다.

흐름: 학습(Drive 체크포인트로 여러 세션에 걸쳐 재개) → 어댑터를 bf16 원본에 합쳐 저장
→ vLLM으로 holdout 1,300 평가, E6 제로샷과 McNemar → (선택) test 제출.

src/common_eval.py·vlm_runtime.py·qlora.py·vllm_eval.py 원문, 학습 구성 목록, holdout ID를 노트북에 넣습니다.
Colab은 비공개 저장소를 clone할 수 없고, 넣어 두면 로컬과 다른 버전을 쓸 수 없습니다.
"""

from _project import ROOT
import ast
import hashlib
import json
import sys

import pandas as pd

config = sys.argv[1] if len(sys.argv) == 2 else None
if config not in {"A", "B", "C"}:
    raise SystemExit("사용: python scripts/build_e7_colab_notebook.py A|B|C")
manifest_path = ROOT / "splits" / f"e7_train_{config}.csv"
if not manifest_path.is_file():
    raise SystemExit(f"{manifest_path}가 없습니다. 먼저 scripts/make_e7_trainset.py {config}를 실행하세요")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": text.strip().splitlines(keepends=True)})


SRC_FILES = ["common_eval.py", "vlm_runtime.py", "qlora.py", "vllm_eval.py"]
SOURCES = {name: (ROOT / "src" / name).read_text(encoding="utf-8") for name in SRC_FILES}
SOURCE_SHA = {name: hashlib.sha256(s.encode("utf-8")).hexdigest() for name, s in SOURCES.items()}

manifest = pd.read_csv(manifest_path)
num = lambda i: int(i.split("_")[1].removesuffix(".jpg"))
TRAIN_NUMS = [num(i) for i in manifest[manifest.source == "train"].id]
DEV_NUMS = [num(i) for i in manifest[manifest.source == "dev"].id]
split = pd.read_csv(ROOT / "splits/e6_split.csv")
HOLDOUT_NUMS = [num(i) for i in split[split.split == "holdout"].id]
VAL672_NUMS = [num(i) for i in split[split.in_val672].id]
MANIFEST_SHA = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
DESCRIPTION = {"A": "E6 학습 분할 5,414 전부",
               "B": "A에서 결함 13문항(선명한 글자와 정답이 어긋남, 전제가 틀림 등. splits/e7_review_highconf.csv)을 뺀 것",
               "C": "A + dev에서 5명 중 3표 이상 단독 최다 답이 있는 문항"}[config]

md(f'''
# E7-{config} Colab — Qwen3.5-27B LoRA 학습·평가 (A100 80GB)

**학습 구성 {config}:** {DESCRIPTION} — **{len(manifest):,}문항** (train {len(TRAIN_NUMS):,} / dev {len(DEV_NUMS):,})

정답은 여러 교육생이 일치해서 고른 답입니다. 제로샷 모델은 사진 속 사실은 잘 맞히지만
일반인 상식·눈에 띄는 쪽·맥락 해석 같은 **사람의 해석 방식**은 모릅니다. 그것을 train 정답으로 학습시킵니다.

| 단계 | 내용 | 시간 (추정) |
| --- | --- | ---: |
| 설치·데이터·모델 다운로드 | | 15~20분 |
| **학습** | 1 epoch, 유효 배치 16, 체크포인트는 Drive에 | **2.5~4시간** (첫 스텝 뒤 실측 ETA 출력) |
| 병합 | LoRA를 bf16 원본에 합쳐 로컬 디스크에 저장 | 약 10분 |
| 평가 | vLLM으로 holdout 1,300, E6 제로샷과 McNemar | 약 15분 (적재 포함) |
| 제출 (선택) | test 6,714 | 약 22분 |

**세션이 끊겨도 됩니다.** 학습은 20스텝(약 320문항)마다 Drive에 체크포인트를 남기고,
다시 실행하면 거기서 이어 갑니다. 여러 세션에 나눠 돌려도 결과는 같습니다.

## 준비물

- A100 80GB 런타임
- Drive의 `MyDrive/SSAFY/AI_Challenge2/ssafy-16-2-ai.zip`
- 평가 비교용: E6 결과 `MyDrive/SSAFY/AI_Challenge2/output/exp_E6_zeroshot_train/` (없으면 비교만 건너뜀)

공통 코드·학습 구성 목록·holdout ID는 노트북 안에 있습니다.
''')

md('''
## 1. 설치

새 런타임에서 가장 먼저 실행하고, 끝나면 **런타임 → 세션 다시 시작** 후 2번 셀부터 실행하세요.
`flash-linear-attention`은 Qwen3.5의 linear attention 층을 빠르게 학습시키는 커널입니다(없으면 매우 느림).
''')

code('''
!pip -q install -U vllm
!pip -q install -U peft bitsandbytes flash-linear-attention "huggingface_hub[hf_xet]" pillow pandas
# Colab 기본 패키지 중 두 개가 충돌합니다. 이 작업에는 둘 다 필요 없으므로 지웁니다.
# - torchaudio: vLLM이 바꾼 torch와 CUDA 버전이 어긋나 transformers import가 실패
# - torchao 0.10: peft가 0.16 미만을 발견하면 LoRA를 붙일 때 ImportError
!pip -q uninstall -y torchaudio torchao
''')

code('''
import gc, json, math, os, random, re, shutil, sys, time
from pathlib import Path

import pandas as pd
import torch
import torchvision
import transformers
import peft
import vllm

print("vllm", vllm.__version__, "/ transformers", transformers.__version__, "/ peft", peft.__version__)
print("torch", torch.__version__, f"(CUDA {torch.version.cuda})", "/ torchvision", torchvision.__version__)
try:
    import torchaudio  # noqa: F401
    raise RuntimeError("torchaudio가 남아 있습니다. `!pip uninstall -y torchaudio` 후 세션을 다시 시작하세요")
except ImportError:
    pass
try:
    import torchao  # noqa: F401
    raise RuntimeError("torchao가 남아 있습니다. `!pip uninstall -y torchao` 후 세션을 다시 시작하세요")
except ImportError:
    pass
try:
    import fla  # noqa: F401
    print("flash-linear-attention: 설치됨")
except ImportError:
    print("경고: flash-linear-attention이 없어 학습이 매우 느립니다")
assert torch.cuda.is_available(), "GPU 런타임을 선택하세요"
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)} / {GPU_GIB:.1f} GiB")
assert GPU_GIB >= 70, "27B bf16 학습은 A100 80GB가 필요합니다"
''')

md('''
## 2. Drive · 데이터 · 공통 코드
''')

code(f'''
from google.colab import drive
drive.mount("/content/drive")

DRIVE_BASE = Path("/content/drive/MyDrive/SSAFY/AI_Challenge2")
ZIP_PATH = DRIVE_BASE / "ssafy-16-2-ai.zip"
ZIP_PASSWORD = "ocrisnotenough"
assert ZIP_PATH.is_file(), f"zip을 찾지 못했습니다: {{ZIP_PATH}}"


def find_data_root():
    for parent, dirs, files in os.walk("/content"):
        dirs[:] = [d for d in dirs if d not in {{"drive", ".cache", "sample_data", "models"}}]
        if "test.csv" in files and "train.csv" in files:
            return Path(parent)
    return None


if find_data_root() is None:
    import subprocess
    subprocess.run(["7z", "x", "-y", f"-p{{ZIP_PASSWORD}}", "-o/content/", str(ZIP_PATH)],
                   check=True, stdout=subprocess.DEVNULL)
DATA_ROOT = find_data_root()
assert DATA_ROOT is not None and (DATA_ROOT / "dev.csv").is_file(), "train/test/dev.csv를 찾지 못했습니다"

SOURCES = {SOURCES!r}
SOURCE_SHA = {SOURCE_SHA!r}
import hashlib
SRC_DIR = Path("/content/src"); SRC_DIR.mkdir(parents=True, exist_ok=True)
for name, text in SOURCES.items():
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == SOURCE_SHA[name], name
    (SRC_DIR / name).write_text(text, encoding="utf-8")
sys.path.insert(0, str(SRC_DIR))
import common_eval as ce
import vlm_runtime as vr
import qlora
import vllm_eval as ve

ce.DATA_DIR = DATA_ROOT
ce._self_test(); vr._self_test(); qlora._self_test(); ve._self_test()
print("데이터:", DATA_ROOT)
''')

md('''
## 3. 설정

`QUANT="bf16"`이 기본입니다. A100 80GB에 27B bf16(약 54 GiB)이 들어가고, 4bit보다 학습이 빠르며
제출 추론(vLLM bf16)과 조건이 같습니다. 메모리가 부족하면 `"nf4"`(QLoRA)로 바꾸세요.

`CONSENSUS`는 E6 3단계에서 합의 지시문이 holdout에서 유의하게 이겼을 때만 `True`로 두세요.
학습과 평가가 같은 지시문을 씁니다.
''')

code(f'''
CONFIG = "{config}"
QUANT = "bf16"               # "bf16"(LoRA) 또는 "nf4"(QLoRA)
CONSENSUS = False            # E6에서 합의 지시문이 이겼을 때만 True
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR = 1e-4
GRAD_ACCUM = 16
EPOCHS = 1
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 20              # 옵티마이저 스텝. 20스텝 = 약 320문항
MAX_TRAIN_ROWS = None        # int면 앞 N문항만 학습(배관 점검용). 결과를 비교에 쓰지 마세요
MAX_NEW_TOKENS = 24

RUN_NAME = f"exp_E7_27b_{{CONFIG}}" + ("_cons" if CONSENSUS else "") + ("_smoke" if MAX_TRAIN_ROWS else "")
DRIVE_RUN = DRIVE_BASE / "output" / RUN_NAME
CKPT_DIR = DRIVE_RUN / "checkpoint"
ADAPTER_DIR = DRIVE_RUN / "adapter"
TRAIN_LOG = DRIVE_RUN / "train_log.jsonl"
BASE_DIR = Path("/content/models/Qwen3.5-27B")
MERGED_DIR = Path("/content/merged") / RUN_NAME
DRIVE_RUN.mkdir(parents=True, exist_ok=True)
DTYPE = torch.bfloat16
torch.manual_seed(42)
print("실행 이름:", RUN_NAME, "/ Drive:", DRIVE_RUN)
''')

md(f'''
## 4. 학습 데이터

구성 목록(`splits/e7_train_{config}.csv`, SHA-256 `{MANIFEST_SHA[:16]}…`)에는 id만 있고,
정답은 train.csv(train 문항)와 dev.csv 다수결(dev 문항, 3표 이상 단독 최다)로 붙입니다.
holdout 문항이 하나라도 섞이면 멈춥니다.
''')

code(f'''
TRAIN_NUMS = {TRAIN_NUMS}
DEV_NUMS = {DEV_NUMS}
HOLDOUT_NUMS = {HOLDOUT_NUMS}
VAL672_NUMS = {VAL672_NUMS}
manifest = pd.DataFrame(
    [{{"id": f"train_{{n:04d}}.jpg", "source": "train"}} for n in TRAIN_NUMS]
    + [{{"id": f"dev_{{n:04d}}.jpg", "source": "dev"}} for n in DEV_NUMS])
HOLDOUT = {{f"train_{{n:04d}}.jpg" for n in HOLDOUT_NUMS}}
VAL672 = {{f"train_{{n:04d}}.jpg" for n in VAL672_NUMS}}
assert len(HOLDOUT) == 1300 and VAL672 <= HOLDOUT

train_df = pd.read_csv(DATA_ROOT / "train.csv", keep_default_na=False)
dev_df = pd.read_csv(DATA_ROOT / "dev.csv", keep_default_na=False)
fit_rows = qlora.rows_from_manifest(manifest, train_df, dev_df, HOLDOUT)
random.Random(42).shuffle(fit_rows)
order = []
for epoch in range(EPOCHS):
    rows = fit_rows[:MAX_TRAIN_ROWS] if MAX_TRAIN_ROWS else fit_rows
    order += random.Random(42 + epoch).sample(rows, len(rows))
print(f"학습 {{len(fit_rows)}}문항 / 정답 분포 {{pd.Series([r['answer'] for r in fit_rows]).value_counts().sort_index().to_dict()}}")
print(f"옵티마이저 스텝 {{len(order) // GRAD_ACCUM}} (유효 배치 {{GRAD_ACCUM}}, epoch {{EPOCHS}})")
''')

md('''
## 5. 원본 모델 다운로드
''')

code('''
from huggingface_hub import snapshot_download

if not (BASE_DIR / "config.json").is_file():
    snapshot_download(repo_id="Qwen/Qwen3.5-27B", local_dir=str(BASE_DIR))
vr.check_weights(BASE_DIR)
print("원본 준비:", BASE_DIR)
''')

md('''
## 6. 학습

첫 스텝과 10스텝마다 손실과 남은 시간을 출력합니다. 손실이 처음부터 작은 게 정상입니다
(제로샷이 이미 97% 가까이 맞힘). 학습 신호는 모델과 사람이 어긋나는 문항에서 오므로 `max`(16문항 중
가장 큰 손실)가 줄어드는지를 보세요.

**끊기면:** 2·3·4·5번 셀을 다시 실행한 뒤 이 셀을 실행하면 Drive 체크포인트에서 이어집니다.
이미 학습이 끝났으면(`adapter/`가 있으면) 건너뜁니다.
''')

code('''
if (ADAPTER_DIR / "adapter_config.json").is_file():
    print("학습이 이미 끝났습니다:", ADAPTER_DIR)
else:
    model, proc = vr.load_vlm(BASE_DIR, four_bit=(QUANT == "nf4"), dtype=DTYPE)
    model = qlora.attach_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    try:
        qlora.train_loop(model, proc, order, grad_accum=GRAD_ACCUM, lr=LR, warmup_ratio=WARMUP_RATIO,
                         max_grad_norm=MAX_GRAD_NORM, save_every=SAVE_EVERY, ckpt_dir=CKPT_DIR,
                         log_path=TRAIN_LOG, consensus=CONSENSUS)
        model.save_pretrained(str(ADAPTER_DIR))
        (DRIVE_RUN / "train_config.json").write_text(json.dumps(dict(
            config=CONFIG, quant=QUANT, consensus=CONSENSUS, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, lr=LR, grad_accum=GRAD_ACCUM, epochs=EPOCHS,
            train_rows=len(fit_rows), max_train_rows=MAX_TRAIN_ROWS,
            peak_gib=torch.cuda.max_memory_allocated() / 1024**3), indent=2), encoding="utf-8")
        print(f"학습 완료 / GPU peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB / 어댑터: {ADAPTER_DIR}")
    finally:
        del model, proc
        gc.collect(); torch.cuda.empty_cache()
''')

md('''
## 7. 병합

어댑터를 bf16 원본에 합쳐 로컬 디스크(`/content/merged/…`, 약 54 GiB)에 저장합니다. vLLM이 이 폴더를
그대로 올립니다. Drive에는 올리지 않습니다(어댑터만 있으면 언제든 다시 합칠 수 있음).
''')

code('''
if (MERGED_DIR / "config.json").is_file():
    print("이미 병합됨:", MERGED_DIR)
else:
    t0 = time.time()
    qlora.merge_and_save(BASE_DIR, ADAPTER_DIR, MERGED_DIR, dtype=DTYPE)
    gc.collect(); torch.cuda.empty_cache()
    print(f"병합 {time.time() - t0:.0f}초 ->", MERGED_DIR)
''')

md('''
## 8. 평가 — holdout 1,300

**vLLM 적재에서 메모리 오류가 나면** *런타임 → 세션 다시 시작* 후 2·3·4번 셀과 이 셀만 실행하세요.
병합한 모델은 로컬 디스크에 남아 있습니다.

비교 기준은 E6 1단계(27B 제로샷)의 같은 holdout 예측입니다(Drive `output/exp_E6_zeroshot_train`).
합의 지시문으로 학습했다면 E6 3단계(`exp_E6_consensus_train`)와 비교합니다.
''')

code('''
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

llm = LLM(model=str(MERGED_DIR), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90,
          limit_mm_per_prompt={"image": 1}, seed=42)
proc = AutoProcessor.from_pretrained(str(MERGED_DIR))
SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, logprobs=20, seed=42)

gold = train_df[train_df.id.isin(HOLDOUT)]
tuned = ve.run(llm, proc, SAMPLING, gold.drop(columns=["answer"]).to_dict("records"),
               Path("/content/out") / RUN_NAME / "holdout", DRIVE_RUN / "holdout",
               consensus=CONSENSUS, desc="holdout 1,300")
t = gold[["id", "answer"]].merge(tuned[["id", "pred", "conf"]], on="id", validate="one_to_one")
print(f"\\nE7-{CONFIG} holdout {(t.pred == t.answer).mean():.4f} ({int((t.pred == t.answer).sum())}/1300)"
      f" / val 672 {(t[t.id.isin(VAL672)].pred == t[t.id.isin(VAL672)].answer).mean():.4f}"
      f" / INVALID {int((t.pred == 'INVALID').sum())}")

base_name = "exp_E6_consensus_train" if CONSENSUS else "exp_E6_zeroshot_train"
base_path = DRIVE_BASE / "output" / base_name / "predictions.jsonl"
if base_path.is_file():
    base = ve.read(base_path)
    ve.paired(gold, base[base.id.isin(HOLDOUT)], tuned, "E6 제로샷", f"E7-{CONFIG}")
    ve.paired(gold[gold.id.isin(VAL672)], base, tuned, "E6 제로샷(val 672)", f"E7-{CONFIG}(val 672)")
else:
    print("E6 결과가 없어 비교를 건너뜁니다:", base_path)
''')

md('''
## 9. 제출 (선택)

위 비교에서 E7이 E6 제로샷보다 **holdout에서 유의하게 좋을 때만** `SUBMIT = True`로 두고 실행하세요.
test 6,714문항을 같은 모델로 풀어 Drive `output/<실행 이름>/test/`에 제출 파일을 만듭니다(약 22분).
''')

code('''
SUBMIT = False

if not SUBMIT:
    print("SUBMIT = False: 제출을 건너뜁니다")
else:
    test_df = pd.read_csv(DATA_ROOT / "test.csv", keep_default_na=False)
    sample_df = pd.read_csv(DATA_ROOT / "sample_submission.csv", keep_default_na=False)
    assert list(test_df.id) == list(sample_df.id)
    res = ve.run(llm, proc, SAMPLING, test_df.to_dict("records"),
                 Path("/content/out") / RUN_NAME / "test", DRIVE_RUN / "test", consensus=CONSENSUS,
                 desc="test 6,714")
    res, n_fill = ve.fill_invalid(res)
    sub = test_df[["id"]].merge(res[["id", "pred"]], on="id", how="left", validate="one_to_one")
    assert sub.pred.isin(ce.CHOICES).all() and list(sub.id) == list(sample_df.id)
    share = sub.pred.value_counts(normalize=True).reindex(ce.CHOICES).fillna(0)
    print(f"폴백 {n_fill}건 / 예측 분포", share.round(3).to_dict())
    name = f"submission_qwen35-27b_lora-{CONFIG}{'-cons' if CONSENSUS else ''}-bf16-vllm.csv"
    sub.rename(columns={"pred": "answer"}).to_csv(DRIVE_RUN / "test" / name, index=False, encoding="utf-8")
    print("제출 파일:", DRIVE_RUN / "test" / name)
''')

md('''
## 끝나면

Drive `output/<실행 이름>/`(train_log.jsonl, train_config.json, holdout/, adapter/)를 로컬 `output/`으로 받아 주세요.
`adapter/`는 수백 MB라 필요할 때만 받으면 됩니다. 구성끼리의 비교(A vs C 등)는 두 실행의 holdout 예측으로 합니다.
''')

for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "colab": {"gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / f"notebooks/experiments/AI_Challenge2_expE7_lora27b_{config}_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
