"""Create the E4 Colab notebook: Qwen3.5-27B bf16 with vLLM on an A100 80GB.

로컬 E4(nf4, transformers)가 val 672에서 96.88%를 냈습니다. 제출 추론 6,714문항을
Colab A100에서 빠르게 돌리려고 vLLM을 씁니다. 추론 엔진과 정밀도가 바뀌므로
**val 672를 먼저 돌려 로컬 E4와 정확도·문항별 일치율을 확인한 뒤** test로 넘어갑니다.

val ID와 로컬 E4 예측은 생성 시점에 노트북 안에 넣습니다. Colab이 비공개 저장소를 clone할 수
없고 output/은 Git 제외라, Drive에 추가 파일을 올리지 않아도 되게 하려는 것입니다.
"""

from _project import ROOT
import ast
import json
import sys

sys.path.insert(0, str(ROOT / "src"))
import common_eval as ce  # noqa: E402

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": text.strip().splitlines(keepends=True)})


ce.DATA_DIR = ROOT / "data"
val_df = ce.load_val(ROOT / ce.BASELINE_RUN)
VAL_IDS = val_df.id.tolist()
VAL_SHA = ce.ids_sha256(VAL_IDS)
e4 = {}
with (ROOT / "output/exp_E4_qwen35_27b/predictions.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            e4[item["id"]] = item["pred"]
assert set(e4) == set(VAL_IDS), "로컬 E4 예측이 val 672와 맞지 않습니다"
E4_LOCAL = "".join(e4[i] if e4[i] in ce.CHOICES else "x" for i in VAL_IDS)
E4_LOCAL_ACC = sum(e4[i] == a for i, a in zip(VAL_IDS, val_df.answer)) / len(VAL_IDS)
VAL_NUMS = [int(i.removeprefix("train_").removesuffix(".jpg")) for i in VAL_IDS]
assert [f"train_{n:04d}.jpg" for n in VAL_NUMS] == VAL_IDS

md(f'''
# E4 Colab — Qwen3.5-27B bf16 + vLLM (A100 80GB)

로컬 E4(nf4 4bit, transformers)가 val 672에서 **{E4_LOCAL_ACC:.2%}**를 냈습니다.
이 노트북은 같은 모델을 **bf16 그대로 vLLM**으로 올려 test 6,714문항을 빠르게 풉니다.

| | 로컬 E4 | 이 노트북 |
| --- | --- | --- |
| 정밀도 | nf4 4bit | **bf16** |
| 추론 엔진 | transformers, 1문항씩 | **vLLM, 묶어서** |
| 프롬프트·이미지 규칙·파서 | `common_eval` | 같은 `common_eval` (Drive 사본) |

엔진과 정밀도가 바뀌므로 **먼저 val 672를 돌려(약 5분) 로컬 E4와 맞춰 봅니다.**
정확도가 크게 벗어나거나 일치율이 낮으면 test로 넘어가지 말고 원인을 보세요.

## 준비물 (Drive)

- `MyDrive/SSAFY/AI_Challenge2/ssafy-16-2-ai.zip` — E1 Colab과 같은 데이터
- `MyDrive/SSAFY/AI_Challenge2/src/common_eval.py` — 로컬 `src/common_eval.py` 사본

val 672 ID와 로컬 E4 예측은 이 노트북 안에 들어 있어 따로 올릴 필요가 없습니다.

## 시간 (A100 80GB 예상)

설치 3~5분 → 모델 다운로드·적재 5~10분 → val 약 5분 → test **40~50분**.
27B는 문항당 약 1,000토큰을 prefill하므로 A100에서도 계산량이 병목입니다.
256문항마다 Drive에 저장하므로 세션이 끊겨도 다시 실행하면 이어집니다.
''')

md('''
## 1. 설치

**새 런타임에서 가장 먼저 실행하세요.** vLLM이 torch를 자기 버전으로 바꿉니다.
설치가 끝나면 **런타임 → 세션 다시 시작**을 한 번 누르고 2번 셀부터 실행하세요(설치는 유지됩니다).
바뀐 torch를 새 프로세스에서 불러와야 하고, torchaudio 충돌도 이때 정리됩니다.
''')

code('''
!pip -q install -U vllm
!pip -q install -U "huggingface_hub[hf_xet]" pillow pandas
# vLLM이 torch를 새 CUDA 빌드로 바꾸면 Colab에 깔려 있던 torchaudio와 CUDA 버전이 어긋나고,
# transformers가 torchaudio를 자동으로 불러오다 멈춥니다. 이 작업에는 필요 없으므로 지웁니다.
!pip -q uninstall -y torchaudio
''')

code('''
import gc, json, math, os, re, shutil, sys, time
from pathlib import Path

import pandas as pd
import torch

import vllm
import transformers
import torchvision

print("vllm", vllm.__version__, "/ transformers", transformers.__version__)
print("torch", torch.__version__, f"(CUDA {torch.version.cuda})", "/ torchvision", torchvision.__version__)
try:
    import torchaudio  # noqa: F401
    raise RuntimeError("torchaudio가 남아 있습니다. `!pip uninstall -y torchaudio` 후 세션을 다시 시작하세요")
except ImportError:
    pass
assert torch.cuda.is_available(), "GPU 런타임을 선택하세요 (런타임 - 런타임 유형 변경)."
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {GPU_NAME} / {GPU_GIB:.1f} GiB")
assert GPU_GIB >= 70, "27B bf16(약 52 GiB)은 A100 80GB가 필요합니다. 40GB면 로컬 4bit 제출을 쓰세요."
''')

md('''
## 2. Drive · 데이터 · 공통 코드
''')

code('''
from google.colab import drive
drive.mount("/content/drive")

DRIVE_BASE = Path("/content/drive/MyDrive/SSAFY/AI_Challenge2")
ZIP_PATH = DRIVE_BASE / "ssafy-16-2-ai.zip"
ZIP_PASSWORD = "ocrisnotenough"
assert ZIP_PATH.is_file(), f"zip을 찾지 못했습니다: {ZIP_PATH}"


def find_data_root():
    for parent, dirs, files in os.walk("/content"):
        dirs[:] = [d for d in dirs if d not in {"drive", ".cache", "sample_data", "models"}]
        if "test.csv" in files and "train.csv" in files:
            return Path(parent)
    return None


if find_data_root() is None:
    import subprocess
    subprocess.run(["7z", "x", "-y", f"-p{ZIP_PASSWORD}", "-o/content/", str(ZIP_PATH)],
                   check=True, stdout=subprocess.DEVNULL)
DATA_ROOT = find_data_root()
assert DATA_ROOT is not None, "압축 해제 후에도 train.csv/test.csv를 찾지 못했습니다."

SRC_DIR = Path("/content/src")
SRC_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_SRC = DRIVE_BASE / "src" / "common_eval.py"
assert DRIVE_SRC.is_file(), f"로컬 src/common_eval.py를 {DRIVE_SRC} 로 복사하세요"
shutil.copy2(DRIVE_SRC, SRC_DIR / "common_eval.py")
sys.path.insert(0, str(SRC_DIR))
import common_eval as ce

ce.DATA_DIR = DATA_ROOT
for name in ("SYSTEM", "make_prompt", "parse_choice", "open_image", "IMAGE_MAX_EDGE", "ids_sha256"):
    assert hasattr(ce, name), f"Drive의 common_eval.py가 오래됐습니다: {name} 없음"
print("데이터:", DATA_ROOT, "/ 이미지 긴 변 제한:", ce.IMAGE_MAX_EDGE, "px")
''')

md('''
## 3. 모델 적재 (vLLM)

`MODEL_ID`를 넘기면 vLLM이 Hugging Face에서 바로 받습니다(52 GiB, 보통 수 분).
이미지는 문항당 1장이고 프롬프트는 1,000토큰 안팎이라 `max_model_len`은 4,096이면 충분합니다.
''')

code('''
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3.5-27B"
MAX_NEW_TOKENS = 24          # 로컬 E1~E4와 동일
LOGPROBS_K = 20              # 첫 토큰 상위 k개 로그확률. a~d 확률을 뽑는 데 씁니다

t0 = time.time()
llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90,
          limit_mm_per_prompt={"image": 1}, seed=42, download_dir="/content/models")
proc = AutoProcessor.from_pretrained(MODEL_ID)
print(f"적재 {time.time() - t0:.0f}초")

SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS_K, seed=42)


def letter_token_ids():
    tok = proc.tokenizer
    table = {}
    for ch in ce.CHOICES:
        cand = {tok.encode(f, add_special_tokens=False)[0]
                for f in (ch, " " + ch, ch.upper(), " " + ch.upper())
                if tok.encode(f, add_special_tokens=False)}
        table[ch] = sorted(cand)
    return table


LETTERS = letter_token_ids()


def make_request(row):
    """로컬과 같은 메시지를 HF 채팅 템플릿으로 렌더링하고, 이미지는 같은 open_image로 줄여 넘깁니다."""
    image = ce.open_image(row)
    messages = [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": ce.make_prompt(row)}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, image.size

print("a~d 토큰 id:", LETTERS)
''')

md('''
## 4. 추론 함수

묶음 단위로 vLLM에 넘기고, 묶음이 끝날 때마다 `predictions.jsonl`에 쓰고 Drive에 복사합니다.
다시 실행하면 기록된 문항을 건너뜁니다. 로컬과 같이 `parse_choice`로 답을 읽고,
첫 토큰의 a~d 로그확률도 함께 남깁니다(상위 20개에 없으면 `-1e9`).
''')

code('''
CHUNK = 256


def letter_logprobs(first_step):
    out = {}
    for ch, ids in LETTERS.items():
        vals = [first_step[i].logprob for i in ids if i in first_step]
        out[ch] = max(vals) if vals else -1e9   # 상위 k개 밖. JSON에 -inf를 쓰지 않으려고 유한값
    return out


def run(rows, out_dir, drive_dir, desc):
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if not pred_path.is_file() and (drive_dir / "predictions.jsonl").is_file():
        shutil.copy2(drive_dir / "predictions.jsonl", pred_path)
        print("Drive에서 이전 기록을 복구했습니다")
    done = set()
    if pred_path.is_file():
        done = {json.loads(l)["id"] for l in pred_path.open(encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{desc}] 대상 {len(rows)} / 기록됨 {len(done)} / 남은 {len(todo)}")

    t_all = time.time()
    for start in range(0, len(todo), CHUNK):
        batch = todo[start:start + CHUNK]
        t0 = time.time()
        reqs, sizes = zip(*(make_request(r) for r in batch))
        if start == 0:
            assert "<|image_pad|>" in reqs[0]["prompt"], "프롬프트에 이미지 토큰이 없습니다"
        outs = llm.generate(list(reqs), SAMPLING, use_tqdm=False)
        elapsed = time.time() - t0
        with pred_path.open("a", encoding="utf-8") as log:
            for row, size, o in zip(batch, sizes, outs):
                c = o.outputs[0]
                raw = c.text.strip()
                log.write(json.dumps(dict(
                    id=row["id"], raw_output=raw, pred=ce.parse_choice(raw),
                    letter_logprobs=letter_logprobs(c.logprobs[0]) if c.logprobs else {},
                    answer_truncated=c.finish_reason == "length", answer_tokens=len(c.token_ids),
                    seconds=elapsed / len(batch), image_width=size[0], image_height=size[1]),
                    ensure_ascii=False) + "\\n")
        for f in out_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, drive_dir / f.name)
        n_done = start + len(batch)
        rate = (time.time() - t_all) / n_done
        print(f"  {n_done}/{len(todo)}  {elapsed / len(batch):.3f}초/문항  "
              f"남은 약 {rate * (len(todo) - n_done) / 60:.0f}분")
    res = pd.read_json(pred_path, lines=True).drop_duplicates("id", keep="last")
    return res
''')

md(f'''
## 5. val 672 — 로컬 E4와 맞춰 보기

로컬 E4와 **같은 672문항**입니다(ID SHA-256 `{VAL_SHA[:16]}…` 대조).
로컬 E4의 문항별 예측도 노트북에 들어 있어 일치율을 바로 봅니다.

판단 기준: 정확도가 로컬({E4_LOCAL_ACC:.2%})에서 ±1%p 안이고 일치율이 95% 이상이면 test로 넘어갑니다.
로컬 E4와 E1(9B)의 불일치가 4.5%였으므로, 같은 모델의 정밀도 차이는 그보다 작아야 정상입니다.
''')

code(f'''
VAL_NUMS = {VAL_NUMS}
VAL_IDS = [f"train_{{n:04d}}.jpg" for n in VAL_NUMS]
assert ce.ids_sha256(VAL_IDS) == "{VAL_SHA}", "val ID 해시가 다릅니다"
E4_LOCAL = "{E4_LOCAL}"   # 로컬 E4(nf4) 예측, VAL_IDS 순서
E4_LOCAL_ACC = {E4_LOCAL_ACC!r}

train_df = pd.read_csv(DATA_ROOT / "train.csv", keep_default_na=False).set_index("id")
val_df = train_df.loc[VAL_IDS].reset_index()
val_rows = val_df.drop(columns=["answer"]).to_dict("records")

VAL_OUT = Path("/content/out/exp_E4_vllm_val")
VAL_DRIVE = DRIVE_BASE / "output" / "exp_E4_vllm_val"
val_res = run(val_rows, VAL_OUT, VAL_DRIVE, "val 672")

v = val_df[["id", "answer"]].merge(val_res[["id", "pred"]], on="id", validate="one_to_one")
v["local"] = list(E4_LOCAL)
acc = (v.pred == v.answer).mean()
agree = (v.pred == v.local).mean()
b = int(((v.local == v.answer) & (v.pred != v.answer)).sum())
c = int(((v.local != v.answer) & (v.pred == v.answer)).sum())
print(f"\\nvLLM bf16 {{acc:.4f}} ({{int((v.pred == v.answer).sum())}}/672) / 로컬 nf4 {{E4_LOCAL_ACC:.4f}}")
print(f"문항별 일치율 {{agree:.1%}} / 로컬만 정답 {{b}} / vLLM만 정답 {{c}} / INVALID {{int((v.pred == 'INVALID').sum())}}")
OK_TO_SUBMIT = abs(acc - E4_LOCAL_ACC) <= 0.01 and agree >= 0.95
print("-> test로 진행하세요" if OK_TO_SUBMIT else
      "-> 로컬과 차이가 큽니다. test로 넘어가기 전에 원인을 확인하세요 (INVALID·프롬프트·이미지 토큰)")
''')

md('''
## 6. test 6,714 — 제출 파일

위 판단이 `test로 진행하세요`일 때만 실행하세요. 파서가 실패한 문항은 기록된 a~d 로그확률 중
가장 높은 글자로 채웁니다(특정 글자로 메우면 분포가 쏠립니다).
''')

code('''
assert OK_TO_SUBMIT, "val 확인을 통과하지 못했습니다. 그래도 진행하려면 OK_TO_SUBMIT = True로 두세요"

test_df = pd.read_csv(DATA_ROOT / "test.csv", keep_default_na=False)
sample_df = pd.read_csv(DATA_ROOT / "sample_submission.csv", keep_default_na=False)
assert list(test_df.id) == list(sample_df.id), "sample_submission과 id 순서가 다릅니다"
assert test_df.id.is_unique

TEST_OUT = Path("/content/out/exp_E4_vllm_test")
TEST_DRIVE = DRIVE_BASE / "output" / "exp_E4_vllm_test"
test_res = run(test_df.to_dict("records"), TEST_OUT, TEST_DRIVE, "test 6,714")
''')

code('''
res = test_res.copy()
fallback = res.pred == "INVALID"
res.loc[fallback, "pred"] = res.loc[fallback, "letter_logprobs"].map(
    lambda lp: max(lp, key=lp.get) if lp else "INVALID")
sub = test_df[["id"]].merge(res[["id", "pred"]], on="id", how="left", validate="one_to_one")
assert sub.pred.isin(ce.CHOICES).all(), f"a~d 이외의 값: {sorted(set(sub.pred) - set(ce.CHOICES))}"
assert list(sub.id) == list(sample_df.id)

share = sub.pred.value_counts(normalize=True).reindex(ce.CHOICES).fillna(0)
print(f"폴백 {int(fallback.sum())}건 / 예측 분포"); print(share.round(3).to_string())
if share.max() > 0.35:
    print(f"경고: '{share.idxmax()}'가 {share.max():.1%}로 쏠렸습니다")

stem = "submission_qwen35-27b_zeroshot-bf16-vllm"
used = [int(m.group(1)) for p in TEST_DRIVE.glob(stem + "_v*.csv")
        if (m := re.fullmatch(re.escape(stem) + r"_v(\\d+)", p.stem))]
name = f"{stem}_v{max(used, default=0) + 1}.csv"
sub.rename(columns={"pred": "answer"}).to_csv(TEST_OUT / name, index=False, encoding="utf-8")
shutil.copy2(TEST_OUT / name, TEST_DRIVE / name)
print("제출 파일:", TEST_DRIVE / name)
''')

md('''
## 끝나면

`MyDrive/SSAFY/AI_Challenge2/output/exp_E4_vllm_val`, `exp_E4_vllm_test`를 로컬 `output/`으로 받아 두세요.
val 결과는 bf16과 nf4의 차이를, test 예측은 이후 로컬 4bit 제출이나 E1 제출과의 비교에 씁니다.
''')

# !로 시작하는 셸 줄은 Colab 전용이라 문법 검사에서 뺍니다.
for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "colab": {"gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/experiments/AI_Challenge2_expE4_qwen35_27b_vllm_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
