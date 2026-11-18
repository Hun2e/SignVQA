"""Create the E6 Colab notebook: 27B zero-shot over all of train and dev, plus the consensus prompt.

멘토 설명으로 정답의 성격이 분명해졌습니다. 정답은 사진 속 사실이 아니라 여러 교육생이 일치해서 고른
답이고, 응답이 갈린 문항은 train/test에서 빠졌습니다. dev.csv는 문항당 최대 5명의 응답을 담고 있고
최다 응답이 3표(1,759문항) 또는 2표(528문항)뿐이라, 응답이 갈린 문항 쪽으로 보입니다.

이 노트북은 QLoRA(E7)에 앞서 데이터를 진단합니다.
1. train 6,714 제로샷 — 새 기준선. 모델과 사람 합의가 어긋나는 문항을 찾습니다.
2. dev 2,683 제로샷 — 모델이 사람 다수 응답을 얼마나 따르는지, 투표 비율과 확신도의 관계를 봅니다.
3. train 6,714 + 합의 지시문 — 지시문 하나만 바꾼 효과를 6,714문항 대응 표본으로 검정합니다.

src/common_eval.py 원문과 평가 분할(splits/e6_split.csv)을 생성 시점에 노트북 안에 넣습니다.
Drive에 공통 코드 사본을 올릴 필요가 없고, 로컬과 다른 버전을 쓸 수도 없습니다.
"""

from _project import ROOT
import ast
import hashlib
import json

import pandas as pd

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": text.strip().splitlines(keepends=True)})


COMMON_EVAL_SRC = (ROOT / "src/common_eval.py").read_text(encoding="utf-8")
COMMON_EVAL_SHA = hashlib.sha256(COMMON_EVAL_SRC.encode("utf-8")).hexdigest()
split = pd.read_csv(ROOT / "splits/e6_split.csv")
SPLIT_SHA = hashlib.sha256((ROOT / "splits/e6_split.csv").read_bytes()).hexdigest()
to_num = lambda ids: [int(i.removeprefix("train_").removesuffix(".jpg")) for i in ids]
HOLDOUT_NUMS = to_num(split[split.split == "holdout"].id)
import sys
sys.path.insert(0, str(ROOT / "src"))
import common_eval as ce  # noqa: E402
ce.DATA_DIR = ROOT / "data"
_val = ce.load_val(ROOT / ce.BASELINE_RUN)
VAL672_NUMS = to_num(_val.id)            # 기준 분할 순서 그대로. 해시가 순서에 의존합니다
VAL672_SHA = ce.ids_sha256(_val.id.tolist())
assert set(_val.id) == set(split[split.in_val672].id)
assert len(HOLDOUT_NUMS) == 1300 and len(VAL672_NUMS) == 672

md('''
# E6 Colab — 27B 제로샷으로 train·dev 전체 진단 (A100 80GB, vLLM)

정답은 **여러 교육생이 일치해서 고른 답**입니다. 응답이 갈린 문항은 train/test에서 빠졌습니다.
그래서 모델이 사람과 다르게 고른 문항은 "라벨 오류"가 아니라, 대부분 **사람들이 해석하는 방식**
(일반인 상식, 눈에 띄는 쪽, 적힌 그대로)을 모델이 모르는 경우일 수 있습니다.

이 노트북은 QLoRA에 앞서 그 차이를 데이터로 봅니다.

| 단계 | 대상 | 알고 싶은 것 | 시간 |
| --- | --- | --- | ---: |
| 1 | train 6,714 | 새 기준선, 모델–사람 불일치 문항 | 약 22분 |
| 2 | dev 2,683 | dev가 진짜 사람 응답인지, 모델이 다수 응답을 얼마나 따르는지 | 약 9분 |
| 3 | train 6,714 + **합의 지시문** | "사람들이 가장 많이 고를 답"을 고르라고 하면 나아지는지 | 약 22분 |

모델 적재까지 합쳐 **약 1시간 10분**입니다. 단계마다 256문항씩 Drive에 저장하므로
세션이 끊기면 같은 셀을 다시 실행해 이어 갑니다. 시간이 모자라면 3단계는 다음 세션으로 미뤄도 됩니다.

## 준비물

- A100 80GB 런타임
- Drive의 `MyDrive/SSAFY/AI_Challenge2/ssafy-16-2-ai.zip`

공통 코드(`common_eval.py`)와 평가 분할은 이 노트북 안에 들어 있습니다.

## 평가 분할

E6부터는 학습을 하므로 평가 전용 분할을 둡니다. train 6,714 = **학습 5,414 + holdout 1,300**이고,
holdout에는 기존 val 672가 전부 들어 있습니다. 이 노트북은 학습이 없어 train 전체를 풀지만,
결과는 전체·holdout·val 672로 나눠 보여줍니다.
''')

md('''
## 1. 설치

새 런타임에서 가장 먼저 실행하고, 끝나면 **런타임 → 세션 다시 시작** 후 2번 셀부터 실행하세요.
vLLM이 torch를 새 CUDA 빌드로 바꾸므로 새 프로세스에서 불러와야 합니다.
''')

code('''
!pip -q install -U vllm
!pip -q install -U "huggingface_hub[hf_xet]" pillow pandas
# vLLM이 바꾼 torch와 Colab 기본 torchaudio의 CUDA 버전이 어긋나 transformers import가 실패합니다.
!pip -q uninstall -y torchaudio
''')

code('''
import gc, json, math, os, re, shutil, sys, time
from math import comb
from pathlib import Path

import pandas as pd
import torch
import torchvision
import transformers
import vllm

print("vllm", vllm.__version__, "/ transformers", transformers.__version__)
print("torch", torch.__version__, f"(CUDA {torch.version.cuda})", "/ torchvision", torchvision.__version__)
try:
    import torchaudio  # noqa: F401
    raise RuntimeError("torchaudio가 남아 있습니다. `!pip uninstall -y torchaudio` 후 세션을 다시 시작하세요")
except ImportError:
    pass
assert torch.cuda.is_available(), "GPU 런타임을 선택하세요"
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)} / {GPU_GIB:.1f} GiB")
assert GPU_GIB >= 70, "27B bf16(약 52 GiB)은 A100 80GB가 필요합니다"
''')

md('''
## 2. Drive · 데이터 · 공통 코드

`common_eval.py`는 노트북 생성 시점의 저장소 원문을 그대로 씁니다(SHA-256 대조).
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

COMMON_EVAL_SRC = {COMMON_EVAL_SRC!r}
import hashlib
assert hashlib.sha256(COMMON_EVAL_SRC.encode("utf-8")).hexdigest() == "{COMMON_EVAL_SHA}"
SRC_DIR = Path("/content/src"); SRC_DIR.mkdir(parents=True, exist_ok=True)
(SRC_DIR / "common_eval.py").write_text(COMMON_EVAL_SRC, encoding="utf-8")
sys.path.insert(0, str(SRC_DIR))
import common_eval as ce

ce.DATA_DIR = DATA_ROOT
print("데이터:", DATA_ROOT, "/ 이미지 긴 변 제한:", ce.IMAGE_MAX_EDGE, "px")
print("합의 지시문:", ce.CONSENSUS_INSTRUCTION)
''')

md('''
## 3. 모델 적재 (vLLM)
''')

code('''
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen3.5-27B"
MAX_NEW_TOKENS = 24          # E1~E4와 동일
LOGPROBS_K = 20

t0 = time.time()
llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90,
          limit_mm_per_prompt={"image": 1}, seed=42, download_dir="/content/models")
proc = AutoProcessor.from_pretrained(MODEL_ID)
SAMPLING = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, logprobs=LOGPROBS_K, seed=42)
print(f"적재 {time.time() - t0:.0f}초")


def letter_token_ids():
    tok = proc.tokenizer
    return {ch: sorted({tok.encode(f, add_special_tokens=False)[0]
                        for f in (ch, " " + ch, ch.upper(), " " + ch.upper())
                        if tok.encode(f, add_special_tokens=False)})
            for ch in ce.CHOICES}


LETTERS = letter_token_ids()
print("a~d 토큰 id:", LETTERS)
''')

md('''
## 4. 추론 함수

E4 제출과 같은 메시지·이미지 규칙·파서입니다. `consensus=True`면 마지막 지시문만 합의 지시문으로 바뀝니다.
256문항마다 `predictions.jsonl`에 쓰고 Drive에 복사하며, 다시 실행하면 기록된 문항을 건너뜁니다.
첫 토큰의 a~d 로그확률(상위 20개 밖이면 `-1e9`)도 남겨 확신도와 사람 투표 비율을 비교합니다.
''')

code('''
CHUNK = 256
OUT_ROOT = Path("/content/out")


def make_request(row, consensus):
    image = ce.open_image(row)
    messages = [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": ce.make_prompt(row, consensus=consensus)}]}]
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, image.size


def letter_logprobs(first_step):
    out = {}
    for ch, ids in LETTERS.items():
        vals = [first_step[i].logprob for i in ids if i in first_step]
        out[ch] = max(vals) if vals else -1e9
    return out


def run(rows, name, consensus=False):
    out_dir, drive_dir = OUT_ROOT / name, DRIVE_BASE / "output" / name
    out_dir.mkdir(parents=True, exist_ok=True); drive_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    if not pred_path.is_file() and (drive_dir / "predictions.jsonl").is_file():
        shutil.copy2(drive_dir / "predictions.jsonl", pred_path)
        print("Drive에서 이전 기록을 복구했습니다")
    done = set()
    if pred_path.is_file():
        done = {json.loads(l)["id"] for l in pred_path.open(encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{name}] 대상 {len(rows)} / 기록됨 {len(done)} / 남은 {len(todo)}")
    t_all = time.time()
    for start in range(0, len(todo), CHUNK):
        batch = todo[start:start + CHUNK]
        t0 = time.time()
        reqs, sizes = zip(*(make_request(r, consensus) for r in batch))
        if start == 0:
            assert "<|image_pad|>" in reqs[0]["prompt"], "프롬프트에 이미지 토큰이 없습니다"
            assert reqs[0]["prompt"].rstrip().count(
                ce.CONSENSUS_INSTRUCTION if consensus else ce.ANSWER_INSTRUCTION) == 1
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
        print(f"  {n_done}/{len(todo)}  {elapsed / len(batch):.3f}초/문항  "
              f"남은 약 {(time.time() - t_all) / n_done * (len(todo) - n_done) / 60:.0f}분")
    res = pd.read_json(pred_path, lines=True).drop_duplicates("id", keep="last")
    res["conf"] = res.letter_logprobs.map(lambda lp: math.exp(max(lp.values())) if lp else float("nan"))
    return res


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
''')

md(f'''
## 5. 평가 분할

`splits/e6_split.csv`(SHA-256 `{SPLIT_SHA[:16]}…`)의 holdout 1,300과 그 안의 val 672입니다.
''')

code(f'''
HOLDOUT_NUMS = {HOLDOUT_NUMS}
VAL672_NUMS = {VAL672_NUMS}
HOLDOUT = {{f"train_{{n:04d}}.jpg" for n in HOLDOUT_NUMS}}
VAL672_LIST = [f"train_{{n:04d}}.jpg" for n in VAL672_NUMS]
assert ce.ids_sha256(VAL672_LIST) == "{VAL672_SHA}", "val 672가 기준 분할과 다릅니다"
VAL672 = set(VAL672_LIST)
assert len(HOLDOUT) == 1300 and VAL672 <= HOLDOUT

train_df = pd.read_csv(DATA_ROOT / "train.csv", keep_default_na=False)
train_df["split"] = train_df.id.map(lambda i: "holdout" if i in HOLDOUT else "train")
print(train_df.split.value_counts().to_dict())


def report(res, title):
    t = train_df[["id", "answer", "split"]].merge(res[["id", "pred", "conf"]], on="id",
                                                  validate="one_to_one")
    t["ok"] = t.pred == t.answer
    print(f"=== {{title}} ===")
    for name, part in [("전체 6,714", t), ("학습 5,414", t[t.split == "train"]),
                       ("holdout 1,300", t[t.split == "holdout"]), ("val 672", t[t.id.isin(VAL672)])]:
        print(f"  {{name:14s}} {{part.ok.mean():.4f}} ({{int(part.ok.sum())}}/{{len(part)}})")
    low = t.conf <= 0.8
    print(f"  INVALID {{int((t.pred == 'INVALID').sum())}} / 확신도 0.8 이하 {{int(low.sum())}}문항"
          f"({{low.mean():.1%}}, 정확도 {{t[low].ok.mean():.3f}}) / 그 밖 정확도 {{t[~low].ok.mean():.4f}}")
    wrong = t[~t.ok]
    print(f"  사람과 어긋난 {{len(wrong)}}문항 중 확신도 0.95 초과 {{int((wrong.conf > 0.95).sum())}}"
          " (모델이 확신하는데 사람과 다른 문항 = 학습 데이터 진단 대상)")
    return t
''')

md('''
## 6. [1단계] train 6,714 제로샷 — 새 기준선

val 672 값이 로컬 E4(96.88%)와 같거나 아주 가까워야 정상입니다(E4 제출 때 일치율 98.8%).
''')

code('''
base = run(train_df.drop(columns=["answer", "split"]).to_dict("records"), "exp_E6_zeroshot_train")
base_t = report(base, "27B 제로샷 (E4와 같은 지시문)")
''')

md('''
## 7. [2단계] dev 2,683 제로샷 — 사람 투표와 비교

dev는 문항당 최대 5명의 응답이 있고 최다 응답이 3표 또는 2표뿐입니다. 진짜 사람 응답이라면
모델은 **3표 문항에서 2표 문항보다 다수 응답과 더 자주 일치**해야 하고, 모델 확신도도
투표가 몰릴수록 높아야 합니다. 이 관계가 보이면 dev를 모호한 사례의 학습 자료(다수결 또는 투표 비율)로
쓸 근거가 됩니다.
''')

code('''
dev_df = pd.read_csv(DATA_ROOT / "dev.csv", keep_default_na=False)
VOTES = [f"answer{i}" for i in range(1, 6)]
dev_res = run(dev_df.drop(columns=VOTES).to_dict("records"), "exp_E6_zeroshot_dev")

d = dev_df.merge(dev_res[["id", "pred", "conf", "letter_logprobs"]], on="id", validate="one_to_one")
d["votes"] = d[VOTES].apply(lambda r: [v for v in r if v in ce.CHOICES], axis=1, result_type="reduce")
d["n"] = d.votes.map(len)
d["top"] = d.votes.map(lambda v: max(set(v), key=v.count) if v else "")
d["top_k"] = d.apply(lambda r: r.votes.count(r.top) if r.n else 0, axis=1)
d["tie"] = d.apply(lambda r: sum(r.votes.count(x) == r.top_k for x in set(r.votes)) > 1 if r.n else True, axis=1)
d["model_share"] = d.apply(lambda r: r.votes.count(r.pred) / r.n if r.n else float("nan"), axis=1)
d["match_top"] = (d.pred == d.top) & ~d.tie

print("=== dev: 최다 응답 표 수별 ===")
g = d[d.n == 5].groupby("top_k").agg(문항=("id", "size"), 다수와일치=("match_top", "mean"),
                                     모델답의득표율=("model_share", "mean"), 평균확신도=("conf", "mean"))
print(g.round(3).to_string())
print(f"\\n모델 답이 한 표도 못 받은 문항 {int((d.model_share == 0).sum())} / {len(d)}")
dev_votes_path = OUT_ROOT / "exp_E6_zeroshot_dev" / "dev_with_votes.jsonl"
d.drop(columns=["letter_logprobs"]).to_json(dev_votes_path, orient="records", lines=True, force_ascii=False)
shutil.copy2(dev_votes_path, DRIVE_BASE / "output" / "exp_E6_zeroshot_dev" / dev_votes_path.name)
''')

md('''
## 8. [3단계] train 6,714 + 합의 지시문

지시문 하나만 바꿨으므로 1단계와 문항 단위로 맞대 McNemar로 봅니다. 6,714문항이면 0.5%p 안팎의
차이도 가려집니다. 판단은 **전체**가 아니라 **holdout** 기준으로 하세요. 이후 QLoRA와 같은 잣대로
비교해야 하기 때문입니다(학습이 없으니 전체 수치도 참고로는 유효합니다).
''')

code('''
cons = run(train_df.drop(columns=["answer", "split"]).to_dict("records"), "exp_E6_consensus_train",
           consensus=True)
cons_t = report(cons, "27B 제로샷 + 합의 지시문")

j = base_t[["id", "answer", "split", "ok"]].merge(cons_t[["id", "ok"]], on="id", suffixes=("_base", "_cons"))
for name, part in [("전체 6,714", j), ("holdout 1,300", j[j.split == "holdout"])]:
    b = int((part.ok_base & ~part.ok_cons).sum()); c = int((~part.ok_base & part.ok_cons).sum())
    diff = (c - b) / len(part); se = math.sqrt(b + c) / len(part)
    print(f"[{name}] 기존만 정답 {b} / 합의만 정답 {c} / 차이 {diff * 100:+.2f}pp "
          f"95%CI [{(diff - 1.96 * se) * 100:+.2f}, {(diff + 1.96 * se) * 100:+.2f}] / McNemar p = {mcnemar_exact(b, c):.4g}")
''')

md('''
## 끝나면

Drive의 `output/exp_E6_zeroshot_train`, `exp_E6_zeroshot_dev`, `exp_E6_consensus_train`을
로컬 `output/`으로 받아 주세요. 다음 단계에서 모델–사람 불일치 문항을 유형별로 나누고
QLoRA 학습 데이터 기준(전부 / 확신 높은 불일치 제외 / dev 추가)을 정합니다.
''')

for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "colab": {"gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/experiments/AI_Challenge2_expE6_diagnose_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
