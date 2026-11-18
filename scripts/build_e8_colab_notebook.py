"""Create the E8 Colab notebook: 추론 단계 기법(Order4 · Gate · Zoom · Blend)을 E7 LoRA 모델에 적용합니다.

사용: python scripts/build_e8_colab_notebook.py [A]   (기본 A. Drive에 exp_E7_27b_<구성>/adapter가 있어야 함)

팀 0.97110(Qwen3.6-27B)은 LoRA 뒤에 Order4·Gate·Zoom·Blend를 더했고, 팀 검증 1,343에서 LoRA 단독 대비
11:1(p=0.006)이었습니다. 다만 Gate 기준과 Blend 비율을 그 1,343으로 정했으므로 독립 확인이 아닙니다.
E8은 **재학습 없이** 3.5 LoRA(E7)에 같은 규칙을 기준값 그대로 적용하고, 한 번도 설정 선택에 쓰지 않은
holdout 1,300에서 단계별로 짝 비교합니다.

src/common_eval.py·vlm_runtime.py·qlora.py·vllm_eval.py·tta.py 원문과 holdout ID를 노트북에 넣습니다.
"""

from _project import ROOT
import ast
import hashlib
import json
import sys

import pandas as pd

import tta

config = sys.argv[1] if len(sys.argv) == 2 else "A"
if config not in {"A", "B", "C"}:
    raise SystemExit("사용: python scripts/build_e8_colab_notebook.py [A|B|C]")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": text.strip().splitlines(keepends=True)})


SRC_FILES = ["common_eval.py", "vlm_runtime.py", "qlora.py", "vllm_eval.py", "tta.py"]
SOURCES = {name: (ROOT / "src" / name).read_text(encoding="utf-8") for name in SRC_FILES}
SOURCE_SHA = {name: hashlib.sha256(s.encode("utf-8")).hexdigest() for name, s in SOURCES.items()}

num = lambda i: int(i.split("_")[1].removesuffix(".jpg"))
# E8-A 오답 분석(output/exp_E8_tta_A/holdout_errors_36.csv)에서 학습으로 고칠 수 있다고 본 유형
TRAINABLE = ("사람 해석", "왜곡된 전사 선지")
errors = pd.read_csv(ROOT / "output/exp_E8_tta_A/holdout_errors_36.csv", encoding="utf-8-sig")
E8A_ERROR_NUMS = [num(i) for i in errors.id]
TARGET_NUMS = [num(i) for i in errors[errors["분류"].isin(TRAINABLE)].id]
split = pd.read_csv(ROOT / "splits/e6_split.csv")
HOLDOUT_NUMS = [num(i) for i in split[split.split == "holdout"].id]
VAL672_NUMS = [num(i) for i in split[split.in_val672].id]

md(f'''
# E8-{config} Colab — E7-{config}(Qwen3.5-27B LoRA) + Order4 · Gate · Zoom · Blend (A100 80GB)

**재학습 없이** E7-{config} 어댑터에 팀 0.97110 파이프라인의 추론 단계 기법을 더합니다.

| 단계 | 내용 | 비교 |
| --- | --- | --- |
| E8-1 **Order4** | 선지 순서 `abcd·bcda·cdab·dabc`로 4번 풀고, 확률을 원래 선지로 되돌려 평균 | E7-{config} 단독 |
| E8-2 **Gate → Zoom → Blend** | 최댓값 < {tta.GATE_MAX_PROB} · 1·2위 차이 < {tta.GATE_MARGIN} · 순서 불일치 중 하나면 고해상도로 다시 Order4, 기존과 {tta.BLEND_WEIGHT:.0%}씩 섞음 | E8-1 |
| E8-3 **제출** | 같은 파이프라인으로 test 6,714 | LB 0.97140(E8-A) · 0.97110(팀) |

**기준값을 고치지 마세요.** Gate·Blend 값은 팀이 자기 검증 1,343으로 정한 값입니다.
holdout 결과를 보고 바꾸면 holdout이 더는 독립 확인이 아니게 됩니다.

| 단계 | 시간 (추정) |
| --- | ---: |
| 설치·데이터·원본 다운로드 | 15~20분 |
| 병합 (Drive 어댑터 → bf16 원본) | 약 5분 |
| holdout Order4 (1,300 × 4) | 약 15분 |
| holdout Zoom (약 100 × 4) | 약 3분 |
| test Order4 + Zoom (6,714 × 4 + 약 500 × 4) | 약 1시간 40분 |

추론은 순서마다 256문항 묶음으로 Drive에 저장되므로, 끊겨도 다시 실행하면 이어 갑니다.

## 준비물

- A100 80GB 런타임
- Drive `MyDrive/SSAFY/AI_Challenge2/ssafy-16-2-ai.zip`
- Drive `MyDrive/SSAFY/AI_Challenge2/output/exp_E7_27b_{config}/adapter/` (E7-{config} 학습 결과)
- 비교용: `output/exp_E7_27b_{config}/holdout/`, `output/exp_E6_zeroshot_train/` (없으면 해당 비교만 건너뜀)
''')

md('''
## 1. 설치

새 런타임에서 가장 먼저 실행하고, 끝나면 **런타임 → 세션 다시 시작** 후 2번 셀부터 실행하세요.
''')

code('''
!pip -q install -U vllm
!pip -q install -U peft "huggingface_hub[hf_xet]" pillow pandas
# E7과 같은 충돌 회피: torchaudio(CUDA 불일치로 transformers import 실패), torchao 0.10(peft와 충돌)
!pip -q uninstall -y torchaudio torchao
''')

code('''
import gc, json, os, sys, time
from pathlib import Path

import pandas as pd
import torch
import transformers
import peft
import vllm

print("vllm", vllm.__version__, "/ transformers", transformers.__version__, "/ peft", peft.__version__)
for mod in ("torchaudio", "torchao"):
    try:
        __import__(mod)
        raise RuntimeError(f"{mod}가 남아 있습니다. `!pip uninstall -y {mod}` 후 세션을 다시 시작하세요")
    except ImportError:
        pass
assert torch.cuda.is_available(), "GPU 런타임을 선택하세요"
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {torch.cuda.get_device_name(0)} / {GPU_GIB:.1f} GiB")
assert GPU_GIB >= 70, "27B bf16 병합·추론은 A100 80GB가 필요합니다"
''')

md('''
## 2. Drive · 데이터 · 공통 코드
''')

code(f'''
from google.colab import drive
print("[1/3] Drive 연결 중… (권한 창이 뜨면 허용)", flush=True)
drive.mount("/content/drive")

DRIVE_BASE = Path("/content/drive/MyDrive/SSAFY/AI_Challenge2")
ZIP_PATH = DRIVE_BASE / "ssafy-16-2-ai.zip"
ZIP_PASSWORD = "ocrisnotenough"
assert ZIP_PATH.is_file(), f"zip을 찾지 못했습니다: {{ZIP_PATH}}"


def find_data_root():
    for parent, dirs, files in os.walk("/content"):
        dirs[:] = [d for d in dirs if d not in {{"drive", ".cache", "sample_data", "models", "merged"}}]
        if "test.csv" in files and "train.csv" in files:
            return Path(parent)
    return None


if find_data_root() is None:
    import subprocess
    print("[2/3] 데이터 zip(1.8 GB) 압축 해제 중… 출력 없이 1~5분 걸립니다. "
          "왼쪽 파일 탭 /content에 train/ 폴더가 생기면 진행 중입니다.", flush=True)
    t0 = time.time()
    subprocess.run(["7z", "x", "-y", f"-p{{ZIP_PASSWORD}}", "-o/content/", str(ZIP_PATH)],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"      압축 해제 {{time.time() - t0:.0f}}초", flush=True)
else:
    print("[2/3] 데이터가 이미 풀려 있어 건너뜁니다", flush=True)
DATA_ROOT = find_data_root()
assert DATA_ROOT is not None, "train/test.csv를 찾지 못했습니다"
print("[3/3] 공통 코드 준비·자체 테스트", flush=True)

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
import tta

ce.DATA_DIR = DATA_ROOT
ce._self_test(); ve._self_test(); tta._self_test()
print("데이터:", DATA_ROOT)
''')

md('''
## 3. 설정
''')

code(f'''
CONFIG = "{config}"
SOURCE_RUN = f"exp_E7_27b_{{CONFIG}}"
RUN_NAME = f"exp_E8_tta_{{CONFIG}}"
DRIVE_RUN = DRIVE_BASE / "output" / RUN_NAME
ADAPTER_DIR = DRIVE_BASE / "output" / SOURCE_RUN / "adapter"
BASE_DIR = Path("/content/models/Qwen3.5-27B")
MERGED_DIR = Path("/content/merged") / SOURCE_RUN
LOCAL_OUT = Path("/content/out") / RUN_NAME
DRIVE_RUN.mkdir(parents=True, exist_ok=True)
assert (ADAPTER_DIR / "adapter_config.json").is_file(), f"E7 어댑터가 없습니다: {{ADAPTER_DIR}}"
print("실행 이름:", RUN_NAME, "/ 어댑터:", ADAPTER_DIR)
print(f"Gate: 최댓값 < {{tta.GATE_MAX_PROB}} 또는 차이 < {{tta.GATE_MARGIN}} 또는 순서 불일치 / Blend {{tta.BLEND_WEIGHT}}")
print(f"Zoom 넓이: {{tta.ZOOM_MIN_PIXELS:,}} ~ {{tta.ZOOM_MAX_PIXELS:,}} 픽셀 (기본 입력은 긴 변 {{ce.IMAGE_MAX_EDGE}})")
''')

md('''
## 4. holdout 1,300

E6에서 나눈 holdout입니다. E7 학습에 쓰지 않았고, 설정 선택에도 쓰지 않았습니다.
''')

code(f'''
HOLDOUT_NUMS = {HOLDOUT_NUMS}
VAL672_NUMS = {VAL672_NUMS}
HOLDOUT = {{f"train_{{n:04d}}.jpg" for n in HOLDOUT_NUMS}}
VAL672 = {{f"train_{{n:04d}}.jpg" for n in VAL672_NUMS}}
assert len(HOLDOUT) == 1300 and VAL672 <= HOLDOUT
train_df = pd.read_csv(DATA_ROOT / "train.csv", keep_default_na=False)
gold = train_df[train_df.id.isin(HOLDOUT)].reset_index(drop=True)
holdout_rows = gold.drop(columns=["answer"]).to_dict("records")
print("holdout", len(gold), "/ 정답 분포", gold.answer.value_counts().sort_index().to_dict())
''')

md('''
## 5. 원본 다운로드 · 병합

E7 어댑터를 bf16 원본에 합쳐 로컬 디스크에 저장합니다(약 54 GiB). 병합본은 Drive에 올리지 않습니다.
''')

code('''
from huggingface_hub import snapshot_download

if not (BASE_DIR / "config.json").is_file():
    snapshot_download(repo_id="Qwen/Qwen3.5-27B", local_dir=str(BASE_DIR))
vr.check_weights(BASE_DIR)
if (MERGED_DIR / "config.json").is_file():
    print("이미 병합됨:", MERGED_DIR)
else:
    t0 = time.time()
    qlora.merge_and_save(BASE_DIR, ADAPTER_DIR, MERGED_DIR, dtype=torch.bfloat16)
    gc.collect(); torch.cuda.empty_cache()
    print(f"병합 {time.time() - t0:.0f}초 ->", MERGED_DIR)
''')

md('''
## 6. vLLM 적재 · Zoom 입력 확인

**메모리 오류가 나면** *세션 다시 시작* 후 2·3·4번 셀과 이 셀부터 실행하세요. 병합본은 디스크에 남아 있습니다.

답은 생성한 글자가 아니라 첫 토큰의 a~d 로그확률로 정하므로 `max_tokens=1`입니다
(그래서 기록의 `answer_truncated`는 항상 True이며 의미가 없습니다).
Zoom이 실제로 비전 토큰을 늘리는지 holdout 몇 장으로 확인하고, 늘지 않으면 멈춥니다.
''')

code('''
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

llm = LLM(model=str(MERGED_DIR), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90,
          limit_mm_per_prompt={"image": 1}, seed=42)
proc = AutoProcessor.from_pretrained(str(MERGED_DIR))
SAMPLING = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, seed=42)

for row in holdout_rows[:5]:
    native, zoom = ce.open_image(row), tta.zoom_image(row)
    n_tok, z_tok = tta.vision_tokens(proc, native), tta.vision_tokens(proc, zoom)
    print(f"{row['id']}: 기본 {native.size} {n_tok}토큰 -> Zoom {zoom.size} {z_tok}토큰")
    assert z_tok > n_tok, "Zoom이 비전 토큰을 늘리지 않습니다. 프로세서 max_pixels를 확인하세요"
''')

md('''
## 7. E8-1 — holdout Order4

`abcd` 순서는 E7 단독과 같은 입력입니다. 먼저 E7 기록과 답이 같은지(재현) 확인한 뒤 Order4와 짝 비교합니다.
''')

code('''
def paired_all(stages, ids=None):
    """이웃 단계끼리, 그리고 첫 단계 대비 마지막 단계를 짝 비교합니다."""
    g = gold if ids is None else gold[gold.id.isin(ids)]
    names = list(stages)
    out = {}
    for a, b in list(zip(names, names[1:])) + ([(names[0], names[-1])] if len(names) > 2 else []):
        out[f"{a} -> {b}"] = ve.paired(g, stages[a], stages[b], a, b)
    return out


h_orders = tta.run_orders(ve, llm, proc, SAMPLING, holdout_rows, LOCAL_OUT / "holdout",
                          DRIVE_RUN / "holdout", desc="holdout")
h_o4 = tta.order4(h_orders)
h_base = h_orders[tta.IDENTITY].assign(pred=lambda d: d.letter_logprobs.map(
    lambda lp: tta.summarize(tta.slot_probs(lp))["pred"]))

prev_path = DRIVE_BASE / "output" / SOURCE_RUN / "holdout" / "predictions.jsonl"
if prev_path.is_file():
    prev = ve.read(prev_path)
    same = h_base[["id", "pred"]].merge(prev[["id", "pred"]], on="id", suffixes=("", "_e7"))
    print(f"재현: E7-{CONFIG} 기록과 abcd 답 일치 {(same.pred == same.pred_e7).mean():.4f} ({len(same)}문항)")
print(f"순서 불일치 문항 {int(h_o4.disagree.sum())} ({h_o4.disagree.mean():.1%})")
h_o4.to_csv(DRIVE_RUN / "holdout_order4.csv", index=False)
r1 = paired_all({f"E7-{CONFIG}": h_base, "Order4": h_o4})
''')

md('''
## 8. E8-2 — Gate → Zoom → Blend

Gate에 걸린 문항만 고해상도로 다시 Order4하고, 기존 Order4와 반씩 섞습니다. 나머지 문항은 Order4를 유지합니다.
''')

code('''
def zoom_blend(o4, rows, split):
    """Gate 선별 → Zoom Order4 → Blend. (최종 표, Zoom 표)를 돌려줍니다."""
    picked = set(o4[tta.gate(o4)].id)
    zoom_rows = [r for r in rows if r["id"] in picked]
    print(f"[{split}] Gate 대상 {len(zoom_rows)} / {len(rows)} ({len(zoom_rows) / len(rows):.1%})")
    z_orders = tta.run_orders(ve, llm, proc, SAMPLING, zoom_rows, LOCAL_OUT / f"{split}_zoom",
                              DRIVE_RUN / f"{split}_zoom", image_fn=tta.zoom_image, desc=f"{split} zoom")
    z = z_orders[tta.IDENTITY]
    assert (z.image_width * z.image_height >= tta.ZOOM_MIN_PIXELS * 0.99).all(), "Zoom 입력이 커지지 않았습니다"
    zoom = tta.order4(z_orders)
    return tta.blend(o4, zoom), zoom


h_final, h_zoom = zoom_blend(h_o4, holdout_rows, "holdout")
h_final.to_csv(DRIVE_RUN / "holdout_final.csv", index=False)
stages = {f"E7-{CONFIG}": h_base, "Order4": h_o4, "Order4+Zoom": h_final}
r2 = paired_all(stages)
gated = set(h_final[h_final.reviewed].id)
print(f"\\nGate 대상 {len(gated)}문항 안에서:")
g_in = gold[gold.id.isin(gated)]
ve.paired(g_in, h_o4, h_final, "Order4", "Order4+Zoom")

e6_path = DRIVE_BASE / "output" / "exp_E6_zeroshot_train" / "predictions.jsonl"
if e6_path.is_file():
    print("\\n27B 제로샷(E6) 대비:")
    ve.paired(gold, ve.read(e6_path), h_final, "E6 제로샷", "E8 최종")
report = dict(config=CONFIG, gate=dict(max_prob=tta.GATE_MAX_PROB, margin=tta.GATE_MARGIN),
              blend=tta.BLEND_WEIGHT, zoom_pixels=[tta.ZOOM_MIN_PIXELS, tta.ZOOM_MAX_PIXELS],
              gated=len(gated), holdout={k: v for k, v in {**r1, **r2}.items()})
(DRIVE_RUN / "holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("\\n저장:", DRIVE_RUN / "holdout_report.json")
''')

if config != "A":
    md(f'''
## 8-1. E8-A와 비교 — 제출 여부 판정

미리 정한 기준(`docs/decisions.md`)으로 판정합니다. 결과를 본 뒤 기준을 바꾸지 마세요.

- **확인 지점 ①** E7-{config}(기본 순서 1회)가 E7-A보다 뚜렷하게 나쁘면(E7-A만 정답이 더 많고 p < 0.1) 멈춤
- **확인 지점 ②** E8-{config} 최종 정확도가 E8-A(97.23%)보다 낮지 않으면 9번 제출, 낮으면 E8-A(0.97140) 유지

E8-A 오답 분석에서 학습으로 고칠 수 있다고 본 {len(TARGET_NUMS)}문항({" · ".join(TRAINABLE)})과
E8-A 오답 {len(E8A_ERROR_NUMS)}문항 전체가 얼마나 고쳐졌는지도 따로 셉니다.
''')

    code(f'''
TARGET = {{f"train_{{n:04d}}.jpg" for n in {TARGET_NUMS}}}
E8A_ERRORS = {{f"train_{{n:04d}}.jpg" for n in {E8A_ERROR_NUMS}}}
E7A_PATH = DRIVE_BASE / "output" / "exp_E7_27b_A" / "holdout" / "predictions.jsonl"
E8A_PATH = DRIVE_BASE / "output" / "exp_E8_tta_A" / "holdout_final.csv"
CHECK1_P = 0.1


def correct_in(res, ids):
    ans = gold.set_index("id").answer.reindex(sorted(ids))
    return int((res.set_index("id").pred.reindex(ans.index) == ans).sum())


if E7A_PATH.is_file():
    c1 = ve.paired(gold, ve.read(E7A_PATH), h_base, "E7-A", f"E7-{{CONFIG}}")
    worse = c1["a_only"] > c1["b_only"] and c1["p"] < CHECK1_P
    print(f"확인 지점 ①: {{'멈춤 (E7-' + CONFIG + '이 뚜렷하게 나쁨)' if worse else '통과'}}\\n")
if E8A_PATH.is_file():
    e8a = pd.read_csv(E8A_PATH)
    c2 = ve.paired(gold, e8a, h_final, "E8-A", f"E8-{{CONFIG}}")
    for name, ids in [(f"학습 대상 {{len(TARGET)}}", TARGET), (f"E8-A 오답 {{len(E8A_ERRORS)}}", E8A_ERRORS)]:
        print(f"  {{name}}문항: E8-A 정답 {{correct_in(e8a, ids)}} -> E8-{{CONFIG}} 정답 {{correct_in(h_final, ids)}}")
    submit_ok = c2["b_only"] >= c2["a_only"]
    print(f"\\n확인 지점 ②: {{'통과 -> 9번 제출' if submit_ok else '미통과 -> 제출하지 않고 E8-A 유지'}}")
    report["vs_E8A"] = c2
    report["target_fixed"] = {{"E8A": correct_in(e8a, TARGET), f"E8{{CONFIG}}": correct_in(h_final, TARGET)}}
    (DRIVE_RUN / "holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print("E8-A 결과가 없어 비교를 건너뜁니다:", E8A_PATH)
''')

md('''
## 9. E8-3 — 제출

`SUBMIT = True`로 두고 실행하면 test 6,714문항에 같은 파이프라인을 적용합니다(약 1시간 40분).
Order4만 적용한 파일도 함께 만들어, LB에서 Zoom의 몫을 따로 볼 수 있게 합니다.
''')

code('''
SUBMIT = False

if not SUBMIT:
    print("SUBMIT = False: 제출을 건너뜁니다")
else:
    test_df = pd.read_csv(DATA_ROOT / "test.csv", keep_default_na=False)
    sample_df = pd.read_csv(DATA_ROOT / "sample_submission.csv", keep_default_na=False)
    assert list(test_df.id) == list(sample_df.id)
    test_rows = test_df.to_dict("records")
    t_o4 = tta.order4(tta.run_orders(ve, llm, proc, SAMPLING, test_rows, LOCAL_OUT / "test",
                                     DRIVE_RUN / "test", desc="test"))
    t_final, _ = zoom_blend(t_o4, test_rows, "test")
    for name, res in [("order4", t_o4), ("order4-zoom-blend", t_final)]:
        sub = sample_df[["id"]].merge(res[["id", "pred"]], on="id", how="left", validate="one_to_one")
        assert sub.pred.isin(ce.CHOICES).all() and list(sub.id) == list(sample_df.id)
        path = DRIVE_RUN / f"submission_qwen35-27b_lora-{CONFIG}_{name}.csv"
        sub.rename(columns={"pred": "answer"}).to_csv(path, index=False, encoding="utf-8")
        print(name, "예측 분포", sub.pred.value_counts(normalize=True).round(3).to_dict(), "->", path)
    print(f"Order4 대비 최종 답 변경 {int((t_o4.set_index('id').pred != t_final.set_index('id').pred).sum())}문항")
''')

md('''
## 끝나면

Drive `output/exp_E8_tta_<구성>/`을 로컬 `output/`으로 받아 주세요. 순서별 예측(`holdout/<순서>/`, `holdout_zoom/<순서>/`),
`holdout_order4.csv`, `holdout_final.csv`, `holdout_report.json`, 제출 파일이 들어 있습니다.
''')

for cell in cells:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        ast.parse("\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!")))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "accelerator": "GPU", "colab": {"gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / f"notebooks/experiments/AI_Challenge2_expE8_tta_{config}_colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
