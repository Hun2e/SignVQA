"""Create the E4 notebook: Qwen3.5-27B zero-shot on the fixed 672 validation split.

E1 오답 39건을 분류하면 판독 실패 11건 + 위치 관계 6건이 모델 용량 문제로 보입니다.
E4는 E1에서 모델만 9B → 27B로 바꿉니다. 적재·추론 코드는 src/vlm_runtime.py에서 import합니다.
"""

from _project import ROOT, NOTEBOOK_SETUP
import ast
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": text.strip().splitlines(keepends=True)})


md('''
# 실험 E4 — Qwen3.5-27B 제로샷

**가설:** E1의 남은 오답은 대부분 **판독 용량** 문제이고, 모델을 키우면 줄어듭니다.

E1 오답 39건을 원인별로 나눈 결과입니다 (`output/exp_E1_qwen35_9b/error_taxonomy.csv`).

| 분류 | 건수 | 이 실험에서 기대 |
| --- | ---: | --- |
| 글자·숫자를 잘못 읽음 | 11 | **줄어야 함** |
| 위치 관계 (층·화살표·버튼) | 6 | **줄어야 함** |
| 추론·대조 | 6 | 약간 |
| 적힌 그대로가 정답 (정규화 습관) | 7 | 그대로일 것 — 모델이 클수록 언어 사전지식도 강함 |
| 라벨 오류 · 정답이 둘 이상 | 9 | 고쳐도 운 |

## E1과 달라지는 것 — 모델 하나뿐

| | E1 | E4 |
| --- | --- | --- |
| 모델 | Qwen3.5-9B | **Qwen3.5-27B** |
| 양자화 | nf4 4bit | nf4 4bit (동일) |
| 임베딩·lm_head | GPU | **CPU** (16 GiB에 맞추려고) |
| 프롬프트·파서·해상도·토큰 한도 | | 동일 (`common_eval`) |

임베딩과 lm_head는 원래도 양자화 대상이 아니므로 bf16 그대로 CPU에서 계산할 뿐, 수치는 같습니다.
9B로 확인한 결과 CPU 실행 여부와 무관하게 a~d 로그확률이 소수점까지 일치했습니다.

## 준비

27B 가중치는 약 52 GiB입니다. 저장소 루트에서 한 번 받아 두세요.

```bash
baseline/Scripts/python.exe scripts/download_challenge_models.py Qwen/Qwen3.5-27B
```

## 비용

9B에 같은 오프로딩을 걸었을 때 문항당 1.3초였고, 27B는 대략 3배로 **672문항 40~60분**을 예상합니다.
문항마다 기록하고 `RESUME`을 지원합니다. 적재에만 수 분이 걸립니다(52 GiB를 읽어 4bit로 변환).
''')

code(NOTEBOOK_SETUP + '''
import gc, json, math, os
from pathlib import Path

import torch
import pandas as pd

import common_eval as ce
import vlm_runtime as vr

MODEL_DIR = Path("downloads/models/Qwen3.5-27B")
OUTPUT_DIR = Path("output/exp_E4_qwen35_27b")
MAX_NEW_TOKENS = 24          # E1과 동일
VLM_4BIT = True              # E1과 동일. 27B bf16은 약 52 GiB
# 16 GiB에 맞추기 위한 배치. OOM이면 CPU_LAST_LAYERS를 2, 4로 올리세요(느려집니다).
CPU_MODULES = ("model.language_model.embed_tokens", "lm_head")
CPU_LAST_LAYERS = 0
RESUME = True
SMOKE_TEST = False           # True면 앞 8문항만 (배관 점검용, 비교에 쓰지 말 것)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"

assert torch.cuda.is_available(), "이 실험 설정은 CUDA GPU를 사용합니다."
vr.check_weights(MODEL_DIR)
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.manual_seed(42)
ce._self_test(); vr._self_test()
print("GPU:", torch.cuda.get_device_name(0), "/ dtype:", DTYPE, "/ 4bit:", VLM_4BIT)
print("CPU 배치:", CPU_MODULES, "/ 마지막 층", CPU_LAST_LAYERS, "개")
''')

md('''
## 검증 분할

A~E3가 쓴 고정 672문항입니다. ID 해시를 대조해 같은 분할임을 확인합니다.
''')

code('''
val_df = ce.load_val()
if SMOKE_TEST:
    print("SMOKE_TEST: 공식 비교 결과로 사용 금지")
    val_df = val_df.head(8).copy()
rows = val_df.drop(columns=["answer"]).to_dict("records")   # 정답은 입력에서 제거
PRED_PATH = OUTPUT_DIR / ("predictions_smoke.jsonl" if SMOKE_TEST else "predictions.jsonl")
print("대상", len(rows), "문항 / 기록:", PRED_PATH)
print(vr.build_messages(rows[0], None)[1]["content"][1]["text"])
''')

md('''
## 추론

`vlm_runtime.run_predictions`가 E1과 같은 메시지·탐욕 생성·`parse_choice`로 답을 읽습니다.
INVALID는 E1처럼 그대로 남깁니다(로짓으로 채우지 않음).

추가로 첫 토큰의 **a~d 로그확률**(`letter_logprobs`)을 기록합니다. 예측에는 쓰지 않고,
나중에 9B·27B 앙상블이나 선지 순서 TTA를 추가 추론 없이 계산하려고 남깁니다.
''')

code('''
model, proc = vr.load_vlm(MODEL_DIR, four_bit=VLM_4BIT, dtype=DTYPE,
                          cpu_modules=CPU_MODULES, cpu_last_layers=CPU_LAST_LAYERS)
print(f"적재 후 GPU {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
try:
    peak = vr.run_predictions(model, proc, rows, PRED_PATH, limit=MAX_NEW_TOKENS,
                              resume=RESUME, desc="E4: 27B")
finally:
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
print(f"완료 / GPU peak {peak:.2f} GiB")
''')

md('''
## 분석
''')

code('''
pred_df = vr.read_predictions(PRED_PATH, val_df)
print(f"채점 대상 {len(pred_df)} / 전체 {len(val_df)}")
score = ce.save_analysis(pred_df, "E4_qwen35_27b", OUTPUT_DIR, extra=dict(
    vlm_model=str(MODEL_DIR), vlm_4bit=VLM_4BIT, image_max_edge=ce.IMAGE_MAX_EDGE,
    answer_max_new_tokens=MAX_NEW_TOKENS, enable_thinking=False, cot=False,
    cpu_modules=list(CPU_MODULES), cpu_last_layers=CPU_LAST_LAYERS, peak_allocated_gib=peak,
    answer_truncated_rate=float(pred_df.answer_truncated.mean()),
    uses_image=True, uses_ocr=False, smoke=SMOKE_TEST))

print("\\n예측 분포 (정답 분포와 비슷해야 정상)")
comp = pd.DataFrame({"정답": val_df.answer.value_counts().reindex(ce.CHOICES),
                     "예측": pred_df.pred.value_counts().reindex(ce.CHOICES)}).fillna(0).astype(int)
print(comp.to_string())
''')

md('''
## E1과 짝 비교

**이 실험의 결론을 내는 셀입니다.** 한쪽만 맞힌 문항 수로 McNemar를 보고,
E1 오답 분류별로 몇 개를 고쳤는지 봅니다. 가설이 맞다면 `misread`·`spatial`에서 많이 고치고
`literal_convention`은 거의 그대로여야 합니다. 후자가 남으면 E5(QLoRA)가 겨냥할 자리입니다.
''')

code('''
paired = vr.paired_vs_e1(pred_df, "E4")

# 두 모델이 서로 다른 문항을 틀린다면 로그확률 앙상블의 여지가 있습니다.
e1 = pd.read_json(vr.E1_PRED, lines=True)[["id", "pred"]].rename(columns={"pred": "pred_e1"})
j = pred_df.merge(e1, on="id")
both_wrong = int(((j.pred != j.answer) & (j.pred_e1 != j.answer)).sum())
print(f"\\n둘 다 틀린 문항 {both_wrong}개 — 9B+27B 앙상블로도 넘기 어려운 바닥")

print("\\n=== E4가 새로 틀린 문항 ===")
for _, r in j[(j.pred != j.answer) & (j.pred_e1 == j.answer)].head(5).iterrows():
    print(f"--- {r.id} 정답 {r.answer}({str(r[r.answer])[:24]}) / E4 {r.pred}: {r.question[:60]}")
''')

md('''
---

## 제출 추론 (test.csv)

위 검증과 **같은 모델·양자화·CPU 배치·프롬프트·생성 규칙**으로 test 6,714문항을 풉니다.
로컬 4bit 검증(E1 94.20%)과 제출(0.94727)이 가까웠던 것처럼, 여기서도 검증 96.88%와
같은 조건으로 제출해야 두 수치를 맞댈 수 있습니다.

**실행 순서:** 셀 1(설정) → 아래 셀들. 검증 섹션은 건너뛰어도 됩니다.
문항당 약 2.06초라 **약 3.8시간**이 걸리고, 문항마다 기록하므로 끊기면 추론 셀만 다시 실행하면 됩니다.

검증과 다른 점은 하나입니다. 제출 파일에 빈칸을 둘 수 없으므로 파서가 실패한 문항은
**기록해 둔 a~d 로그확률 중 가장 높은 글자**로 채웁니다(추가 forward 없음).
특정 글자로 메우면 그 글자만 과대 예측됩니다(초기 베이스라인 `a` 42%).
''')

code('''
import re

# --- 제출 설정 (여기만 바꾸면 됩니다) --------------------------------------
SUBMISSION_CSV     = Path("data/test.csv")               # 추론 대상
SAMPLE_SUBMISSION  = Path("data/sample_submission.csv")  # id 순서 기준. test가 아니면 무시
SUBMISSION_METHOD  = "zeroshot-4bit"                     # 파일명 방식 태그
SUBMISSION_VERSION = None        # None이면 같은 모델·방식의 다음 번호를 자동으로 붙입니다
SUBMISSION_LIMIT   = None        # int면 앞 N개만(배관 점검용). 제출 파일은 만들지 않습니다
SUBMIT_RESUME      = True        # 기록된 문항을 건너뛰고 이어서 진행
SEC_PER_ITEM       = 2.06        # E4 검증 실측. 남은 시간 추정에만 씁니다
# ---------------------------------------------------------------------------

SUBMISSION_DIR = Path("output") / f"exp_E4_{SUBMISSION_CSV.stem}"   # test.csv -> exp_E4_test
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
SUB_PRED_PATH = SUBMISSION_DIR / "predictions.jsonl"
META_PATH = SUBMISSION_DIR / "run_meta.json"
RUN_META = {"source_csv": str(SUBMISSION_CSV), "model": str(MODEL_DIR), "vlm_4bit": VLM_4BIT,
            "max_new_tokens": MAX_NEW_TOKENS, "enable_thinking": False,
            "image_max_edge": ce.IMAGE_MAX_EDGE, "cpu_modules": list(CPU_MODULES),
            "cpu_last_layers": CPU_LAST_LAYERS}

target_df = pd.read_csv(SUBMISSION_CSV, keep_default_na=False)
missing_cols = sorted({"id", "path", "question", "a", "b", "c", "d"} - set(target_df.columns))
assert not missing_cols, f"CSV 누락 컬럼: {missing_cols}"
assert target_df.id.is_unique, "id 중복이 있습니다"
missing_img = [p for p in target_df.path if not (ce.DATA_DIR / str(p)).is_file()]
assert not missing_img, f"이미지 누락 {len(missing_img)}개. 예: {missing_img[:3]}"

IS_TEST_SPLIT = SUBMISSION_CSV.name == "test.csv"
sample_df = pd.read_csv(SAMPLE_SUBMISSION, keep_default_na=False) if IS_TEST_SPLIT else None
if sample_df is not None:
    assert list(target_df.id) == list(sample_df.id), \\
        "test.csv와 sample_submission.csv의 id 순서가 다릅니다. 제출 형식이 깨집니다"

# 조건이 다른 예측이 한 파일에 섞이지 않도록 기존 기록의 실행 조건을 대조합니다.
if SUBMIT_RESUME and SUB_PRED_PATH.is_file() and META_PATH.is_file():
    prev = json.loads(META_PATH.read_text(encoding="utf-8"))
    changed = {k: f"{prev.get(k)} -> {v}" for k, v in RUN_META.items() if prev.get(k) != v}
    assert not changed, f"기존 예측과 실행 조건이 다릅니다: {changed}. SUBMIT_RESUME=False로 새로 시작하세요"
META_PATH.write_text(json.dumps(RUN_META, ensure_ascii=False, indent=2), encoding="utf-8")

target_rows = target_df.drop(columns=["answer"], errors="ignore").to_dict("records")
if SUBMISSION_LIMIT:
    target_rows = target_rows[:SUBMISSION_LIMIT]
done = 0
if SUBMIT_RESUME and SUB_PRED_PATH.is_file():
    done = sum(1 for line in SUB_PRED_PATH.open(encoding="utf-8") if line.strip())
left = max(len(target_rows) - done, 0)
print(f"대상 {len(target_rows)}문항 / 기록됨 {done} / 남은 {left} "
      f"(약 {left * SEC_PER_ITEM / 3600:.1f}시간) / 출력: {SUBMISSION_DIR}")
''')

code('''
model, proc = vr.load_vlm(MODEL_DIR, four_bit=VLM_4BIT, dtype=DTYPE,
                          cpu_modules=CPU_MODULES, cpu_last_layers=CPU_LAST_LAYERS)
try:
    sub_peak = vr.run_predictions(model, proc, target_rows, SUB_PRED_PATH, limit=MAX_NEW_TOKENS,
                                  resume=SUBMIT_RESUME, desc="E4 제출 추론")
finally:
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
print(f"완료 / GPU peak {sub_peak:.2f} GiB")
''')

md('''
### 제출 파일 생성과 검증

행 수·id 순서·값 범위를 확인하고, 예측 분포가 한쪽으로 쏠리면 경고합니다.
확신도(첫 토큰 a~d 확률의 최댓값)가 낮은 문항 수도 함께 봅니다. 검증에서는 0.8 이하 47문항(7%)에
오답 21건 중 약 14건이 몰려 있었으므로, 이 문항만 다시 묻는 후속 실험의 규모를 가늠할 수 있습니다.
''')

code('''
def submission_filename():
    """submission_<모델>_<방식>_v<버전>.csv. 버전을 비우면 기존 파일을 보고 다음 번호를 붙입니다."""
    stem = f"submission_{MODEL_DIR.name.replace('.', '').lower()}_{SUBMISSION_METHOD}"
    version = SUBMISSION_VERSION
    if version is None:
        pattern = re.compile(re.escape(stem) + r"_v(\\d+)$")
        used = [int(m.group(1)) for p in SUBMISSION_DIR.glob(stem + "_v*.csv")
                if (m := pattern.fullmatch(p.stem))]
        version = max(used, default=0) + 1
    return f"{stem}_v{version}.csv"


sub_pred = pd.read_json(SUB_PRED_PATH, lines=True).drop_duplicates("id", keep="last")
fallback = sub_pred.pred == "INVALID"
sub_pred.loc[fallback, "pred"] = sub_pred.loc[fallback, "letter_logprobs"].map(
    lambda lp: max(lp, key=lp.get))
conf = sub_pred.letter_logprobs.map(lambda lp: math.exp(max(lp.values())))
print(f"기록된 예측 {len(sub_pred)} / 대상 {len(target_rows)} / 로그확률 폴백 {int(fallback.sum())}건")
print(f"확신도 0.8 이하 {int((conf <= 0.8).sum())}문항 ({(conf <= 0.8).mean():.1%})")

if SUBMISSION_LIMIT:
    print("SUBMISSION_LIMIT이 설정되어 제출 파일을 만들지 않습니다.")
else:
    assert len(sub_pred) >= len(target_df), \\
        f"예측 부족 {len(sub_pred)}/{len(target_df)}. 추론 셀을 다시 실행해 이어서 채우세요"
    sub = target_df[["id"]].merge(sub_pred[["id", "pred"]], on="id", how="left",
                                  validate="one_to_one")
    assert sub.pred.isin(ce.CHOICES).all(), f"a~d 이외의 값: {sorted(set(sub.pred) - set(ce.CHOICES))}"
    if sample_df is not None:
        assert list(sub.id) == list(sample_df.id), "id 순서가 sample_submission과 다릅니다"

    share = sub.pred.value_counts(normalize=True).reindex(ce.CHOICES).fillna(0)
    print("\\n예측 분포"); print(share.round(3).to_string())
    if share.max() > 0.35:
        print(f"경고: '{share.idxmax()}'가 {share.max():.1%}로 쏠렸습니다. 파서와 폴백을 점검하세요")

    if IS_TEST_SPLIT:
        out_path = SUBMISSION_DIR / submission_filename()
        assert not out_path.exists(), f"같은 이름의 제출 파일이 이미 있습니다: {out_path.name}"
        sub.rename(columns={"pred": "answer"}).to_csv(out_path, index=False, encoding="utf-8")
        print("제출 파일:", out_path.resolve())
    else:
        print("test.csv가 아니므로 제출 파일은 만들지 않습니다")
''')

md('''
### E1 제출과 비교

`output/exp_E1_test/`에 E1(9B, A100 bf16) 제출 예측이 있으면 두 예측이 갈린 문항 수를 봅니다.
정답이 없으므로 검정은 제출 점수가 나온 뒤에 합니다. 점수 차이가 `c − b`를, 불일치 문항 수가
`b + c`의 상한을 주므로 불일치 전부가 정오를 가른다고 놓으면 가장 보수적인 p값이 나옵니다.
''')

code('''
# Kaggle 0.94727을 받은 E1 제출 실행의 예측. exp_E1_test 바로 아래 파일은 배관 점검용이라 쓰지 않습니다.
E1_TEST_PRED = Path("output/exp_E1_test/submission_qwen35-9b_zeroshot-bf16_v1/predictions.jsonl")
if not E1_TEST_PRED.is_file():
    print("E1 제출 예측이 없어 비교를 건너뜁니다:", E1_TEST_PRED)
else:
    e1t = pd.read_json(E1_TEST_PRED, lines=True).drop_duplicates("id", keep="last")
    jt = sub_pred[["id", "pred"]].merge(e1t[["id", "pred"]].rename(columns={"pred": "pred_e1"}),
                                        on="id")
    disagree = int((jt.pred != jt.pred_e1).sum())
    print(f"{len(jt)}문항 중 E1과 예측이 갈린 문항 {disagree}개 ({disagree / len(jt):.1%})")
    print("검증에서는 672문항 중 30개(4.5%)가 갈렸습니다.")


def kaggle_delta(score_e4, score_e1=0.94727, n=len(target_df), disagree=None):
    """제출 점수 두 개로 보수적 McNemar p를 계산합니다(불일치 문항이 전부 정오를 갈랐다고 가정)."""
    net = round((score_e4 - score_e1) * n)
    d = disagree if disagree is not None else n
    c = (d + net) // 2
    b = d - c
    print(f"순증 {net}문항 / 불일치 {d} 가정 시 E4만 {c} vs E1만 {b} -> "
          f"p <= {vr.mcnemar_exact(b, c):.4g}")
''')

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/experiments/AI_Challenge2_expE4_qwen35_27b.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
