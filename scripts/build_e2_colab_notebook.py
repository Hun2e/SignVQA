"""Derive the E2 (CoT) Colab submission notebook from the E1 Colab notebook.

E1 Colab에 이미 Drive 동기화·RESUME·제출 파일 생성 배관이 다 있습니다. E2는 그 위에
프롬프트/파서/토큰 한도만 바꾸면 되므로, 복사본을 손으로 관리하지 않고 여기서 생성합니다.
E1 Colab 노트북을 고치면 이 스크립트를 다시 실행해 E2를 맞춰 주세요.
"""

from _project import ROOT
import ast
import json

root = ROOT / 'notebooks/experiments'
nb = json.loads((root / 'AI_Challenge2_expE1_qwen35_9b_colab.ipynb').read_text(encoding='utf-8'))


def source(index, text):
    nb['cells'][index]['source'] = text.strip().splitlines(keepends=True)


def new_cell(kind, text):
    cell = {'cell_type': kind, 'metadata': {}, 'source': text.strip().splitlines(keepends=True)}
    if kind == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell


# --- 0. 표지 -----------------------------------------------------------------
source(0, '''# 실험 E2 (Colab) — Qwen3.5-9B + CoT 제출 추론

E1 Colab 노트북([AI_Challenge2_expE1_qwen35_9b_colab.ipynb](AI_Challenge2_expE1_qwen35_9b_colab.ipynb))에
**CoT만** 얹은 것입니다. Drive 동기화·RESUME·제출 파일 생성은 그대로입니다.
이 파일은 `scripts/build_e2_colab_notebook.py`가 생성하므로 직접 고치지 마세요.

## E1 Colab과 달라지는 것

| | E1 | E2 |
| --- | --- | --- |
| system | `ce.SYSTEM` (설명 금지) | **`ce.SYSTEM_COT`** |
| 사용자 프롬프트 | `ce.make_prompt(row)` | **`ce.make_prompt(row, cot=True)`** |
| 출력 상한 | 24토큰 | **512토큰** |
| 파서 | `ce.parse_choice` | **`ce.parse_choice_cot`** |
| 폴백 | a~d 로짓 비교 | **비-CoT로 재질의 → 그래도 실패하면 로짓** |
| 출력 폴더 | `exp_E1_<대상>` | `exp_E2_<대상>` |

질문과 선지 본문은 두 모드가 **완전히 동일**하고 마지막 지시문 한 줄만 교체됩니다.

## 기본 대상은 test.csv 6714문항 (제출)

**학습이 없으므로 train.csv를 꼭 거칠 필요가 없습니다.** 제로샷이라 train.csv는
"학습용"이 아니라 그냥 정답이 붙은 문제지이고, 용도는 *제출하지 않고 미리 재보는 것* 하나뿐입니다.
Kaggle 점수로 확인할 생각이라면 바로 test.csv로 가는 편이 시간이 절반입니다.

이미 E1 zeroshot을 같은 A100 bf16으로 test.csv 전체에 돌려 **Kaggle 0.94727**을 받아 뒀습니다
(`output/exp_E1_test/`). 여기에 E2를 test.csv로 돌려 제출하면 6714문항 규모에서 바로 비교됩니다.

| | 이 노트북 (test.csv) | train.csv로 돌릴 때 |
| --- | --- | --- |
| E1 추가 실행 | 불필요 (0.94727 재사용) | 필요 — 약 37분 |
| 채점 | Kaggle | 내가 직접 |
| CoT가 이기면 | 제출 파일이 이미 나와 있음 | test.csv에 다시 4~5시간 |
| 총 시간 | **4~5시간** | 9~10시간 |
| 얻는 것 | 점수 두 개 + 불일치 분석 | 문항별 정오표·McNemar·오답 분석 |

**제출 횟수가 빠듯하다면** `SUBMISSION_CSV = DATA_ROOT / "train.csv"`로 바꾸세요.
채점 모드로 돌고, E1도 같은 train.csv로 돌려 두면 16번이 McNemar까지 해 줍니다.

참고로 로컬 4bit 672문항에서는 E2가 E1보다 +0.89%p(0.9509 vs 0.9420)였지만
McNemar exact p = 0.34, 차이 95% CI [-0.65%p, +2.44%p]로 **우연과 구분되지 않았습니다.**
표본이 작아서 판별이 안 된 것이지 차이가 없다는 뜻은 아닙니다.

## 실행 순서

1. `런타임 - 런타임 유형 변경`에서 **A100 GPU** 선택
2. 1번 설치 셀 실행 → **세션 다시 시작** → 2번부터 이어서 실행
3. `SUBMISSION_LIMIT = 8`로 한 번 돌려 파서·폴백·파일명 확인
4. 기본값 그대로 `test.csv` 본 추론(4~5시간) → 14번에서 제출 파일 생성
5. **16번**에서 E1 예측과 맞대어 불일치 문항 수 확인 → 제출 후 점수가 나오면
   같은 셀의 `kaggle_delta(...)`로 그 차이가 유의한지 판정
''')

# --- 10. 런타임 설정 ---------------------------------------------------------
source(10, '''FORCE_BF16 = False    # L4 등에서 bf16을 강제할 때만 True

VLM_4BIT = not (BF16_OK and (GPU_GIB >= 30 or FORCE_BF16))
DTYPE = torch.bfloat16 if BF16_OK else torch.float16
MAX_NEW_TOKENS = 512        # CoT 과정을 담을 길이. 로컬 E2와 동일 (검증에서 p99 211토큰)
RETRY_MAX_NEW_TOKENS = 24   # 폴백 재질의(비-CoT)용. 로컬 E1과 동일
ENABLE_THINKING = False     # Qwen3.5 자체 thinking 모드. CoT는 프롬프트로만 유도합니다
RETRY_ON_TRUNCATED = True   # 512토큰 안에 결론을 못 낸 출력은 파싱에 성공해도 재질의

torch.manual_seed(42)
print(f"{GPU_NAME}: 4bit={VLM_4BIT} / dtype={DTYPE}")
if VLM_4BIT:
    print("  -> 양자화 경로입니다. A100을 쓰면 bf16으로 더 빨라집니다")
else:
    print("  -> bf16 그대로 올립니다 (가중치 약 18.0 GiB)")
print(f"  CoT: 출력 상한 {MAX_NEW_TOKENS}토큰 / 폴백 재질의 {RETRY_MAX_NEW_TOKENS}토큰")
''')

# --- 13/14. 프롬프트 ---------------------------------------------------------
source(13, '''## 6. 프롬프트 · 모델 로드

로컬 E2 노트북 셀 5·7과 동일합니다. system과 마지막 지시문 한 줄만 E1과 다릅니다.
폴백용 비-CoT 프롬프트도 함께 만들어 둡니다.
''')

source(14, '''import gc, json, time
import pandas as pd
from tqdm.auto import tqdm


def build_messages(row, image):
    """CoT용. 본문은 E1과 같고 system과 마지막 지시문 한 줄만 다릅니다."""
    return [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM_COT}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": ce.make_prompt(row, cot=True)}]}]


def build_messages_plain(row, image):
    """폴백 재질의용. E1과 글자 단위로 같은 프롬프트입니다."""
    return [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": ce.make_prompt(row)}]}]


DEMO_ROW = {"question": "(예시) 이 간판의 상호명은 무엇인가요?",
            "a": "가게1", "b": "가게2", "c": "가게3", "d": "가게4"}
print(build_messages(DEMO_ROW, None)[1]["content"][1]["text"])
print("\\n파서 확인:", ce.parse_choice_cot("간판을 보면 ...\\n정답: c"))
''')

# --- 7번 섹션(신규): 검증 672문항 먼저 ----------------------------------------
VAL_MD = '''## 7. 평가 대상 고르기

| `SUBMISSION_CSV` | 정답 | 나오는 것 | 언제 |
| --- | --- | --- | --- |
| **`test.csv`** (기본) | 없음 | 제출 파일 | Kaggle 점수로 비교할 때 |
| `train.csv` | 있음 | 정확도·혼동행렬 | 제출 안 쓰고 미리 잴 때 |

학습이 없으므로 둘 중 아무거나 써도 됩니다. `train.csv`는 "학습에 쓴 데이터"가 아니라
그냥 정답이 붙은 문제지라, 거기에 돌려 채점해도 누수가 없습니다. 반대로 꼭 거쳐야 할
이유도 없습니다. 출력 폴더가 `exp_E2_test` / `exp_E2_train`으로 갈려 섞이지 않습니다.

`train.csv`로 돌린 수치는 **A~D 실험 비교표에 넣지 마세요.** 그 표는 672 분할 기준입니다.

아래 셀은 선택 사항입니다. 로컬의 `output/paddleocr_vl16_halftrain/full/train_half_val.csv`를
Drive의 `SSAFY/AI_Challenge2/output/paddleocr_vl16_halftrain/full/`에 올려 두면,
`train.csv`로 돌렸을 때 16번이 전체 점수와 함께 **로컬 실험과 같은 672 부분집합 점수**도 찍어
기존 수치와 이어 볼 수 있습니다. `test.csv`로만 갈 거면 없어도 됩니다.
'''

VAL_CODE = '''# 선택 — 로컬 4bit 실험이 쓴 672 분할의 ID. 있으면 부분집합 점수도 함께 보고합니다.
VAL_SRC = DRIVE_BASE / "output" / "paddleocr_vl16_halftrain" / "full" / "train_half_val.csv"
LOCAL_VAL_SHA16 = "0aad93d6210e1b23"       # 로컬 실험이 쓴 분할의 ID 해시 앞 16자리
VAL_IDS = set()

if VAL_SRC.is_file():
    val_preview = pd.read_csv(VAL_SRC, keep_default_na=False)
    VAL_IDS = set(val_preview.id)
    got = ce.ids_sha256(val_preview.id.tolist())[:16]
    print(f"672 분할 로드 ({len(VAL_IDS)}문항)")
    print(f"ID 해시 {got} / 로컬 기준 {LOCAL_VAL_SHA16}",
          "-> 일치" if got == LOCAL_VAL_SHA16 else "-> 불일치! 로컬 수치와 이어 보지 마세요")
else:
    print("672 분할이 Drive에 없습니다 (선택 사항):", VAL_SRC)
    print("올려 두면 전체 점수와 함께 로컬 실험과 같은 672 부분집합 점수도 찍습니다")
'''

nb['cells'][16:16] = [new_cell('markdown', VAL_MD), new_cell('code', VAL_CODE)]
# 이 뒤로 인덱스가 2씩 밀립니다.

# --- 8. 제출 설정 (원래 16/17 -> 18/19) ---------------------------------------
source(18, '''## 8. 제출 설정

`SUBMISSION_CSV` 한 줄만 바꾸면 대상이 바뀌고 출력 폴더도 따라갑니다.
결과는 `/content`에 쓰고 일정 간격마다 Drive로 복사하므로, 런타임이 끊겨도 이어서 돌릴 수 있습니다.

`RUN_META`에 `cot`와 토큰 한도가 들어가므로, E1 예측이 남은 폴더에 E2를 이어 붙이려 하면
RESUME 단계에서 막힙니다.

기본값은 `test.csv`(제출)입니다. 제출을 쓰지 않고 미리 재려면 `train.csv`로 바꾸세요.
출력 폴더가 `exp_E2_test` / `exp_E2_train`으로 갈리므로 두 실행이 섞이지 않습니다.
''')

source(19, '''# --- 제출 설정 (여기만 바꾸면 됩니다) --------------------------------------
# 기본값은 제출용 test.csv입니다. 제출을 쓰지 않고 미리 재려면 train.csv로 바꾸세요(채점 모드).
SUBMISSION_CSV     = DATA_ROOT / "test.csv"
SUBMISSION_NAME    = None        # 이름을 직접 정할 때. None이면 아래 두 값으로 자동 생성
SUBMISSION_METHOD  = "cot"       # 방식 태그. 파일명에 들어갑니다
SUBMISSION_VERSION = None        # None이면 같은 모델·방식의 다음 번호
SUBMISSION_LIMIT   = None        # int면 앞 N개만 추론(배관 점검용)
RESUME             = True        # 남은 문항부터 이어서
SYNC_EVERY         = 100         # 이 건수마다 Drive로 복사. CoT는 길어 E1보다 촘촘히 둡니다
BENCH_N            = 16          # 속도 측정에 쓸 건수
# ---------------------------------------------------------------------------

STEM = SUBMISSION_CSV.stem                       # train (채점) / test (제출)
OUT_DIR = Path("/content/out") / f"exp_E2_{STEM}"
DRIVE_DIR = DRIVE_BASE / "output" / f"exp_E2_{STEM}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = OUT_DIR / "predictions.jsonl"
DRIVE_PRED = DRIVE_DIR / "predictions.jsonl"
META_PATH = OUT_DIR / "run_meta.json"

# cot / 토큰 한도를 넣어 두면 E1 예측이 남은 폴더에 이어 붙는 사고를 RESUME이 막아 줍니다.
RUN_META = {"source_csv": SUBMISSION_CSV.name, "model": VLM_MODEL_DIR.name,
            "vlm_4bit": VLM_4BIT, "max_new_tokens": MAX_NEW_TOKENS,
            "enable_thinking": ENABLE_THINKING, "image_max_edge": ce.IMAGE_MAX_EDGE,
            "cot": True, "retry_max_new_tokens": RETRY_MAX_NEW_TOKENS,
            "retry_on_truncated": RETRY_ON_TRUNCATED}

# 이전 세션에서 Drive에 남긴 예측이 있으면 가져와 이어서 진행합니다.
if RESUME and not PRED_PATH.is_file() and DRIVE_PRED.is_file():
    import shutil
    shutil.copy2(DRIVE_PRED, PRED_PATH)
    if (DRIVE_DIR / "run_meta.json").is_file():
        shutil.copy2(DRIVE_DIR / "run_meta.json", META_PATH)
    print("Drive에서 이전 예측을 복구했습니다")

target_df = pd.read_csv(SUBMISSION_CSV, keep_default_na=False)
REQUIRED = ["id", "path", "question", "a", "b", "c", "d"]
assert not (set(REQUIRED) - set(target_df.columns)), "CSV 컬럼이 부족합니다"
assert target_df["id"].is_unique, "id 중복이 있습니다"
missing_img = [p for p in target_df.path if not (ce.DATA_DIR / str(p)).is_file()]
assert not missing_img, f"이미지 누락 {len(missing_img)}개. 예: {missing_img[:3]}"

if "answer" in target_df.columns:
    GOLD = target_df.set_index("id")["answer"].astype(str).str.strip().str.lower()
    print(f"정답 컬럼 발견 -> 채점 모드 ({len(GOLD)}개)")
else:
    GOLD = None
    print("정답 컬럼 없음 -> 제출 전용 모드")

IS_TEST_SPLIT = SUBMISSION_CSV.name == "test.csv"
sample_df = (pd.read_csv(DATA_ROOT / "sample_submission.csv", keep_default_na=False)
             if IS_TEST_SPLIT else None)
if sample_df is not None:
    assert list(target_df.id) == list(sample_df.id), "sample_submission과 id 순서가 다릅니다"

target_rows = target_df.drop(columns=["answer"], errors="ignore").to_dict("records")
if SUBMISSION_LIMIT:
    target_rows = target_rows[:SUBMISSION_LIMIT]
    print(f"SUBMISSION_LIMIT={SUBMISSION_LIMIT} -> 제출 파일은 만들지 않습니다")

done_ids = set()
if RESUME and PRED_PATH.is_file():
    if META_PATH.is_file():
        prev = json.loads(META_PATH.read_text(encoding="utf-8"))
        changed = {k: f"{prev.get(k)} -> {v}" for k, v in RUN_META.items() if prev.get(k) != v}
        assert not changed, (f"기존 예측과 실행 조건이 다릅니다: {changed}. "
                             "RESUME=False로 두거나 폴더를 비우세요")
    with PRED_PATH.open(encoding="utf-8") as f:
        done_ids = {json.loads(line)["id"] for line in f if line.strip()}
    print(f"이어서 진행: 이미 끝난 {len(done_ids)}개 건너뜀")
elif not RESUME and PRED_PATH.is_file():
    PRED_PATH.unlink()

META_PATH.write_text(json.dumps(RUN_META, ensure_ascii=False, indent=2), encoding="utf-8")
todo = [r for r in target_rows if r["id"] not in done_ids]
print(f"출력: {OUT_DIR}  (Drive 사본: {DRIVE_DIR})")
print(f"추론 대상 {len(todo)}개")
''')

# --- 9. 폴백 (원래 18/19 -> 20/21) -------------------------------------------
source(20, '''## 9. 폴백 · 파일명

파싱 실패를 특정 글자로 메우면 그 글자만 과대 예측되어 분포가 무너집니다
(기존 베이스라인이 `return "a"`를 써서 `a` 42%가 됐습니다).

**E1의 로짓 폴백을 CoT 입력에 그대로 쓰면 안 됩니다.** CoT 프롬프트에서 다음 토큰은
선지 글자가 아니라 설명의 첫 글자라, a~d 로짓을 비교해도 의미가 없습니다.
그래서 2단으로 갑니다.

1. 같은 이미지를 **E1과 동일한 비-CoT 프롬프트**로 한 번 더 묻습니다 (24토큰)
2. 그래도 INVALID면 **그 비-CoT 입력 위에서** a~d 로짓을 비교합니다

로컬 E2는 폴백이 없어 INVALID 1건이 그대로 오답이 됐습니다. 여기서는 0건이 됩니다.
''')

source(21, r'''import re


def letter_token_ids(proc):
    """a~d를 시작하는 첫 토큰 후보. 선행 공백·대소문자 변형을 모두 모읍니다."""
    tok = proc.tokenizer
    table = {}
    for ch in ce.CHOICES:
        cand = set()
        for form in (ch, " " + ch, ch.upper(), " " + ch.upper()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cand.add(int(enc[0]))
        assert cand, f"'{ch}'의 토큰 id를 찾지 못했습니다"
        table[ch] = sorted(cand)
    return table


@torch.inference_mode()
def choose_by_logit(model, proc, inputs, table):
    """다음 토큰이 선지 글자인 입력에서만 의미가 있습니다 (= 비-CoT 입력)."""
    inputs = inputs.to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    scores = {ch: max(logits[i].item() for i in ids) for ch, ids in table.items()}
    return max(scores, key=scores.get), scores


def answer_without_cot(model, proc, row, image, table):
    """CoT가 최종 선지를 내지 못했을 때의 폴백. (pred, 단계, 재질의 원문)을 돌려줍니다."""
    inputs = proc.apply_chat_template(
        build_messages_plain(row, image), add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=ENABLE_THINKING)
    raw, _, _ = generate_text(model, proc, inputs, RETRY_MAX_NEW_TOKENS)
    pred = ce.parse_choice(raw)
    if pred != "INVALID":
        return pred, "retry", raw
    pred, _ = choose_by_logit(model, proc, inputs, table)
    return pred, "logit", raw


def submission_filename():
    """submission_<모델>_<방식>_v<버전>.csv"""
    if SUBMISSION_NAME:
        name = str(SUBMISSION_NAME)
        return name if name.endswith(".csv") else name + ".csv"
    model_tag = VLM_MODEL_DIR.name.replace(".", "").lower()
    quant_tag = "4bit" if VLM_4BIT else "bf16"
    stem = f"submission_{model_tag}_{SUBMISSION_METHOD}-{quant_tag}"
    version = SUBMISSION_VERSION
    if version is None:
        pattern = re.compile(re.escape(stem) + r"_v(\d+)$")
        used = []
        for path in list(OUT_DIR.glob(stem + "_v*.csv")) + list(DRIVE_DIR.glob(stem + "_v*.csv")):
            matched = pattern.fullmatch(path.stem)
            if matched:
                used.append(int(matched.group(1)))
        version = max(used, default=0) + 1
    return f"{stem}_v{version}.csv"
''')

# --- 10. 모델 적재 (원래 20 -> 22) --------------------------------------------
source(22, '''## 10. 모델 적재

18 GB를 GPU에 올립니다. 1~2분 걸립니다. 이 셀은 **한 번만** 실행하세요.
''')

# --- 11. 추론 함수 (원래 22/23 -> 24/25) --------------------------------------
source(24, '''## 11. 추론 함수

측정 셀과 본 실행 셀이 같은 함수를 씁니다. 한 건 끝날 때마다 기록하므로
중간에 끊겨도 `RESUME`으로 이어집니다.

`RETRY_ON_TRUNCATED`가 켜져 있으면 512토큰 안에 결론을 못 낸 출력은 **파싱에 성공해도**
폴백으로 넘깁니다. 잘린 추론의 마지막 줄에 우연히 남은 글자를 최종 답으로 삼으면 안 되기 때문입니다
(로컬 검증에서 잘린 2건은 2건 모두 오답이었습니다).
''')

source(25, '''import shutil


def run_items(items, desc):
    """items를 추론해 predictions.jsonl에 append하고, 걸린 시간 목록을 돌려줍니다."""
    times, n_fallback = [], 0
    with PRED_PATH.open("a", encoding="utf-8") as log:
        for i, row in enumerate(tqdm(items, desc=desc), 1):
            torch.cuda.synchronize(); start = time.perf_counter()
            im = ce.open_image(row)
            inputs = proc.apply_chat_template(
                build_messages(row, im), add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=ENABLE_THINKING)
            raw, capped, n_tok = generate_text(model, proc, inputs, MAX_NEW_TOKENS)
            pred = ce.parse_choice_cot(raw)
            # 잘린 출력은 결론에 도달하지 못한 것이라, 파싱에 성공해도 최종 답으로 보지 않습니다.
            needs_fallback = pred == "INVALID" or (RETRY_ON_TRUNCATED and capped)
            stage, retry_raw = "", None
            if needs_fallback:
                pred, stage, retry_raw = answer_without_cot(model, proc, row, im, LETTERS)
                n_fallback += 1
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            times.append(elapsed)
            log.write(json.dumps(dict(
                id=row["id"], raw_output=raw, pred=pred, fallback=needs_fallback,
                fallback_stage=stage, retry_raw_output=retry_raw,
                answer_truncated=capped, answer_tokens=n_tok, seconds=elapsed,
                image_width=im.width, image_height=im.height), ensure_ascii=False) + "\\n")
            log.flush()
            del inputs, im
            if i % SYNC_EVERY == 0:
                shutil.copy2(PRED_PATH, DRIVE_PRED)
    shutil.copy2(PRED_PATH, DRIVE_PRED)
    shutil.copy2(META_PATH, DRIVE_DIR / "run_meta.json")
    if items:
        print(f"폴백 {n_fallback}개 ({n_fallback / len(items):.2%}) / "
              f"GPU peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    return times
''')

# --- 12. 속도 측정 (원래 24/25 -> 26/27) --------------------------------------
source(26, '''## 12. 속도 측정

앞 `BENCH_N`건으로 실제 속도를 재고 전체 소요 시간을 추정합니다.
여기서 나온 예측도 그대로 저장되므로 버려지는 계산은 없습니다.

참고: 로컬 RTX 5060 Ti(4bit, 최적화 커널 없음)에서 검증 672문항이 **5.927초/건**이었습니다
(E1은 0.805초/건 — CoT가 7.4배). 평균 생성 81토큰, p99 211토큰.
''')

source(27, '''import statistics

LOCAL_E2_SEC = 5.927          # 로컬 4bit CoT 평균 기준값

bench = run_items(todo[:BENCH_N], "속도 측정")
if bench:
    warm = bench[3:] if len(bench) > 3 else bench      # 앞 몇 건은 워밍업
    per = statistics.median(warm)
    remaining = len(todo) - len(bench)
    print(f"\\n워밍업 제외 중앙값 {per:.3f}초/건 (1번째 {bench[0]:.2f}초)")
    print(f"남은 {remaining}개 예상 {per * remaining / 60:.0f}분 "
          f"({per * remaining / 3600:.1f}시간)")
    print(f"로컬 4bit CoT {LOCAL_E2_SEC}초/건 대비 {LOCAL_E2_SEC / per:.1f}배")
    print(f"참고: E1은 같은 A100 bf16에서 0.334초/건 (6714문항 37분)이었습니다")
    if per * remaining > 4 * 3600:
        print("\\n4시간을 넘깁니다. Colab 세션이 중간에 끊길 것을 전제로 두세요.")
        print("  끊기면 Drive 사본에서 복구해 이어서 돌리면 됩니다 (13번 섹션 설명 참고).")
    if per > 10:
        print("\\n경고: 10초/건을 넘습니다. VRAM 포화나 CPU 폴백을 의심하세요.")
        print("  nvidia-smi로 다른 프로세스가 GPU를 쓰고 있지 않은지 확인하세요.")
else:
    print("측정할 문항이 없습니다 (이미 모두 끝났습니다)")
''')

# --- 13. 본 추론 (원래 26/27 -> 28/29) ----------------------------------------
source(28, '''## 13. 본 추론

위 예상 시간이 납득되면 실행하세요. Colab은 유휴 상태로 두면 끊기므로 탭을 열어 두세요.
끊겼다면 런타임을 다시 잡고 **1번 설치 → 세션 재시작 → 2번부터 이 셀까지** 실행하면
Drive에 저장된 예측을 복구해 이어서 진행합니다.

6714문항은 A100 bf16에서 4~5시간입니다(train.csv·test.csv 모두 6714행). 한 세션에 끝나지
않는 것이 정상이라고 보고, `SYNC_EVERY = 100`으로 최대 100건까지만 잃도록 해 두었습니다.
''')

source(29, '''done_now = set()
with PRED_PATH.open(encoding="utf-8") as f:
    done_now = {json.loads(line)["id"] for line in f if line.strip()}
rest = [r for r in target_rows if r["id"] not in done_now]
print(f"남은 {len(rest)}개")

times = run_items(rest, "E2 제출 추론 (CoT)")
print("완료")
''')

# --- 14. 제출 파일 (원래 28 -> 30) --------------------------------------------
source(30, '''## 14. 제출 파일 생성

폴백이 항상 a~d를 돌려주므로 INVALID는 남지 않습니다. 그래도 분포 점검은 그대로 합니다.
''')

# --- 15. 채점 (원래 30/31 -> 32/33) -------------------------------------------
source(32, '''## 15. 채점 (정답이 있는 CSV일 때)

기본값 `train.csv`로 돌렸다면 여기서 bf16 CoT의 6714문항 정확도가 나옵니다.

이 수치를 로컬 4bit 672문항(E1 0.9420 / E2 0.9509)과 **직접 비교하지 마세요.**
분할도 다르고 적재 방식도 다릅니다. 판단은 다음 16번의 짝 비교로 하세요.
''')

source(33, '''if GOLD is None:
    print("정답이 없는 CSV입니다. 채점을 건너뜁니다.")
else:
    scored = pred_df[["id", "pred", "seconds"]].copy()
    scored["answer"] = scored.id.map(GOLD)
    scored = scored.dropna(subset=["answer"])
    scored["correct"] = scored.pred == scored.answer

    ce.save_analysis(scored, f"E2_colab_{STEM}", OUT_DIR, extra=dict(
        vlm_model=str(VLM_MODEL_DIR), vlm_4bit=VLM_4BIT, gpu=GPU_NAME,
        image_max_edge=ce.IMAGE_MAX_EDGE, answer_max_new_tokens=MAX_NEW_TOKENS,
        enable_thinking=ENABLE_THINKING, cot=True, source_csv=SUBMISSION_CSV.name,
        fallbacks=int(pred_df.fallback.sum()),
        answer_truncated_rate=float(pred_df.answer_truncated.mean()),
        mean_answer_tokens=float(pred_df.answer_tokens.mean()),
        uses_image=True, uses_ocr=False))

    from math import sqrt
    k, n, z = int(scored.correct.sum()), len(scored), 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    print(f"\\n정확도 {p:.4f} ({k}/{n}) / 95% CI [{centre - half:.4f}, {centre + half:.4f}] "
          f"= +-{half * 100:.2f}pp")
    print("  로컬 4bit 672문항(E1 0.9420 / E2 0.9509)과는 분할도 적재도 달라 직접 비교 불가")
    print("  판단은 16번 짝 비교로 하세요")

    print("\\n=== CoT 진단 ===")
    stages = pred_df.fallback_stage.fillna("").value_counts()
    print(f"폴백             {int(pred_df.fallback.sum()):4d}건 "
          f"(재질의 {int(stages.get('retry', 0))} / 로짓 {int(stages.get('logit', 0))})")
    print(f"{MAX_NEW_TOKENS}토큰 내 미완결   {int(pred_df.answer_truncated.sum()):4d}건 "
          f"({pred_df.answer_truncated.mean():.1%})")
    print(f"평균 생성 토큰    {pred_df.answer_tokens.mean():.1f}")
    print("\\n출력 샘플 2건:")
    for _, r in scored.head(2).merge(pred_df[["id", "raw_output"]], on="id").iterrows():
        print(f"  [{r.answer} -> {r.pred}] {str(r.raw_output)[:300]}")

    for f in OUT_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, DRIVE_DIR / f.name)
    print("Drive로 결과 복사 완료:", DRIVE_DIR)
''')

# --- 17. 정리 (원래 32 -> 34, 아래 16번 삽입으로 다시 36이 됩니다) -------------
source(34, '''## 17. 정리

메모리를 비웁니다. 다시 돌리려면 10번 섹션(모델 적재)부터 실행하세요.
''')

# --- 16. E1 ↔ E2 짝 비교 (신규, 정리 셀 앞에 끼워 넣습니다) --------------------
COMPARE_MD = '''## 16. E1 ↔ E2 짝 비교

**이 실행의 결론을 내는 셀입니다.** 대상 CSV에 따라 두 가지로 동작합니다.

**`test.csv`(기본) — 정답이 없을 때.** 두 예측이 몇 문항에서 갈리는지, 분포가 어떻게
달라지는지 봅니다. 채점은 Kaggle이 하므로, 새 점수가 나오면 같은 셀의
`kaggle_delta(...)`에 넣으세요. 정답을 몰라도 **보수적인 McNemar 판정**은 됩니다.
점수 차이로 `c - b`를 알 수 있고, 불일치 문항 수가 `b + c`의 상한이기 때문입니다.
`b + c`가 클수록 p값이 커지므로, 불일치 전부가 정오를 가른다고 놓으면 가장 불리한 판정이 됩니다.

**`train.csv` — 정답이 있을 때.** 문항 단위 McNemar 검정과 정확도 차 신뢰구간을 계산합니다.
정확도 두 개를 나란히 놓는 것만으로는 부족합니다. 같은 문항을 같은 모델로 푼 대응 표본이라
독립 표본처럼 비교하면 검정력을 버립니다. 실제로 갈리는 것은 **한쪽만 맞힌 문항 수(b, c)** 뿐이고
McNemar는 정확히 그것만 봅니다.

E1 예측이 없으면 안내만 출력하고 넘어갑니다.
'''

COMPARE_CODE = '''import math
from math import comb

E1_DRIVE_DIR = DRIVE_BASE / "output" / f"exp_E1_{STEM}"


def load_preds(path):
    if not Path(path).is_file():
        return None
    frame = pd.read_json(path, lines=True).drop_duplicates("id", keep="last")
    return frame.set_index("id")["pred"]


def wilson(k, n, z=1.96):
    """정규근사는 극단값에서 구간이 0/1을 넘어가므로 Wilson 구간을 씁니다."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, centre - half, centre + half


def mcnemar_exact(b, c):
    """양측 exact 이항검정. b, c는 한쪽만 맞힌 문항 수입니다."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_report(label, ids, e1, e2, gold):
    ids = [i for i in ids if i in e1.index and i in e2.index and i in gold.index]
    if not ids:
        print(f"[{label}] 맞댈 문항이 없습니다")
        return
    ok1 = {i: e1[i] == gold[i] for i in ids}
    ok2 = {i: e2[i] == gold[i] for i in ids}
    n = len(ids)
    k1, k2 = sum(ok1.values()), sum(ok2.values())
    b = sum(1 for i in ids if ok1[i] and not ok2[i])     # E1만 정답
    c = sum(1 for i in ids if ok2[i] and not ok1[i])     # E2만 정답
    p1, lo1, hi1 = wilson(k1, n)
    p2, lo2, hi2 = wilson(k2, n)
    diff = (c - b) / n
    se = math.sqrt(b + c) / n
    pval = mcnemar_exact(b, c)
    print(f"\\n[{label}] {n}문항")
    print(f"  E1 zeroshot {p1:.4f} ({k1}/{n})  95%CI [{lo1:.4f}, {hi1:.4f}]")
    print(f"  E2 CoT      {p2:.4f} ({k2}/{n})  95%CI [{lo2:.4f}, {hi2:.4f}]")
    print(f"  둘 다 정답 {k1 - b} / 둘 다 오답 {n - k1 - c} / E1만 {b} / E2만 {c}")
    print(f"  E2 - E1 = {diff * 100:+.2f}pp  95%CI "
          f"[{(diff - 1.96 * se) * 100:+.2f}pp, {(diff + 1.96 * se) * 100:+.2f}pp]")
    print(f"  McNemar exact p = {pval:.4g} -> "
          + ("차이가 유의합니다" if pval < 0.05 else "두 조건은 구분되지 않습니다"))
    print(f"  예측 일치 {sum(1 for i in ids if e1[i] == e2[i]) / n:.3f} / "
          f"둘 중 하나라도 정답 {sum(1 for i in ids if ok1[i] or ok2[i]) / n:.4f} (합치기 상한)")


E1_KAGGLE = 0.94727      # E1 zeroshot bf16 test.csv 제출 점수


def kaggle_delta(new_score, old_score=E1_KAGGLE, n_scored=None, m=None):
    """정답 없이, Kaggle 점수 차이와 불일치 문항 수만으로 보수적인 McNemar 판정을 합니다.

    b(E1만 정답)와 c(E2만 정답)를 따로는 알 수 없지만 두 가지는 압니다.
      c - b = (new - old) * n_scored     (점수 차이)
      b + c <= m                          (예측이 갈린 문항 수)
    b + c가 클수록 p값이 커지므로 b + c = m 이 가장 불리한 가정입니다.
    그 값으로도 유의하면 결론을 믿어도 됩니다.

    n_scored는 **LB가 실제로 채점한 행 수**입니다. public/private가 나뉘어 있으면
    전체 6714가 아니라 그 부분집합 크기를 넣으세요.
    """
    n_scored = n_scored if n_scored is not None else PAIR_N
    m = m if m is not None else PAIR_M
    gap = round((new_score - old_score) * n_scored)
    print(f"\\n점수 차 {(new_score - old_score) * 100:+.3f}pp = 약 {gap:+d}문항 / 불일치 {m}문항")
    if abs(gap) > m:
        print("  경고: 점수 차이가 불일치 문항보다 큽니다. n_scored를 LB 채점 행 수로 맞추세요")
        return
    lo = (m - abs(gap)) // 2                      # 적게 맞힌 쪽
    pval = mcnemar_exact(lo, m - lo)
    print(f"  가장 불리한 분할표 {lo} vs {m - lo} (불일치 전부가 정오를 가른다고 가정)")
    print(f"  McNemar exact p <= {pval:.4g} -> "
          + ("이 가정에서도 유의합니다" if pval < 0.05 else "유의하다고 말할 수 없습니다"))


e1_pred = load_preds(E1_DRIVE_DIR / "predictions.jsonl")
e2_pred = load_preds(DRIVE_PRED)
PAIR_N = PAIR_M = 0

if e1_pred is not None and e2_pred is not None and GOLD is None:
    # test.csv: 채점은 Kaggle이 합니다. 여기서는 두 예측이 어떻게 갈리는지만 봅니다.
    common = [i for i in target_df.id if i in e1_pred.index and i in e2_pred.index]
    PAIR_N = len(common)
    PAIR_M = sum(1 for i in common if e1_pred[i] != e2_pred[i])
    print(f"맞댄 문항 {PAIR_N} / 대상 {len(target_df)}")
    print(f"예측이 갈린 문항 {PAIR_M} ({PAIR_M / PAIR_N:.2%}) / 일치 {1 - PAIR_M / PAIR_N:.2%}")
    print("\\n예측 분포")
    print(pd.DataFrame({
        "E1 zeroshot": e1_pred[common].value_counts().reindex(ce.CHOICES).fillna(0).astype(int),
        "E2 CoT": e2_pred[common].value_counts().reindex(ce.CHOICES).fillna(0).astype(int),
    }).to_string())
    print("\\n정답이 없어 문항별 채점은 불가능합니다. 제출 후 점수가 나오면:")
    print(f"  kaggle_delta(new_score=0.95000)      # E1 {E1_KAGGLE} 와 비교")
    print("  public/private가 나뉘어 있으면 n_scored에 LB 채점 행 수를 넣으세요")
elif GOLD is None:
    print("정답이 없고 E1 예측도 없어 맞댈 것이 없습니다:",
          E1_DRIVE_DIR / "predictions.jsonl")
elif e1_pred is None:
    print("E1 예측을 찾지 못했습니다:", E1_DRIVE_DIR / "predictions.jsonl")
    print(f"E1 Colab 노트북에서 SUBMISSION_CSV를 {SUBMISSION_CSV.name}으로 두고 돌린 뒤")
    print("이 셀을 다시 실행하세요 (A100 bf16에서 약 37분).")
else:
    common = [i for i in target_df.id if i in e1_pred.index and i in e2_pred.index]
    print(f"짝 비교 가능한 문항 {len(common)} / 대상 {len(target_df)}")
    if len(common) < len(target_df):
        print("  한쪽이 덜 끝났습니다. 아래 수치는 겹치는 문항에 한정됩니다.")
    paired_report(f"{STEM} 전체", common, e1_pred, e2_pred, GOLD)
    if VAL_IDS:
        paired_report("672 부분집합 (로컬 4bit 실험과 같은 분할)",
                      [i for i in common if i in VAL_IDS], e1_pred, e2_pred, GOLD)
    print("\\n판단 기준: p < 0.05 이고 차이가 양수일 때만 CoT로 test.csv를 다시 돌릴 값어치가 있습니다.")
    print("  E2는 E1의 약 7배 시간이 듭니다. 구분되지 않으면 E1 제출을 유지하세요.")
'''

nb['cells'][34:34] = [new_cell('markdown', COMPARE_MD), new_cell('code', COMPARE_CODE)]

# --- 맺음말 (원래 34 -> 36 -> 38) ---------------------------------------------
source(38, '''---

## E1 Colab과 다른 점 (요약)

| | E1 Colab | E2 Colab |
| --- | --- | --- |
| system | `ce.SYSTEM` | `ce.SYSTEM_COT` |
| 프롬프트 | `make_prompt(row)` | `make_prompt(row, cot=True)` |
| 출력 상한 | 24토큰 | 512토큰 |
| 파서 | `parse_choice` | `parse_choice_cot` |
| 폴백 | a~d 로짓 | 비-CoT 재질의 → 로짓 |
| 잘린 출력 | 해당 없음 | 폴백으로 넘김 (`RETRY_ON_TRUNCATED`) |
| Drive 동기화 | 200건마다 | 100건마다 |
| 기본 대상 | `test.csv` | `test.csv` |
| 출력 폴더 | `exp_E1_<대상>` | `exp_E2_<대상>` |
| 제출 파일명 | `..._zeroshot-bf16_v1.csv` | `..._cot-bf16_v1.csv` |
| 6714문항 소요 | 약 37분 | **약 4~5시간** |
| 짝 비교 | 없음 | 16번 섹션 (McNemar) |

모델·양자화·이미지 규칙은 **동일**합니다. 그래야 `E2 − E1`이 CoT의 효과가 됩니다.

## 주의

- **E1을 다시 돌릴 필요는 없습니다.** `output/exp_E1_test/`에 같은 A100 bf16으로 돌린
  6714문항 예측이 있고 Kaggle 점수도 0.94727로 나와 있습니다. 16번이 그 파일을 읽습니다.
  단, Drive의 `output/exp_E1_test/predictions.jsonl`에 사본이 있어야 합니다
- **제출 횟수가 빠듯하면** `SUBMISSION_CSV = DATA_ROOT / "train.csv"`로 바꾸세요. 채점 모드로
  돌고, E1도 같은 train.csv로 돌려 두면(약 37분) 16번이 문항별 McNemar까지 해 줍니다.
  대신 CoT가 이겼을 때 test.csv에 다시 4~5시간이 듭니다
- **train.csv 수치를 A~D 비교표에 넣지 마세요.** 그 표는 672 분할 기준입니다.
  이어 보려면 16번이 함께 찍는 672 부분집합 값을 쓰세요
- **Kaggle 점수 차이를 그대로 믿지 마세요.** public/private가 나뉘어 있으면 소수점 차이로
  고르는 건 public 과적합입니다. 16번의 `kaggle_delta(...)`로 보수적 판정을 먼저 해 보세요
- **배관부터 확인.** `SUBMISSION_LIMIT = 8`로 한 번 돌려 파서·폴백·파일명을 본 뒤 본 추론으로
  가세요. A100은 컴퓨팅 유닛 소모가 큽니다
- **폴백 비율을 보세요.** 5%를 넘으면 모델이 `정답: x` 형식을 안 지키고 있는 것이고,
  그러면 정확도는 CoT의 효과가 아니라 형식 실패를 재는 값이 됩니다
- `predictions.jsonl`에 E1에 없던 `fallback_stage`, `retry_raw_output` 컬럼이 추가됩니다.
  E1 결과와 나란히 읽을 때 컬럼이 다른 점에 유의하세요
- Colab은 유휴 시 끊깁니다. 끊기면 Drive 사본에서 복구해 이어서 진행합니다
- 저장소가 비공개라 Colab에서 `git clone`이 인증 없이 실패합니다. `src/common_eval.py`를
  Drive의 `SSAFY/AI_Challenge2/src/`에 올려 두는 쪽이 가장 간단합니다
''')

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        # IPython shell escapes are parsed separately by IPython, not Python ast.
        code = ''.join(cell['source'])
        ast.parse('\n'.join(line[:len(line) - len(line.lstrip())] + 'pass'
                            if line.lstrip().startswith('!') else line for line in code.splitlines()))

out = root / 'AI_Challenge2_expE2_qwen35_9b_cot_colab.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(out)
