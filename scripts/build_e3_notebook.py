"""Create the E3 notebook: hide the options, transcribe first, then match.

가설은 E1 오답 분석에서 나왔습니다. CoT로도 못 고친 22건 중 정답 문자열이 추론문에
등장한 것은 5건뿐이었고, 모델은 간판의 '미술랭투스타'를 흔한 표기 '미슐랭 스타'로
교정해 읽었습니다. 선지를 함께 주면 선지가 판독을 유도한다는 뜻입니다.
E3는 1단계에서 선지를 감춰 그 닻을 끊습니다.

프롬프트와 파서는 src/common_eval.py에서 import합니다. 복사하면 조용히 어긋납니다.
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
# 실험 E3 — 선지를 감추고 먼저 읽기 (read-first)

**가설:** 질문과 선지를 한꺼번에 주면 **선지가 판독을 유도합니다.**

E1 오답 39건을 E2의 CoT 추론문으로 들여다본 결과입니다.

| | 정답 문자열이 추론문에 등장 |
| --- | ---: |
| CoT로도 못 고친 22건 | **5건 (23%)** |
| CoT가 구제한 17건 | 15건 (88%) |

근본적으로 틀릴 때 모델은 **정답을 후보로 올려본 적조차 없습니다.** 실패는 추론이 아니라
인지에서 납니다. 구체적으로는 네 가지로 나타났습니다.

- **정규화** — 간판의 `미술랭투스타 선정`을 `미슐랭 스타 선정`으로 교정해 읽음
- **유의어 치환** — `하여튼 희한해`를 `하여튼 이상해`로
- **지식으로 빈칸 채우기** — `우족`을 못 찾자 "우족은 사골과 같은 부위"라며 사골 가격을 씀
- **환각** — 햇반 용기에 `냉장 보관`이 "명확히 적혀 있습니다"

이 데이터셋의 오답 선지는 정확히 그걸 노립니다. 정답이 "이상해 보이는 쪽"이고
함정이 "자연스러운 쪽"입니다. 선지를 먼저 보여주면 모델은 자연스러운 쪽으로 끌려갑니다.

## E1과 달라지는 것 — 프롬프트 구성 하나뿐

| | E1 | E3 |
| --- | --- | --- |
| 1단계 | 없음 | **선지 없이** 이미지에서 글자만 전사 (160토큰) |
| 2단계 | 이미지 + 질문 + 선지 → 한 글자 | 이미지 + **전사문** + 질문 + 선지 → 한 글자 |
| 파서 | `parse_choice` | `parse_choice` (동일) |

모델·양자화·해상도·seed·토큰 한도는 **전부 E1과 같습니다.** 그래야 `E3 − E1`이
"선지를 보기 전에 읽었는가"의 효과가 됩니다.

## 비용

2패스라 E1(0.805초/건)보다 오래 걸립니다. 1단계가 전사문을 생성하므로 E2(5.93초/건)에
가깝고, 전체 **60~80분**을 예상합니다. 문항마다 기록하고 `RESUME`을 지원하므로
중간에 멈춰도 거기까지로 E1과 짝 비교가 됩니다.
''')

code(NOTEBOOK_SETUP + '''
import gc, json, os, time
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

import common_eval as ce

VLM_MODEL_DIR = Path("downloads/models/Qwen3.5-9B")
OUTPUT_DIR = Path("output/exp_E3_readfirst")
READ_MAX_NEW_TOKENS = 160   # 1단계 전사. E2 평균 생성이 81토큰이었으므로 충분합니다
ANSWER_MAX_NEW_TOKENS = 24  # 2단계. E1과 동일
VLM_4BIT = True             # E1과 동일. 9B bf16 18.0 GiB는 16 GiB GPU에 안 올라갑니다
ENABLE_THINKING = False
RESUME = True               # 남은 문항부터 이어서
SMOKE_TEST = False          # True면 앞 8문항만 (배관 점검용)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"

assert torch.cuda.is_available(), "이 실험 설정은 CUDA GPU를 사용합니다."
assert (VLM_MODEL_DIR / "config.json").is_file(), f"로컬 모델을 준비하세요: {VLM_MODEL_DIR}"
index = VLM_MODEL_DIR / "model.safetensors.index.json"
if index.is_file():
    need = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    miss = [f for f in need if not (VLM_MODEL_DIR / f).is_file()]
    assert not miss, f"가중치 shard {len(miss)}/{len(need)}개 누락: {miss[:2]}"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.manual_seed(42)
ce._self_test()
print("GPU:", torch.cuda.get_device_name(0), "/ dtype:", DTYPE, "/ 4bit:", VLM_4BIT)
print("이미지 긴 변 제한:", ce.IMAGE_MAX_EDGE, "px (E1과 동일)")
print(f"1단계 {READ_MAX_NEW_TOKENS}토큰 / 2단계 {ANSWER_MAX_NEW_TOKENS}토큰")
''')

md('''
## 검증 분할

A/B/C/D/E1/E2가 쓴 고정 672문항입니다. ID 해시를 대조해 같은 분할임을 확인합니다.
E1과 문항 단위로 맞대야 하므로 여기서 바꾸면 안 됩니다.
''')

code('''
val_df = ce.load_val()
rows = val_df.to_dict("records")
if SMOKE_TEST:
    rows = rows[:8]
    print("SMOKE_TEST: 앞 8문항만 돌립니다")
PRED_PATH = OUTPUT_DIR / "predictions.jsonl"
print("대상", len(rows), "문항 / 기록:", PRED_PATH)
''')

md('''
## 프롬프트 (2단계)

**1단계는 선지를 절대 넣지 않습니다.** 넣는 순간 이 실험은 무의미해집니다.
`common_eval._self_test()`가 1단계 프롬프트에 선지가 섞였는지 매번 검사합니다.

2단계는 전사문만 앞에 붙고 질문·선지 본문은 E1과 글자 단위로 같습니다. 이것도 자기 테스트가 봅니다.
''')

code('''
def build_read_messages(row, image):
    """1단계 — 선지 없이 전사만."""
    return [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM_READ}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": ce.make_read_prompt(row)}]}]


def build_match_messages(row, image, transcript):
    """2단계 — 전사문 + 질문 + 선지."""
    return [{"role": "system", "content": [{"type": "text", "text": ce.SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text",
                                          "text": ce.make_match_prompt(row, transcript)}]}]


print("--- 1단계 ---")
print(build_read_messages(rows[0], None)[1]["content"][1]["text"])
print("\\n--- 2단계 ---")
print(build_match_messages(rows[0], None, "(1단계 전사 결과가 여기 들어갑니다)")[1]["content"][1]["text"])
''')

md('''
## 모델 로드

E1과 동일한 적재 설정입니다. 폴백도 E1과 같은 a~d 로짓 비교를 씁니다.
2단계 프롬프트는 다음 토큰이 선지 글자이므로 로짓 비교가 유효합니다
(CoT에서 이게 왜 안 되는지는 트러블슈팅 20번 참고).
''')

code('''
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def load_vlm():
    proc = AutoProcessor.from_pretrained(str(VLM_MODEL_DIR), local_files_only=True)
    kwargs = dict(local_files_only=True, dtype=DTYPE, device_map={"": 0})
    if VLM_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(VLM_MODEL_DIR), **kwargs)
    return model.eval(), proc


def encode(proc, messages):
    return proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, return_tensors="pt",
                                    enable_thinking=ENABLE_THINKING)


@torch.inference_mode()
def generate_text(model, proc, inputs, limit):
    inputs = inputs.to(model.device)
    prefix_len = inputs["input_ids"].shape[-1]
    output = model.generate(**inputs, max_new_tokens=limit, do_sample=False, use_cache=True)
    new = output[0, prefix_len:]
    eos = model.generation_config.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    truncated = len(new) >= limit and (not len(new) or int(new[-1]) not in eos_ids)
    return proc.decode(new, skip_special_tokens=True).strip(), bool(truncated), int(len(new))


def letter_token_ids(proc):
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
    """2단계 입력에서만 호출합니다. 다음 토큰이 선지 글자인 자리여야 의미가 있습니다."""
    inputs = inputs.to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    scores = {ch: max(logits[i].item() for i in ids) for ch, ids in table.items()}
    return max(scores, key=scores.get)
''')

md('''
## 추론

문항마다 두 번 생성합니다. 한 건 끝날 때마다 `predictions.jsonl`에 기록하고 flush하므로
중간에 멈춰도 거기까지는 남습니다. 다시 실행하면 `RESUME`이 끝난 문항을 건너뜁니다.

전사문(`transcript`)을 통째로 저장합니다. 틀린 문항에서 **모델이 실제로 무엇을 읽었는지**
나중에 다시 볼 수 있어야 하기 때문입니다.
''')

code('''
done_ids = set()
if RESUME and PRED_PATH.is_file():
    with PRED_PATH.open(encoding="utf-8") as f:
        done_ids = {json.loads(line)["id"] for line in f if line.strip()}
    print(f"이어서 진행: 이미 끝난 {len(done_ids)}개 건너뜀")
elif not RESUME and PRED_PATH.is_file():
    PRED_PATH.unlink()

todo = [r for r in rows if r["id"] not in done_ids]
print(f"남은 {len(todo)}개")

model, proc = load_vlm()
LETTERS = letter_token_ids(proc)
torch.cuda.reset_peak_memory_stats()
n_fallback = 0
try:
    with PRED_PATH.open("a", encoding="utf-8") as log:
        for row in tqdm(todo, desc="E3: 전사 -> 대조"):
            torch.cuda.synchronize(); start = time.perf_counter()
            im = ce.open_image(row)

            # 1단계 — 선지 없이 전사
            read_inputs = encode(proc, build_read_messages(row, im))
            transcript, read_capped, read_tok = generate_text(
                model, proc, read_inputs, READ_MAX_NEW_TOKENS)

            # 2단계 — 전사문을 붙여 선택
            match_inputs = encode(proc, build_match_messages(row, im, transcript))
            raw, capped, n_tok = generate_text(
                model, proc, match_inputs, ANSWER_MAX_NEW_TOKENS)
            pred = ce.parse_choice(raw)
            fallback = pred == "INVALID"
            if fallback:
                pred = choose_by_logit(model, proc, match_inputs, LETTERS)
                n_fallback += 1

            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            log.write(json.dumps(dict(
                id=row["id"], transcript=transcript, transcript_truncated=read_capped,
                transcript_tokens=read_tok, raw_output=raw, pred=pred, fallback=fallback,
                answer_truncated=capped, answer_tokens=n_tok, seconds=elapsed,
                image_width=im.width, image_height=im.height), ensure_ascii=False) + "\\n")
            log.flush()
            del read_inputs, match_inputs, im
    peak = torch.cuda.max_memory_allocated() / 1024**3
finally:
    del model, proc
    gc.collect(); torch.cuda.empty_cache()
print(f"완료 / 폴백 {n_fallback}개 / GPU peak {peak:.2f} GiB")
''')

md('''
## 분석

`transcript_truncated`를 먼저 보세요. 높으면 160토큰 안에 전사를 못 끝낸 것이므로
정확도가 가설의 효과가 아니라 길이 제한을 측정한 값이 됩니다.
''')

code('''
results = [json.loads(l) for l in PRED_PATH.open(encoding="utf-8") if l.strip()]
res_df = pd.DataFrame(results).drop_duplicates("id", keep="last")
pred_df = val_df.merge(res_df, on="id", validate="one_to_one", sort=False)
pred_df["correct"] = pred_df.pred == pred_df.answer
print(f"채점 대상 {len(pred_df)} / 전체 {len(val_df)}")

score = ce.save_analysis(pred_df, "E3_readfirst", OUTPUT_DIR, extra=dict(
    vlm_model=str(VLM_MODEL_DIR), vlm_4bit=VLM_4BIT, image_max_edge=ce.IMAGE_MAX_EDGE,
    read_max_new_tokens=READ_MAX_NEW_TOKENS, answer_max_new_tokens=ANSWER_MAX_NEW_TOKENS,
    enable_thinking=ENABLE_THINKING, two_stage=True, options_hidden_in_stage1=True,
    fallbacks=int(pred_df.fallback.sum()),
    transcript_truncated_rate=float(pred_df.transcript_truncated.mean()),
    mean_transcript_tokens=float(pred_df.transcript_tokens.mean()),
    uses_image=True, uses_ocr=False, smoke=SMOKE_TEST))

print("\\n=== 진단 ===")
print(f"폴백                {int(pred_df.fallback.sum()):4d}건")
print(f"전사 160토큰 미완결 {int(pred_df.transcript_truncated.sum()):4d}건 "
      f"({pred_df.transcript_truncated.mean():.1%})")
print(f"평균 전사 토큰       {pred_df.transcript_tokens.mean():.1f}")
print(f"평균 소요            {pred_df.seconds.mean():.2f}초/건 "
      f"(E1 0.805 / E2 5.927)")
print("\\n전사 샘플 2건:")
for _, r in pred_df.head(2).iterrows():
    print(f"  [{r.answer} -> {r.pred}] {str(r.transcript)[:160]}")
''')

md('''
## E1과 짝 비교

**이 실험의 결론을 내는 셀입니다.** 같은 672문항을 같은 모델로 푼 대응 표본이므로
정확도 두 개를 나란히 놓는 것으로는 부족하고, 한쪽만 맞힌 문항 수로 McNemar를 봅니다.

E3가 이긴 문항의 전사문을 함께 출력합니다. 가설이 맞다면 **E1이 흔한 표기로 끌려갔던 자리에서
E3가 이상한 표기를 그대로 읽어낸 것**이 보여야 합니다.
''')

code('''
import math
from math import comb

E1_PRED = Path("output/exp_E1_qwen35_9b/predictions.jsonl")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


if not E1_PRED.is_file():
    print("E1 예측이 없어 짝 비교를 건너뜁니다:", E1_PRED)
else:
    e1 = pd.read_json(E1_PRED, lines=True).drop_duplicates("id", keep="last")
    joined = pred_df.merge(e1[["id", "pred"]].rename(columns={"pred": "pred_e1"}), on="id")
    ok1 = joined.pred_e1 == joined.answer
    ok3 = joined.pred == joined.answer
    n = len(joined)
    b = int((ok1 & ~ok3).sum())      # E1만 정답
    c = int((~ok1 & ok3).sum())      # E3만 정답
    diff = (c - b) / n
    se = math.sqrt(b + c) / n
    pval = mcnemar_exact(b, c)
    print(f"[{n}문항]")
    print(f"  E1 zeroshot  {ok1.mean():.4f} ({int(ok1.sum())}/{n})")
    print(f"  E3 read-first {ok3.mean():.4f} ({int(ok3.sum())}/{n})")
    print(f"  둘 다 정답 {int((ok1 & ok3).sum())} / 둘 다 오답 {int((~ok1 & ~ok3).sum())} "
          f"/ E1만 {b} / E3만 {c}")
    print(f"  E3 - E1 = {diff * 100:+.2f}pp  95%CI "
          f"[{(diff - 1.96 * se) * 100:+.2f}pp, {(diff + 1.96 * se) * 100:+.2f}pp]")
    print(f"  McNemar exact p = {pval:.4g} -> "
          + ("차이가 유의합니다" if pval < 0.05 else "두 조건은 구분되지 않습니다"))

    print("\\n=== E3만 맞힌 문항 (선지를 감춘 효과) ===")
    for _, r in joined[~ok1 & ok3].head(5).iterrows():
        print(f"--- {r.id} 정답 {r.answer}({r[r.answer][:24]}) / E1 {r.pred_e1}({r[r.pred_e1][:24]})")
        print(f"    전사: {str(r.transcript)[:150]}")
    print("\\n=== E1만 맞힌 문항 (전사가 오히려 방해한 경우) ===")
    for _, r in joined[ok1 & ~ok3].head(3).iterrows():
        print(f"--- {r.id} 정답 {r.answer}({r[r.answer][:24]}) / E3 {r.pred}({r[r.pred][:24]})")
        print(f"    전사: {str(r.transcript)[:150]}")
''')

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/experiments/AI_Challenge2_expE3_readfirst.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
