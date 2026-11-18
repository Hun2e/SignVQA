"""Create the E5 notebook: QLoRA on Qwen3.5-9B, evaluated on the fixed 672 validation split.

E1 오답 중 7건은 '적힌 그대로가 정답'(미술랭투스타, LYDiA, 生맥주 …)인데 모델이 흔한 표기로
고쳐 읽어 틀렸습니다. 프롬프트로는 거의 안 고쳐졌으므로(E2 3건, E3 1건) 데이터로 규칙을 가르칩니다.
학습 코드는 src/qlora.py, 적재·추론은 src/vlm_runtime.py에서 import합니다.
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
# 실험 E5 — Qwen3.5-9B QLoRA

**가설:** E1이 틀리는 이유 중 하나는 판독이 아니라 **이 대회의 정답 규칙을 모르는 것**입니다.

간판의 `미술랭투스타`, `LYDiA`, `生맥주`처럼 이상한 표기가 정답인데, 모델은 이를 흔한 표기로
고쳐 읽습니다. E3에서 선지를 감춰도 교정이 일어났으니 이건 언어 사전지식입니다.
프롬프트로는 안 고쳐졌고(E2 3/7, E3 1/7), 정답이 붙은 문항 6,042개가 놀고 있습니다.

## E1과 달라지는 것 — 가중치(LoRA 어댑터)뿐

| | E1 | E5 |
| --- | --- | --- |
| 모델 | Qwen3.5-9B nf4 | 같은 모델 + **LoRA** |
| 학습 데이터 | 없음 | train.csv − 검증 672 = **6,042문항**, 1 epoch |
| 프롬프트·파서·해상도·생성 | | 동일 (`vlm_runtime`) |

학습 입력은 추론 입력과 **토큰 단위로 같습니다.** 추론 프롬프트 뒤에 정답 글자와 `<|im_end|>`만
붙이고, 손실도 그 두 토큰에만 겁니다. 스모크 테스트에서 프롬프트 부분이 추론 입력과 같은지 검사합니다.

## 설정 근거

| 항목 | 값 | 이유 |
| --- | --- | --- |
| LoRA 대상 | 언어 모델의 attention·linear attention·MLP 선형층 | 비전 인코더는 얼림. 판독 실패가 안 줄면 그때 merger를 엶 |
| r / alpha / dropout | 16 / 32 / 0.05 | 학습 파라미터 4,011만 개(0.42%) |
| lr | 1e-4, warmup 3% 후 cosine | 이미 94%라 과적합이 더 큰 위험 |
| 유효 배치 | 1 × 누적 16 | 이미지 크기가 제각각이라 배치 1 |
| epoch | 1 | 같은 이유 |

**검증 672문항으로 하이퍼파라미터를 여러 번 고르지 마세요.** 그 순간 val이 학습 데이터가 되고
LB 대리지표로서의 의미가 사라집니다. 튜닝이 필요하면 학습 데이터에서 따로 떼어 쓰세요.

## 비용 (RTX 5060 Ti 16 GiB)

스텝당 약 2초, 최대 8.6 GiB. 6,042문항 1 epoch는 **약 3.5시간**, 이어서 평가 약 15분입니다.
25 옵티마이저 스텝(400문항)마다 체크포인트를 남기므로 끊겨도 다시 실행하면 이어집니다.

27B QLoRA는 이 GPU에 들어가지 않습니다(4bit 가중치만으로 약 12 GiB, 학습 여유 없음).
E5가 E1보다 유의하게 좋을 때만 Colab A100에서 같은 설정으로 27B를 돌리세요.
''')

code(NOTEBOOK_SETUP + '''
import gc, json, os, time
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

import common_eval as ce
import vlm_runtime as vr
import qlora

MODEL_DIR = Path("downloads/models/Qwen3.5-9B")
OUTPUT_DIR = Path("output/exp_E5_qlora_9b")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR = 1e-4
GRAD_ACCUM = 16
EPOCHS = 1
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 25              # 옵티마이저 스텝 단위
MAX_NEW_TOKENS = 24          # 평가 생성. E1과 동일
RESUME = True
SMOKE_TEST = False           # True면 학습 32문항 / 평가 8문항 (배관 점검용)

if SMOKE_TEST:
    OUTPUT_DIR = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "_smoke")
CKPT_DIR = OUTPUT_DIR / "checkpoint"
ADAPTER_DIR = OUTPUT_DIR / "adapter"
TRAIN_LOG = OUTPUT_DIR / "train_log.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"

assert torch.cuda.is_available(), "이 실험 설정은 CUDA GPU를 사용합니다."
vr.check_weights(MODEL_DIR)
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
torch.manual_seed(42)
ce._self_test(); vr._self_test(); qlora._self_test()
print("GPU:", torch.cuda.get_device_name(0), "/ dtype:", DTYPE)
print(f"LoRA r={LORA_R} alpha={LORA_ALPHA} / lr={LR} / 유효 배치 {GRAD_ACCUM} / epoch {EPOCHS}")
''')

md('''
## 데이터

검증 672문항은 기준 분할을 그대로 읽고(ID 해시 대조), 학습은 train.csv에서 그 672문항을 뺀
나머지 전부입니다. ID와 이미지 경로가 검증과 겹치지 않는지 확인합니다.
''')

code('''
val_df = ce.load_val()
fit_df = qlora.load_train_rows(val_df)
if SMOKE_TEST:
    fit_df, val_df = fit_df.head(32), val_df.head(8).copy()
print(f"학습 {len(fit_df)}문항 / 정답 분포 {fit_df.answer.value_counts().sort_index().to_dict()}")

order = []
for epoch in range(EPOCHS):   # epoch마다 다른 순서. seed 고정이라 재개해도 같은 순서
    order += fit_df.sample(frac=1.0, random_state=42 + epoch).to_dict("records")
TOTAL_STEPS = len(order) // GRAD_ACCUM
WARMUP = max(1, int(TOTAL_STEPS * WARMUP_RATIO))
print(f"마이크로 스텝 {len(order)} / 옵티마이저 스텝 {TOTAL_STEPS} (warmup {WARMUP})")
''')

md('''
## 학습

문항 하나씩 추론과 같은 입력을 만들어 정답 두 토큰의 손실을 계산하고, 16개마다 한 번 갱신합니다.
옵티마이저 스텝마다 `train_log.jsonl`에 손실·학습률·소요 시간을 남깁니다.

손실이 처음부터 매우 작게 나오는 것이 정상입니다. E1이 이미 94%를 맞히므로 대부분의 문항은
확률이 1에 가깝고, 학습 신호는 주로 **E1이 틀리는 6% 쪽**에서 옵니다. 그래서 스텝별 손실보다
`max_loss`(누적 16문항 중 가장 큰 손실)가 줄어드는지를 보세요.
''')

code('''
model, proc = vr.load_vlm(MODEL_DIR, dtype=DTYPE)
model = qlora.attach_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
model.print_trainable_parameters()
model.train()

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=LR, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: qlora.cosine_lr(s, TOTAL_STEPS, WARMUP))

start_step = 0
if RESUME and (CKPT_DIR / "progress.json").is_file():
    start_step = qlora.load_checkpoint(model, optimizer, scheduler, CKPT_DIR)["step"]
    print(f"체크포인트에서 이어서: 옵티마이저 스텝 {start_step}/{TOTAL_STEPS}")
elif TRAIN_LOG.is_file():
    TRAIN_LOG.unlink()

torch.cuda.reset_peak_memory_stats()
try:
    with TRAIN_LOG.open("a", encoding="utf-8") as log, \\
            tqdm(total=TOTAL_STEPS, initial=start_step, desc="E5: QLoRA") as bar:
        for step in range(start_step, TOTAL_STEPS):
            t0 = time.perf_counter()
            losses = []
            for row in order[step * GRAD_ACCUM:(step + 1) * GRAD_ACCUM]:
                batch, tgt = qlora.make_example(proc, row, ce.open_image(row))
                loss = qlora.answer_loss(model, batch, tgt)
                (loss / GRAD_ACCUM).backward()
                losses.append(loss.item())
                del batch, tgt, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            item = dict(step=step + 1, loss=sum(losses) / len(losses), max_loss=max(losses),
                        lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                        seconds=time.perf_counter() - t0)
            log.write(json.dumps(item) + "\\n"); log.flush()
            bar.update(1); bar.set_postfix(loss=f"{item['loss']:.4f}", max=f"{item['max_loss']:.3f}")
            if (step + 1) % SAVE_EVERY == 0 or step + 1 == TOTAL_STEPS:
                qlora.save_checkpoint(model, optimizer, scheduler, {"step": step + 1}, CKPT_DIR)
    train_peak = torch.cuda.max_memory_allocated() / 1024**3
    model.save_pretrained(str(ADAPTER_DIR))
    print(f"학습 완료 / GPU peak {train_peak:.2f} GiB / 어댑터: {ADAPTER_DIR}")
finally:
    del model, proc, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
''')

code('''
tl = pd.read_json(TRAIN_LOG, lines=True)
window = max(1, len(tl) // 10)
summary = tl.assign(block=(tl.step - 1) // window).groupby("block").agg(
    steps=("step", "max"), loss=("loss", "mean"), max_loss=("max_loss", "mean"),
    lr=("lr", "last"), sec=("seconds", "mean"))
print(summary.round(4).to_string())
print(f"총 학습 시간 {tl.seconds.sum() / 3600:.2f}시간")
''')

md('''
## 평가

학습한 어댑터를 E1과 같은 nf4 모델 위에 얹고, E1과 같은 메시지·생성·파서로 672문항을 풉니다.
''')

code('''
PRED_PATH = OUTPUT_DIR / "predictions.jsonl"
model, proc = vr.load_vlm(MODEL_DIR, dtype=DTYPE, adapter_dir=ADAPTER_DIR)
rows = val_df.drop(columns=["answer"]).to_dict("records")
try:
    peak = vr.run_predictions(model, proc, rows, PRED_PATH, limit=MAX_NEW_TOKENS,
                              resume=RESUME, desc="E5: 평가")
finally:
    del model, proc
    gc.collect(); torch.cuda.empty_cache()

pred_df = vr.read_predictions(PRED_PATH, val_df)
score = ce.save_analysis(pred_df, "E5_qlora_9b", OUTPUT_DIR, extra=dict(
    vlm_model=str(MODEL_DIR), vlm_4bit=True, adapter=str(ADAPTER_DIR),
    lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, lr=LR,
    grad_accum=GRAD_ACCUM, epochs=EPOCHS, train_rows=len(fit_df), finetuned=True,
    image_max_edge=ce.IMAGE_MAX_EDGE, answer_max_new_tokens=MAX_NEW_TOKENS,
    peak_allocated_gib=peak, uses_image=True, uses_ocr=False, smoke=SMOKE_TEST))
''')

md('''
## E1과 짝 비교

**이 실험의 결론을 내는 셀입니다.** 가설이 맞다면 E1 오답 분류 중 `literal_convention`에서
가장 많이 고쳐야 합니다. `misread`만 줄었다면 규칙 학습보다 일반적인 적응 효과입니다.

E5만 틀린 문항도 꼭 보세요. 파인튜닝이 원래 잘하던 것을 망가뜨렸는지 여기서 드러납니다.
''')

code('''
paired = vr.paired_vs_e1(pred_df, "E5")

e1 = pd.read_json(vr.E1_PRED, lines=True)[["id", "pred"]].rename(columns={"pred": "pred_e1"})
j = pred_df.merge(e1, on="id")
print("\\n=== E5가 새로 틀린 문항 ===")
for _, r in j[(j.pred != j.answer) & (j.pred_e1 == j.answer)].head(8).iterrows():
    print(f"--- {r.id} 정답 {r.answer}({str(r[r.answer])[:24]}) / E5 {r.pred}: {r.question[:60]}")
''')

for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/experiments/AI_Challenge2_expE5_qlora_9b.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(out)
