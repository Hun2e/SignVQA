"""Verify the generated E8 (TTA) Colab notebook without a GPU.

E8의 test 제출은 A100에서 약 1시간 40분입니다. 순서 복원 방향이나 Gate 연결이 한 곳만 틀려도 그 시간을
버리므로, 노트북의 3·4·6·7·8·9번 셀을 가짜 vLLM으로 실제 data/ 위에서 끝까지 실행합니다.

가짜 모델: 정답 선지에 높은 점수를 주되 a 자리를 더 좋아하는 버릇과 문항별 잡음이 있고,
Zoom 입력(넓이 ≥ ZOOM_MIN_PIXELS)에서는 정답을 조금 더 잘 봅니다.
확인하는 것: 셀이 오류 없이 돌고, 순서별·Zoom 기록과 보고서·제출 파일이 생기며,
Order4가 자리 버릇을 줄이고, Zoom 입력이 실제로 커지는지.

사용: python tests/verify_e8_colab_notebook.py [A|C]   (기본 A. data/train.csv·test.csv·이미지 필요, 수 분 소요)
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hashlib
import json
import math
import re
import shutil
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

import common_eval as ce
import tta
import vllm_eval as ve

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "A"
NB = ROOT / f"notebooks/experiments/AI_Challenge2_expE8_tta_{CONFIG}_colab.ipynb"
DATA = ROOT / "data"
TEST_ROWS = 60          # 제출 셀은 앞 60문항으로만 확인합니다
GOLD_SCORE = 2.0        # 가짜 모델이 정답 선지에 주는 점수
SLOT_A_BIAS = 1.2       # a 자리 버릇
NOISE = 3.0             # 문항·선지별 잡음 폭
ZOOM_BONUS = 2.0        # Zoom 입력에서 정답에 더하는 점수
OPTION_LINE = re.compile(r"^([abcd])\. (.*)$", re.M)


def noise(key):
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (h - 0.5) * NOISE


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ch = text.strip().lower()
        return [ce.CHOICES.index(ch) + 1] if ch in ce.CHOICES else [99]


class FakeProc:
    tokenizer = FakeTokenizer()

    class image_processor:  # noqa: N801 - HF 프로세서 속성 이름을 흉내 냄
        merge_size = 2

        def __new__(cls, images, return_tensors=None):
            w, h = images[0].size
            return {"image_grid_thw": [np.array([1, h // 16, w // 16])]}

    def apply_chat_template(self, messages, **kw):
        text = messages[-1]["content"][-1]["text"]
        return f"<|im_start|>user\n<|image_pad|>{text}<|im_end|>\n<|im_start|>assistant\n"


def fake_llm(gold_by_question):
    def generate(reqs, sampling, use_tqdm=False):
        outs = []
        for req in reqs:
            prompt, image = req["prompt"], req["multi_modal_data"]["image"]
            shown = dict(OPTION_LINE.findall(prompt))
            question = prompt.split("<|image_pad|>")[1].split("\na. ")[0]
            gold_text = gold_by_question.get((question, tuple(sorted(shown.values()))))
            zoom = image.width * image.height >= tta.ZOOM_MIN_PIXELS * 0.99
            scores = {ch: (GOLD_SCORE + (ZOOM_BONUS if zoom else 0) if shown.get(ch) == gold_text else 0.0)
                      + (SLOT_A_BIAS if ch == "a" else 0.0) + noise(question + shown.get(ch, ch)) for ch in ce.CHOICES}
            top = max(scores.values())
            norm = math.log(sum(math.exp(s - top) for s in scores.values())) + top
            lp = {ce.CHOICES.index(ch) + 1: SimpleNamespace(logprob=s - norm) for ch, s in scores.items()}
            best = max(scores, key=scores.get)
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(
                text=best, logprobs=[lp], finish_reason="length", token_ids=[1])]))
        return outs
    return SimpleNamespace(generate=generate)


def question_key(row):
    """가짜 모델이 프롬프트에서 문항을 알아보는 열쇠: 질문 줄 + 선지 첫 줄 묶음(순서 무관).

    같은 질문("이 가게의 이름은?")이 여러 문항에 있어 질문만으로는 구분되지 않습니다.
    """
    prompt = ce.make_prompt(row)
    return prompt.split("\na. ")[0], tuple(sorted(dict(OPTION_LINE.findall(prompt)).values()))


def gold_map(frames):
    out = {}
    for df in frames:
        for r in df.to_dict("records"):
            if r.get("answer") in ce.CHOICES:
                out[question_key(r)] = str(r[r["answer"]]).split("\n")[0]
    return out


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def prepare_drive(base):
    """노트북이 Drive에서 읽는 것: E7 어댑터 표시 파일, E7-A·E6 holdout 기록, (A가 아니면) E8-A 최종 표."""
    adapter = base / f"output/exp_E7_27b_{CONFIG}/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    for src, dst in [("output/exp_E7_27b_A/holdout", base / "output/exp_E7_27b_A/holdout"),
                     ("output/exp_E6_zeroshot_train", base / "output/exp_E6_zeroshot_train")]:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src / "predictions.jsonl", dst / "predictions.jsonl")
    if CONFIG != "A":
        (base / "output/exp_E8_tta_A").mkdir(parents=True)
        shutil.copy2(ROOT / "output/exp_E8_tta_A/holdout_final.csv", base / "output/exp_E8_tta_A/holdout_final.csv")


def install_fakes(gold):
    vllm = ModuleType("vllm")
    vllm.LLM = lambda **kw: fake_llm(gold)
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda path: FakeProc())
    sys.modules["vllm"], sys.modules["transformers"] = vllm, transformers


def patch_submit(src):
    """제출 셀: 실행하도록 켜고 test는 앞 TEST_ROWS문항만 씁니다."""
    src = src.replace("SUBMIT = False", "SUBMIT = True")
    for name in ("test.csv", "sample_submission.csv"):
        old = f'pd.read_csv(DATA_ROOT / "{name}", keep_default_na=False)'
        assert src.count(old) == 1, name
        src = src.replace(old, old + f".head({TEST_ROWS})")
    return src


def run_notebook(base):
    cells = code_cells(json.loads(NB.read_text(encoding="utf-8")))
    # 0 설치 · 1 import 확인 · 2 Drive/코드 · 3 설정 · 4 holdout · 5 병합 · 6 vLLM · 7 Order4 · 8 Zoom
    # · (A가 아니면) 9 E8-A 비교 · 마지막 제출
    run_cells = [3, 4, 6, 7, 8] + ([] if CONFIG == "A" else [9])
    assert len(cells) == run_cells[-1] + 2, len(cells)
    ns = dict(Path=Path, pd=pd, json=json, time=__import__("time"), gc=__import__("gc"),
              DATA_ROOT=DATA, DRIVE_BASE=base, ce=ce, ve=ve, tta=tta)
    local = lambda src: src.replace('Path("/content/', f'Path("{base}/content/')  # Colab 로컬 디스크 → 임시 폴더
    for k in run_cells:
        exec(compile(local(cells[k]), f"cell{k}", "exec"), ns)
    exec(compile(local(patch_submit(cells[-1])), "submit", "exec"), ns)
    return ns


def check(ns, base):
    run = base / f"output/exp_E8_tta_{CONFIG}"
    for perm in tta.ORDERS:
        assert sum(1 for _ in (run / "holdout" / perm / "predictions.jsonl").open()) == 1300, perm
    report = json.loads((run / "holdout_report.json").read_text())
    assert report["gate"] == {"max_prob": 0.85, "margin": 0.25} and report["gated"] > 0
    assert CONFIG == "A" or ("vs_E8A" in report and "target_fixed" in report), "E8-A 비교가 보고서에 없습니다"
    zoom = ve.read(run / "holdout_zoom" / tta.IDENTITY / "predictions.jsonl")
    assert len(zoom) == report["gated"] and (zoom.image_width * zoom.image_height >= tta.ZOOM_MIN_PIXELS * 0.99).all()
    base_share = ns["h_base"].pred.value_counts(normalize=True).get("a", 0)
    o4_share = ns["h_o4"].pred.value_counts(normalize=True).get("a", 0)
    assert o4_share < base_share, "Order4가 a 자리 버릇을 줄이지 못했습니다"
    changed = int((ns["h_o4"].set_index("id").pred != ns["h_final"].set_index("id").pred).sum())
    assert changed > 0, "Zoom 입력이 좋아졌는데 Blend가 답을 하나도 바꾸지 않았습니다"
    for name in ("order4", "order4-zoom-blend"):
        sub = pd.read_csv(run / f"submission_qwen35-27b_lora-{CONFIG}_{name}.csv")
        assert len(sub) == TEST_ROWS and sub.answer.isin(ce.CHOICES).all(), name
    print(f"a 예측 비율 {base_share:.3f} -> {o4_share:.3f} / Gate {report['gated']}문항 / Zoom이 바꾼 답 {changed}")


def main():
    ce.DATA_DIR = DATA
    train = pd.read_csv(DATA / "train.csv", keep_default_na=False)
    install_fakes(gold_map([train]))
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        prepare_drive(base)
        ns = run_notebook(base)
        check(ns, base)
    print("E8 노트북 검증 OK")


if __name__ == "__main__":
    main()
