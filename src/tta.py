"""E8 추론 단계 기법(TTA): 선지 순서 4회 평균(Order4) → 불확실 문항 선별(Gate) → 고해상도 재평가(Zoom) → 결합(Blend).

팀 0.97110 파이프라인(팀원 제출 패키지의
experiment/scripts/mcq_core.py, output_zoom_order4_bf16_v1/qwen36_zoom_colab.py)과 같은 규칙입니다.
Gate 기준과 Blend 비율은 팀이 자기 검증 1,343문항으로 정한 값을 **고치지 않고** 씁니다.
holdout에서 다시 고르면 holdout이 독립 확인 세트가 아니게 됩니다.

확률은 a~d 네 글자의 로그확률을 소프트맥스한 상대 점수입니다. 보정된 정답 확률이 아닙니다.
"""
import math
from pathlib import Path

import pandas as pd

import common_eval as ce

# 네 순환 순서. 모든 선지가 모든 자리에 정확히 한 번씩 놓입니다. perm[i]는 i번째 자리에 보이는 원래 선지입니다.
ORDERS = ("abcd", "bcda", "cdab", "dabc")
IDENTITY = ORDERS[0]

GATE_MAX_PROB = 0.85      # 평균 확률 최댓값이 이보다 낮으면 재평가
GATE_MARGIN = 0.25        # 1·2위 차이가 이보다 작으면 재평가
BLEND_WEIGHT = 0.5        # 재평가 확률의 비중

ZOOM_MIN_PIXELS = 1_474_560   # 작은 이미지는 이 넓이까지 키움 (팀 Zoom 입력과 같음)
ZOOM_MAX_PIXELS = 2_097_152   # 큰 이미지는 이 넓이까지 줄임. 비전 토큰 약 2,048개

PROB_COLS = [f"p_{ch}" for ch in ce.CHOICES]


def permute_row(row, perm):
    """perm 순서로 선지를 바꾼 문항 사본. 질문·이미지 경로·id는 그대로입니다."""
    shown = dict(row)
    for slot, orig in zip(ce.CHOICES, perm):
        shown[slot] = row[orig]
    return shown


def slot_probs(letter_logprobs):
    """화면 자리 a~d의 로그확률을 네 글자 안에서 소프트맥스합니다. 기록이 없으면 None."""
    if not isinstance(letter_logprobs, dict) or not letter_logprobs:
        return None
    vals = [letter_logprobs[ch] for ch in ce.CHOICES]
    top = max(vals)
    exps = [math.exp(v - top) for v in vals]
    return [e / sum(exps) for e in exps]


def to_original(probs, perm):
    """자리별 확률을 원래 선지 a~d 순서로 되돌립니다. 되돌리기는 한 번만 해야 합니다."""
    out = [0.0] * len(ce.CHOICES)
    for slot, orig in enumerate(perm):
        out[ce.CHOICES.index(orig)] = probs[slot]
    return out


def _pred(p):
    return ce.CHOICES[max(range(len(p)), key=p.__getitem__)]


def summarize(probs, order_preds=None):
    """원래 순서 확률 한 줄 → pred·conf·margin·순서 불일치."""
    top, second = sorted(probs, reverse=True)[:2]
    row = dict(zip(PROB_COLS, probs), pred=_pred(probs), conf=top, margin=top - second)
    row["disagree"] = bool(order_preds) and len(set(order_preds)) > 1
    return row


def order4(results):
    """{perm: ve.read 결과} → 문항별 평균 확률 표. 네 순서가 모두 같은 문항을 가져야 합니다."""
    ids = set(results[IDENTITY].id)
    per_order = {}
    for perm, res in results.items():
        if set(res.id) != ids:
            raise ValueError(f"{perm}: 문항 집합이 기준 순서와 다릅니다")
        for i, lp in zip(res.id, res.letter_logprobs):
            p = slot_probs(lp)
            if p is None:
                raise ValueError(f"{perm}/{i}: a~d 로그확률이 없습니다")
            per_order.setdefault(i, []).append(to_original(p, perm))
    rows = []
    for i, ps in per_order.items():
        mean = [sum(col) / len(ps) for col in zip(*ps)]
        rows.append(dict(id=i, **summarize(mean, [_pred(p) for p in ps])))
    return pd.DataFrame(rows)


def gate(df, max_prob=GATE_MAX_PROB, margin=GATE_MARGIN):
    """재평가할 문항 표시. 팀 기준: 최댓값 < 0.85 또는 차이 < 0.25 또는 순서 불일치."""
    return (df.conf < max_prob) | (df.margin < margin) | df.disagree


def blend(base, review, weight=BLEND_WEIGHT):
    """review에 있는 문항만 base와 섞고 나머지는 base를 유지합니다."""
    if not set(review.id) <= set(base.id):
        raise ValueError("재평가 문항이 기준 표에 없습니다")
    rev = review.set_index("id")[PROB_COLS]
    rows = []
    for rec in base.to_dict("records"):
        p = [rec[c] for c in PROB_COLS]
        if rec["id"] in rev.index:
            p = [(1 - weight) * a + weight * b for a, b in zip(p, rev.loc[rec["id"]])]
        rows.append(dict(id=rec["id"], reviewed=rec["id"] in rev.index, **summarize(p)))
    return pd.DataFrame(rows)


def zoom_image(row, min_pixels=ZOOM_MIN_PIXELS, max_pixels=ZOOM_MAX_PIXELS):
    """원본(960 제한 없이)을 넓이 [min_pixels, max_pixels] 안으로 맞춥니다. 작은 이미지는 키웁니다."""
    from PIL import Image
    with Image.open(ce.DATA_DIR / str(row["path"])) as im:
        im = im.convert("RGB")
        area = im.width * im.height
        target = min(max(area, min_pixels), max_pixels)
        if target != area:
            scale = math.sqrt(target / area)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.Resampling.LANCZOS)
        return im


def vision_tokens(proc, image):
    """HF 프로세서 기준 비전 토큰 수. Zoom이 실제로 입력을 키웠는지 확인할 때 씁니다."""
    grid = proc.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
    merge = getattr(proc.image_processor, "merge_size", 2)
    return int(grid.prod()) // (merge * merge)


def run_orders(ve, llm, proc, sampling, rows, out_dir, drive_dir=None, image_fn=None, desc=""):
    """네 순서를 차례로 추론해 순서별 폴더에 저장하고 {perm: 결과}를 돌려줍니다. 끊기면 이어 갑니다."""
    results = {}
    for perm in ORDERS:
        shown = [permute_row(r, perm) for r in rows]
        sub_drive = Path(drive_dir) / perm if drive_dir is not None else None
        results[perm] = ve.run(llm, proc, sampling, shown, Path(out_dir) / perm, sub_drive,
                               image_fn=image_fn, desc=f"{desc} {perm}")
    return results


def _self_test():
    row = {"id": "x", "a": "A", "b": "B", "c": "C", "d": "D"}
    assert [permute_row(row, "bcda")[s] for s in ce.CHOICES] == ["B", "C", "D", "A"]
    # 'bcda'로 보여 주고 자리 d(원래 a)를 고르면 원래 a가 가장 높아야 합니다.
    assert _pred(to_original([0.1, 0.1, 0.1, 0.7], "bcda")) == "a"
    p = slot_probs({"a": -0.1, "b": -3.0, "c": -1e9, "d": -5.0})
    assert abs(sum(p) - 1) < 1e-9 and p[2] == 0.0
    df = pd.DataFrame([summarize([0.9, 0.05, 0.03, 0.02], list("aaaa")),
                       summarize([0.9, 0.05, 0.03, 0.02], list("aaab")),
                       summarize([0.6, 0.3, 0.05, 0.05], list("aaaa"))])
    assert list(gate(df)) == [False, True, True]
    base = pd.DataFrame([dict(id="x", **summarize([0.45, 0.40, 0.1, 0.05]))])
    rev = pd.DataFrame([dict(id="x", **summarize([0.20, 0.70, 0.05, 0.05]))])
    assert blend(base, rev).pred[0] == "b"
    print("tta OK")


if __name__ == "__main__":
    _self_test()
