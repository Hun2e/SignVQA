"""B와 C가 갈린 문항(McNemar 불일치 셀)을 열어보고 OCR이 실제로 기여했는지 확인합니다.

정확도 차이와 p값만으로는 "OCR이 도움이 됐다"를 말할 수 없습니다.
여기서는 갈린 문항마다 OCR 원문을 직접 대조해 두 가지를 셉니다.

1. 정답 선지 문구가 OCR 결과에 실려 있었는가 (양쪽 셀에서 같은 잣대로)
2. 그 문구의 글자가 실제로 작았는가 (Spotting 좌표로 글자 높이를 측정)

사용: python scripts/analyze_discordant.py [--list]
출력: 콘솔 요약 + output/discordant_bc.csv
"""

from _project import ROOT
import argparse
import json
import re
import statistics

import pandas as pd

B_PRED = ROOT / "output/exp_B_vlm_only/validation_predictions.csv"
C_PRED = ROOT / "output/exp_C_ocr_vlm/validation_predictions.csv"
OCR_RAW = ROOT / "output/paddleocr_vl16_halftrain/full/ocr_llm/stage1.jsonl"
OUT_CSV = ROOT / "output/discordant_bc.csv"

# 정답 문구가 OCR에 담겼다고 볼 최소 비율. 4글자 창 기준.
SUPPORT_THRESHOLD = 0.4
# Spotting 출력은 "텍스트" 뒤에 꼭짓점 4개(x, y 8개)를 LOC 토큰으로 붙입니다.
SEGMENT = re.compile(r"([^<]+)((?:<\|LOC_\d+\|>){8})")


def load_ocr():
    evidence = {}
    with OCR_RAW.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                evidence[item["id"]] = item["stage1_raw"]
    return evidence


def segments(raw):
    """(텍스트, 글자 높이) 목록. 높이는 이미지 높이를 1000으로 둔 값입니다."""
    out = []
    for text, loc in SEGMENT.findall(raw or ""):
        ys = [int(v) for v in re.findall(r"LOC_(\d+)", loc)][1::2]
        out.append((text.strip(), max(ys) - min(ys)))
    return out


def normalize(s):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(s)).lower()


def containment(gold, text):
    """정답 문구를 4글자 창으로 쪼개 text 안에 얼마나 들어 있는지."""
    g, t = normalize(gold), normalize(text)
    if not g or not t:
        return 0.0
    if len(g) <= 4:
        return 1.0 if g in t else 0.0
    windows = [g[i:i + 4] for i in range(len(g) - 3)]
    return sum(1 for w in windows if w in t) / len(windows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="갈린 문항을 하나씩 출력")
    args = parser.parse_args()

    for path in (B_PRED, C_PRED, OCR_RAW):
        if not path.is_file():
            raise SystemExit(f"필요한 산출물이 없습니다: {path}")

    b = pd.read_csv(B_PRED, keep_default_na=False)
    c = pd.read_csv(C_PRED, keep_default_na=False)
    m = b[["id", "question", "a", "b", "c", "d", "answer", "pred"]].merge(
        c[["id", "pred"]], on="id", suffixes=("_B", "_C"), validate="one_to_one")
    m["B_ok"] = m.pred_B == m.answer
    m["C_ok"] = m.pred_C == m.answer

    both = int((m.B_ok & m.C_ok).sum())
    neither = int((~m.B_ok & ~m.C_ok).sum())
    only_b = m[m.B_ok & ~m.C_ok]
    only_c = m[~m.B_ok & m.C_ok]
    print(f"대응표본 2x2 (n={len(m)})")
    print(f"  둘 다 정답 {both} / 둘 다 오답 {neither} / "
          f"B만 정답 {len(only_b)} / C만 정답 {len(only_c)}")

    evidence = load_ocr()
    all_heights = [h for raw in evidence.values() for _, h in segments(raw) if h > 0]
    quartiles = statistics.quantiles(all_heights, n=4)
    print(f"\nOCR 세그먼트 {len(all_heights)}개 / 글자 높이(이미지 높이 1000 기준)")
    print(f"  1사분위 {quartiles[0]:.0f} · 중앙값 {quartiles[1]:.0f} · 3사분위 {quartiles[2]:.0f}")

    rows = []
    for label, subset in (("C만 정답", only_c), ("B만 정답", only_b)):
        print(f"\n{label} {len(subset)}개")
        supported, heights = 0, []
        for _, r in subset.iterrows():
            gold = r[r.answer]
            raw = evidence.get(r.id, "")
            segs = segments(raw)
            best = max(((containment(gold, t), h, t) for t, h in segs), default=(0.0, 0, ""))
            # 지원 여부는 OCR 출력 전체 기준입니다. 정답 문구가 여러 줄로 쪼개져
            # 전사될 수 있어, 단일 세그먼트만 보면 실제보다 적게 셉니다.
            whole = containment(gold, raw)
            if whole >= SUPPORT_THRESHOLD:
                supported += 1
                # 글자 높이는 한 줄로 잡힌 경우에만 잴 수 있습니다.
                if best[0] >= SUPPORT_THRESHOLD:
                    heights.append(best[1])
            rows.append({"id": r.id, "cell": label, "question": r.question,
                         "answer": r.answer, "gold_text": gold,
                         "pred_B": r.pred_B, "pred_C": r.pred_C,
                         "ocr_containment": round(whole, 3),
                         "best_segment": best[2], "segment_height": best[1]})
            if args.list:
                print(f"  {r.id} 정답 {r.answer} / B {r.pred_B} / C {r.pred_C} "
                      f"| OCR 포함 {whole:.0%}")
                print(f"    {r.question}")
        rate = supported / len(subset) if len(subset) else 0
        print(f"  정답 문구가 OCR에 실려 있던 문항 {supported}개 ({rate:.0%})")
        if heights:
            print(f"  그중 한 줄로 잡힌 {len(heights)}개의 글자 높이 평균 "
                  f"{statistics.mean(heights):.0f} (전체 중앙값 {quartiles[1]:.0f})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_CSV}")
    print("두 셀의 지원율이 비슷하면 OCR 내용이 승패를 갈랐다는 근거가 없습니다.")


if __name__ == "__main__":
    main()
