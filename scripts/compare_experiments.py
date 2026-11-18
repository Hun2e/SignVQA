"""실험 A/B/C/D의 metrics.json을 모아 비교표를 만듭니다.

사용: python scripts/compare_experiments.py
출력: output/experiment_comparison.csv + 콘솔 요약
"""

from _project import ROOT
import json

import pandas as pd

RUNS = [
    ("D_llm_blind", (ROOT / "output/exp_D_llm_blind/metrics.json"), False, False),
    ("A_ocr_llm", (ROOT / "output/paddleocr_vl16_halftrain/full/ocr_llm/metrics.json"), False, True),
    ("B_vlm_only", (ROOT / "output/exp_B_vlm_only/metrics.json"), True, False),
    ("C_ocr_vlm", (ROOT / "output/exp_C_ocr_vlm/metrics.json"), True, True),
]
RANDOM_BASELINE = 0.25


def main():
    rows, hashes = [], {}
    for name, path, uses_image, uses_ocr in RUNS:
        if not path.is_file():
            print(f"  (아직 없음) {name}: {path}")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        hashes[name] = m.get("validation_ids_sha256")
        rows.append({
            "experiment": name, "image": uses_image, "ocr": uses_ocr,
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "invalid_rate": m["invalid_rate"],
            "mean_sec": m.get("mean_inference_seconds"),
            "rows": m["validation_rows"],
        })
    if not rows:
        print("완료된 실험이 없습니다.")
        return

    # 같은 검증 분할에서 나온 결과가 아니면 비교 자체가 성립하지 않습니다.
    present = {k: v for k, v in hashes.items() if v}
    if len(set(present.values())) > 1:
        raise SystemExit(f"검증 분할이 서로 다릅니다. 비교 불가:\n{json.dumps(present, indent=2)}")
    if present:
        print(f"검증 분할 해시 일치: {next(iter(present.values()))[:16]}… ({len(present)}개 실험)\n")

    df = pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)
    df["vs_random"] = df.accuracy - RANDOM_BASELINE
    (ROOT / "output").mkdir(exist_ok=True)
    df.to_csv(ROOT / "output/experiment_comparison.csv", index=False, encoding="utf-8-sig")
    print(df.round(4).to_string(index=False))

    acc = dict(zip(df.experiment, df.accuracy))
    print("\n기여도 분해")
    print(f"  무작위 추측                     {RANDOM_BASELINE:.4f}")
    if "D_llm_blind" in acc:
        print(f"  D 언어 편향만                   {acc['D_llm_blind']:.4f}  ({acc['D_llm_blind']-RANDOM_BASELINE:+.4f})")
    for key, label in [("A_ocr_llm", "A OCR의 기여 (A-D)"), ("B_vlm_only", "B 이미지의 기여 (B-D)")]:
        if key in acc and "D_llm_blind" in acc:
            print(f"  {label:30s} {acc[key]-acc['D_llm_blind']:+.4f}")
    if "C_ocr_vlm" in acc and "B_vlm_only" in acc:
        delta = acc["C_ocr_vlm"] - acc["B_vlm_only"]
        print(f"  {'C OCR을 VLM에 얹은 순효과 (C-B)':30s} {delta:+.4f}   <- 핵심 지표")
        verdict = "OCR 단계가 값을 함" if delta > 0.02 else "OCR 단계가 값을 못 함 (B 채택 검토)"
        print(f"  판정: {verdict}")
    print("\n저장: output/experiment_comparison.csv")


if __name__ == "__main__":
    main()
