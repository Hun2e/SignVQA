"""실험 공통 코드.

모든 실험(A: OCR+LLM, B: VLM 단독, C: OCR+VLM, D: LLM blind)이
동일한 검증 분할 / 질문 프롬프트 / 출력 파서 / 분석 지표를 쓰도록 한곳에 모읍니다.
각 노트북이 코드를 복사하면 조용히 어긋나므로 반드시 import 해서 사용하세요.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CHOICES = list("abcd")
SYSTEM = ("You are a visual question answering assistant. "
          "Answer with exactly one lowercase letter: a, b, c, or d. No explanation.")
# 실험 A가 만든 기준 분할. 모든 실험은 이 val 672문항을 그대로 씁니다.
BASELINE_RUN = Path("output/paddleocr_vl16_halftrain/full")


ANSWER_INSTRUCTION = "이미지를 보고 정답 선지의 영문 소문자 한 글자만 답하세요."
# CoT에서는 "한 글자만" 지시와 "단계적으로 설명" 지시가 충돌하므로 마지막 줄을 통째로 교체합니다.
COT_INSTRUCTION = ("이미지에서 근거가 되는 부분을 찾아 간단히 단계적으로 설명한 뒤, "
                   "마지막 줄에 반드시 `정답: x` 형식으로 a, b, c, d 중 한 글자만 쓰세요.")


# E6: 정답은 사진 속 사실이 아니라 여러 교육생이 일치해서 고른 답입니다(응답이 갈린 문항은 제외됨).
# E1 오답 중 '라벨 오류'로 본 문항도 사람 기준으로는 일반인 상식(그리스 문자 -> 라틴어)이나
# 눈에 띄는 쪽(두 수건 중 앞의 흰 수건)을 고른 것이었습니다. 그 기준을 지시문으로 알려 봅니다.
CONSENSUS_INSTRUCTION = ("이 문제의 정답은 여러 사람이 같은 사진을 보고 일치해서 고른 답입니다. "
                         "사진에 적힌 표기를 그대로 따르고, 전문 지식보다 일반적인 상식과 "
                         "눈에 잘 띄는 정보를 기준으로 사람들이 가장 많이 고를 선지의 "
                         "영문 소문자 한 글자만 답하세요.")


def make_prompt(row, cot=False, consensus=False):
    """질문/선지 렌더링. 본문은 모든 실험에서 동일하고 마지막 지시문만 달라집니다."""
    assert not (cot and consensus), "CoT와 합의 지시문은 함께 쓰지 않습니다"
    instruction = COT_INSTRUCTION if cot else CONSENSUS_INSTRUCTION if consensus else ANSWER_INSTRUCTION
    return (f"질문: {row['question']}\n"
            f"a. {row['a']}\nb. {row['b']}\nc. {row['c']}\nd. {row['d']}\n"
            + instruction)


def parse_choice(raw):
    """출력이 정확히 선지 한 글자일 때만 인정하고, 나머지는 INVALID로 남깁니다."""
    raw = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S).strip()
    raw = re.sub(r"^(?:정답|답|answer)\s*[:：]\s*", "", raw, flags=re.I).strip()
    m = re.fullmatch(r"(?:\(?([abcd])\)?[.。]?)", raw, flags=re.I)
    return m.group(1).lower() if m else "INVALID"


# CoT 모드에서는 system 프롬프트의 '설명 금지'를 풀어야 합니다.
SYSTEM_COT = ("You are a visual question answering assistant. "
              "Think briefly, then end your reply with a final line "
              "in the form `정답: x` where x is one of a, b, c, d.")

_FINAL_RE = re.compile(r"(?:정답|답|answer|final answer)\s*[:：]\s*\(?\s*([abcd])\b",
                       re.IGNORECASE)
_BARE_RE = re.compile(r"(?<![a-z])\(?([abcd])\)?(?![a-z])", re.IGNORECASE)


def parse_choice_cot(raw):
    """CoT 출력에서 최종 선지를 추출합니다.

    parse_choice()는 출력이 정확히 한 글자일 때만 인정하므로 CoT 문장은 전부 INVALID가 됩니다.
    여기서는 `정답: x` 형태를 우선 찾고, 없으면 마지막 줄의 단독 a~d를 씁니다.
    추론 과정에 등장하는 선지 언급을 최종 답으로 오인하지 않도록 뒤에서부터 찾습니다.
    """
    text = re.sub(r"<think>.*?</think>", " ", str(raw), flags=re.S)
    matches = _FINAL_RE.findall(text)
    if matches:
        return matches[-1].lower()
    # `정답:` 표기가 없으면 마지막 비어 있지 않은 줄에서만 단독 글자를 찾습니다.
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        found = _BARE_RE.findall(line)
        if len(found) == 1:
            return found[0].lower()
        if found:
            break            # 한 줄에 여러 선지가 있으면 최종 답으로 볼 수 없습니다.
    return "INVALID"


# --- E3: 선지를 감추고 먼저 읽게 하는 2단계 프롬프트 -------------------------
# E1/E2는 질문과 선지 4개를 한꺼번에 줍니다. 그러면 선지가 판독을 유도합니다.
# 실제로 E1 오답 중 CoT로도 못 고친 22건에서 정답 문자열이 추론문에 등장한 것은 5건뿐이고,
# 모델은 간판의 '미술랭투스타'를 흔한 표기 '미슐랭 스타'로 교정해 읽었습니다.
# 1단계에서 선지를 감추면 그 닻이 사라지는지 보는 것이 E3의 가설입니다.

SYSTEM_READ = ("You are a careful transcription assistant. "
               "Write down exactly what is printed in the image. "
               "Never correct spelling, never normalize to a more common wording, never guess.")

READ_INSTRUCTION = ("위 질문에 답하는 데 필요한 부분의 글자를 이미지에 적힌 그대로 옮겨 적으세요. "
                    "맞춤법을 고치거나 더 흔한 표현으로 바꾸지 마세요. "
                    "글자가 보이지 않으면 '확인 불가'라고만 쓰세요.")

MATCH_INSTRUCTION = "전사 내용과 이미지를 함께 보고 정답 선지의 영문 소문자 한 글자만 답하세요."


def make_read_prompt(row):
    """1단계. 선지를 넣지 않습니다. 넣는 순간 실험이 무의미해집니다."""
    return f"질문: {row['question']}\n" + READ_INSTRUCTION


def make_match_prompt(row, transcript):
    """2단계. 1단계 전사를 앞에 붙이고 나머지는 E1과 동일하게 렌더링합니다."""
    return (f"이미지에서 읽은 글자: {str(transcript).strip()}\n\n"
            f"질문: {row['question']}\n"
            f"a. {row['a']}\nb. {row['b']}\nc. {row['c']}\nd. {row['d']}\n"
            + MATCH_INSTRUCTION)


IMAGE_MAX_EDGE = 960  # 원본 노트북의 공통 이미지 규칙. B와 C가 반드시 같은 값을 써야 비교가 성립합니다.


def open_image(row, max_edge=IMAGE_MAX_EDGE):
    """원본 노트북 open_image와 동일: RGB 변환 후 긴 변을 max_edge로 제한."""
    from PIL import Image
    with Image.open(DATA_DIR / str(row["path"])) as im:
        im = im.convert("RGB")
        if max_edge is not None and max(im.size) > max_edge:
            scale = max_edge / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.Resampling.LANCZOS)
        return im


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def load_val(run_dir=BASELINE_RUN):
    """기준 실험이 저장한 검증 분할을 그대로 읽고, ID 해시로 동일 분할임을 확인합니다.

    sklearn 버전이 달라져도 재계산하지 않으므로 분할이 어긋날 수 없습니다.
    """
    run_dir = Path(run_dir)
    val_path = run_dir / "train_half_val.csv"
    if not val_path.is_file():
        raise FileNotFoundError(
            f"기준 분할이 없습니다: {val_path}. 먼저 실험 A 노트북의 분할 셀을 실행하세요.")
    val_df = pd.read_csv(val_path, keep_default_na=False)
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        expected = json.loads(config_path.read_text(encoding="utf-8")).get("validation_ids_sha256")
        actual = ids_sha256(val_df.id.tolist())
        if expected and expected != actual:
            raise ValueError(f"검증 분할이 기준과 다릅니다: {actual} != {expected}")
        print(f"검증 분할 해시 일치: {actual[:16]}…")
    missing = [p for p in val_df.path if not (DATA_DIR / p).is_file()]
    if missing:
        raise FileNotFoundError(f"이미지 누락 {len(missing)}개. 예: {missing[:3]}")
    print(f"검증 문항 {len(val_df)}개 / 정답 분포:\n{val_df.answer.value_counts().sort_index()}")
    return val_df


def load_ocr_evidence(run_dir=BASELINE_RUN, mode="ocr_llm"):
    """실험 A가 남긴 stage1.jsonl(Spotting 결과)을 id -> 텍스트로 읽습니다. 실험 C가 사용."""
    path = Path(run_dir) / mode / "stage1.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"OCR 결과가 없습니다: {path}. 실험 A를 먼저 완료하세요.")
    evidence = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            evidence[item["id"]] = item["stage1_raw"]
    empty = sum(1 for v in evidence.values() if not v.strip())
    print(f"OCR 근거 {len(evidence)}건 로드 (빈 결과 {empty}건)")
    return evidence


def save_analysis(pred_df, name, out_dir, extra=None):
    """정확도/macro-F1/혼동행렬/분류리포트를 저장하고 지표 dict를 반환합니다."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    import matplotlib.pyplot as plt

    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(dest / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pred_df.loc[~pred_df["correct"]].to_csv(dest / "validation_errors.csv",
                                            index=False, encoding="utf-8-sig")
    # INVALID를 열로 유지하여 오답이 사라지거나 임의로 a가 되지 않도록 합니다.
    labels = CHOICES + ["INVALID"]
    cm = confusion_matrix(pred_df["answer"], pred_df["pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{k}" for k in labels],
                         columns=[f"pred_{k}" for k in labels])
    cm_df.to_csv(dest / "confusion_matrix.csv", encoding="utf-8-sig")
    class_df = pd.DataFrame(classification_report(
        pred_df["answer"], pred_df["pred"], labels=CHOICES,
        output_dict=True, zero_division=0)).T
    class_df.to_csv(dest / "classification_report.csv", encoding="utf-8-sig")
    score = {
        "experiment": name,
        "validation_rows": len(pred_df),
        "correct": int(pred_df.correct.sum()),
        "accuracy": float(accuracy_score(pred_df.answer, pred_df.pred)),
        "macro_f1": float(f1_score(pred_df.answer, pred_df.pred, labels=CHOICES,
                                   average="macro", zero_division=0)),
        "invalid": int((pred_df.pred == "INVALID").sum()),
        "invalid_rate": float((pred_df.pred == "INVALID").mean()),
        "mean_inference_seconds": float(pred_df.seconds.mean()),
        "total_inference_seconds": float(pred_df.seconds.sum()),
        "validation_ids_sha256": ids_sha256(pred_df.id.tolist()),
    }
    score.update(extra or {})
    (dest / "metrics.json").write_text(json.dumps(score, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[{name}] Accuracy={score['accuracy']:.4f} / macro-F1={score['macro_f1']:.4f} "
          f"/ INVALID={score['invalid']}")
    print(class_df.loc[CHOICES, ["precision", "recall", "f1-score", "support"]].round(3))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm[:4, :], cmap="Blues")
    ax.set_xticks(range(5), labels)
    ax.set_yticks(range(4), CHOICES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} | Accuracy={score['accuracy']:.3f}")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(dest / "confusion_matrix.png", dpi=180)
    plt.show()
    plt.close(fig)
    return score


def _self_test():
    assert parse_choice("c") == "c"
    assert parse_choice("a 또는 b") == "INVALID"
    assert parse_choice("") == "INVALID"
    assert parse_choice("정답: D") == "d"
    assert parse_choice("간판을 보면 ... 정답: c") == "INVALID"   # 엄격 파서는 문장을 거부
    # CoT 파서
    assert parse_choice_cot("a 와 b 를 비교하면...\n정답: c") == "c"
    assert parse_choice_cot("생각: b 인 것 같지만 아니다.\n정답: (D)") == "d"
    assert parse_choice_cot("설명이 길다\n\nb") == "b"
    assert parse_choice_cot("a 또는 b 중 하나") == "INVALID"
    assert parse_choice_cot("") == "INVALID"
    assert parse_choice_cot("정답: a 이지만 정답: b 로 정정") == "b"   # 마지막 선언을 채택
    row = {"question": "Q", "a": "A", "b": "B", "c": "C", "d": "D"}
    assert make_prompt(row).endswith(ANSWER_INSTRUCTION)
    assert make_prompt(row, cot=True).endswith(COT_INSTRUCTION)
    # 질문/선지 본문은 두 모드가 완전히 동일해야 비교가 성립합니다.
    assert make_prompt(row).rsplit(chr(10), 1)[0] == make_prompt(row, cot=True).rsplit(chr(10), 1)[0]
    assert make_prompt(row, consensus=True).endswith(CONSENSUS_INSTRUCTION)
    assert make_prompt(row).rsplit(chr(10), 1)[0] == make_prompt(row, consensus=True).rsplit(chr(10), 1)[0]
    # E3 1단계는 선지가 새면 안 됩니다.
    read = make_read_prompt(row)
    assert read.endswith(READ_INSTRUCTION)
    for key in CHOICES:
        assert f"{key}. {row[key]}" not in read, "1단계 프롬프트에 선지가 들어갔습니다"
    # E3 2단계는 전사문만 앞에 붙고 질문/선지 본문은 E1과 동일해야 합니다.
    match = make_match_prompt(row, "테스트 전사")
    assert "테스트 전사" in match and match.endswith(MATCH_INSTRUCTION)
    body = lambda s: s.split("질문: ", 1)[1].rsplit(chr(10), 1)[0]
    assert body(match) == body(make_prompt(row)), "2단계 질문/선지 본문이 E1과 다릅니다"
    print("공통 프롬프트/파서 OK (엄격 + CoT + E3 2단계)")


if __name__ == "__main__":
    _self_test()
